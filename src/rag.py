"""Keresés (BM25 / dense / hibrid RRF / + reranker) Qdrantban, és válaszgenerálás.
A kollekciók neve: kiberjog_{A,B,C}. A Qdrant más kollekcióihoz nem nyúlunk."""
import json, math, re, zlib
from collections import Counter
from pathlib import Path
import numpy as np, Stemmer
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
QDRANT = "http://localhost:6333"
LLM_URL = "http://localhost:8191/v1"
STEM = Stemmer.Stemmer("hungarian")
TOK = re.compile(r"\w+", re.U)
K1, BM = 1.2, 0.75


def terms(text):
    return STEM.stemWords([w.lower() for w in TOK.findall(text) if not w.isdigit() or len(w) > 1])


def tid(t): return zlib.crc32(t.encode()) & 0x7FFFFFFF


def doc_sparse(text, avg):
    tf = Counter(terms(text)); n = sum(tf.values())
    agg = {}
    for t, f in tf.items():
        agg[tid(t)] = agg.get(tid(t), 0) + f * (K1 + 1) / (f + K1 * (1 - BM + BM * n / avg))
    return models.SparseVector(indices=list(agg), values=list(agg.values()))


def query_sparse(text):
    ids = sorted({tid(t) for t in terms(text)})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


def load_chunks(v): return [json.loads(l) for l in open(DATA / f"chunks_{v}.jsonl")]


_emb = _rr = None


def embedder():
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer("BAAI/bge-m3", device="cuda"); _emb.half(); _emb.max_seq_length = 1024
    return _emb


def reranker():
    global _rr
    if _rr is None:
        from sentence_transformers import CrossEncoder
        _rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=1024,
                           model_kwargs={"torch_dtype": "float16"})
    return _rr


def build_index(v, qc=None):
    qc = qc or QdrantClient(QDRANT)
    ch = load_chunks(v); name = f"kiberjog_{v}"
    cache = DATA / f"emb_{v}.npy"
    if cache.exists(): E = np.load(cache)
    else:
        E = embedder().encode([c["szoveg"] for c in ch], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        np.save(cache, E)
    avg = np.mean([len(terms(c["szoveg"])) for c in ch])
    if qc.collection_exists(name): qc.delete_collection(name)
    qc.create_collection(name, vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                         sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    pts = [models.PointStruct(id=i, payload=c, vector={"dense": E[i].tolist(), "bm25": doc_sparse(c["szoveg"], avg)})
           for i, c in enumerate(ch)]
    for i in range(0, len(pts), 200): qc.upsert(name, pts[i:i + 200])
    return len(pts)


class Retriever:
    def __init__(self, v="B"):
        self.qc, self.name = QdrantClient(QDRANT), f"kiberjog_{v}"

    def search(self, q, mode="hybrid", rerank=True, n=10, pool=20):
        """mode: bm25 | dense | hybrid. Visszatér: payload-lista rangsorban."""
        fl = models.Filter(must_not=[models.FieldCondition(key="tipus", match=models.MatchValue(value="preambulum"))])
        sp = models.Prefetch(query=query_sparse(q), using="bm25", limit=50, filter=fl)
        if mode == "bm25":
            r = self.qc.query_points(self.name, query=sp.query, using="bm25", limit=pool if rerank else n, query_filter=fl)
        else:
            dv = embedder().encode([q], normalize_embeddings=True)[0].tolist()
            dp = models.Prefetch(query=dv, using="dense", limit=50, filter=fl)
            if mode == "dense":
                r = self.qc.query_points(self.name, query=dv, using="dense", limit=pool if rerank else n, query_filter=fl)
            else:
                r = self.qc.query_points(self.name, prefetch=[dp, sp], query=models.FusionQuery(fusion=models.Fusion.RRF),
                                         limit=pool if rerank else n)
        hits = [p.payload for p in r.points]
        if rerank and hits:
            sc = reranker().predict([(q, h["szoveg"]) for h in hits], batch_size=16)
            hits = [h for _, h in sorted(zip(sc, hits), key=lambda x: -x[0])]
        return hits[:n]


# ---------- generálás ----------
SYS_RAG = ("Kiberbiztonsági jogszabályokról kérdeznek. Csak az alábbi szövegrészek alapján válaszolj, magyarul, "
           "legfeljebb 5 mondatban. Minden állítás után add meg a forrást szögletes zárójelben, pl. [NIS2 23. cikk (4) bek.] "
           "vagy [Kibertv. 2. § (1) bek.]. A hivatkozást a szövegrész fejlécéből másold. "
           "Ha a szövegrészek alapján nem lehet válaszolni, ezt írd: „A megadott jogszabályhelyek alapján erre nem tudok válaszolni.”")
SYS_B0 = ("Kiberbiztonsági jogszabályokról (NIS2-irányelv, DORA-rendelet, a 2024. évi LXIX. törvény) kérdeznek. "
          "A saját tudásod alapján válaszolj, magyarul, legfeljebb 5 mondatban. Minden állítás után add meg a jogszabályhelyet "
          "szögletes zárójelben, pl. [NIS2 23. cikk (4) bek.] vagy [Kibertv. 2. § (1) bek.]. "
          "Ha nem tudod, ezt írd: „A megadott jogszabályhelyek alapján erre nem tudok válaszolni.”")
REFUSAL = "erre nem tudok válaszolni"


def ctx_block(hits):
    return "\n\n".join(f"[{h['hivatkozas']}] {h['text']}" for h in hits)


def generate(q, hits=None, url=LLM_URL, max_tokens=500):
    import requests
    if hits is None: msgs = [{"role": "system", "content": SYS_B0}, {"role": "user", "content": f"Kérdés: {q}"}]
    else: msgs = [{"role": "system", "content": SYS_RAG},
                  {"role": "user", "content": f"Szövegrészek:\n{ctx_block(hits)}\n\nKérdés: {q}"}]
    last = None
    for _ in range(4):
        r = requests.post(url + "/chat/completions", json=dict(model="m", messages=msgs, temperature=0, max_tokens=max_tokens,
                          chat_template_kwargs={"enable_thinking": False}), timeout=300).json()
        if "choices" in r and r["choices"][0]["message"]["content"].strip(): return r["choices"][0]["message"]["content"].strip()
        last = r
    raise RuntimeError(str(last)[:300])


CIT = re.compile(r"(NIS2|DORA|Kibertv\.?)\s*(\d+(?:/[A-Z])?)\.\s*(?:cikk|§)(?:\s*\((\d+[a-z]?)\)\s*bek)?", re.I)


def citations(ans):
    out = []
    for m in re.finditer(r"\[([^\]]+)\]", ans):
        for c in CIT.finditer(m.group(1)):
            s = c.group(1); s = "Kibertv." if s.lower().startswith("kib") else s.upper()
            out.append((s, c.group(2), c.group(3)))
    return list(dict.fromkeys(out))


def atom_match(atoms, forras, gold):
    """gold: [forras, cikk, bek|None]; atoms: [[cikk, bek],..]"""
    if forras != gold[0]: return False
    return any(c == gold[1] and (gold[2] is None or b is None or b == gold[2]) for c, b in atoms)


def hit_gold(h, golds): return any(atom_match(h["atoms"], h["forras"], g) for g in golds)


def cit_ok(c, golds): return any(c[0] == g[0] and c[1] == g[1] and (g[2] is None or c[2] is None or c[2] == g[2]) for g in golds)
