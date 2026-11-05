"""Kiértékelés: keresés (minden konfiguráció), válaszgenerálás (Qwen3.5-9B), bírálat (nagy modell).
Újraindítható: a kész részeket kihagyja.  python src/run_eval.py [retr|gen|judge]"""
import json, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from math import log2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import rag
from rag import *

R = ROOT / "results"; R.mkdir(exist_ok=True)
TEST = [json.loads(l) for l in open(ROOT / "eval" / "testset.jsonl")]
JUDGE = "http://localhost:8082/v1"
# név: (darabolás, mód, reranker)
RET = {"S1_bm25": ("B", "bm25", False), "S2_dense": ("B", "dense", False), "S3_hibrid": ("B", "hybrid", False),
       "S4_hibrid+rerank": ("B", "hybrid", True),
       "A_hibrid": ("A", "hybrid", False), "A_hibrid+rerank": ("A", "hybrid", True),
       "C_hibrid": ("C", "hybrid", False), "C_hibrid+rerank": ("C", "hybrid", True)}
GEN = ["B0_keresés_nélkül", "S1_bm25", "S3_hibrid", "S4_hibrid+rerank", "A_hibrid+rerank", "C_hibrid+rerank", "S4_hibrid+rerank_k8"]
ALL = {}  # chunk id -> chunk


def label(h):
    if h["tipus"] != "ablak": return h["hivatkozas"]
    u = lambda f: "§" if f == "Kibertv." else "cikk"
    return "; ".join(f"{h['forras']} {c}. {u(h['forras'])}" + (f" ({b}) bek." if b else "") for c, b in h["atoms"][:4])


def ctxb(hits): return "\n\n".join(f"[{label(h)}] {h['text']}" for h in hits)


rag.ctx_block = ctxb


def retrieval():
    out = {}
    for name, (v, m, rr) in RET.items():
        r = Retriever(v)
        out[name] = {}
        for t in TEST:
            out[name][t["id"]] = r.search(t["kerdes"], m, rr, 10)
        print("keresés kész", name, flush=True)
    json.dump(out, open(R / "retrieval.json", "w"), ensure_ascii=False)


def gen():
    ret = json.load(open(R / "retrieval.json"))
    for name in GEN:
        p = R / f"answers_{name}.jsonl"
        done = {json.loads(l)["id"] for l in open(p)} if p.exists() else set()
        def one(t):
            hits = None if name.startswith("B0") else ret[name.split("_k")[0] if name.endswith("_k8") else name][t["id"]][:8 if name.endswith("_k8") else 5]
            return dict(id=t["id"], valasz=generate(t["kerdes"], hits), kontextus=[label(h) for h in hits] if hits else [])
        todo = [t for t in TEST if t["id"] not in done]
        with ThreadPoolExecutor(3) as ex, open(p, "a") as f:
            for a in ex.map(one, todo):
                f.write(json.dumps(a, ensure_ascii=False) + "\n"); f.flush()
        print("generálás kész", name, flush=True)


JP = """Egy jogi kérdés-válasz rendszer válaszát értékeled. Légy szigorú és következetes.
KÉRDÉS: {q}
HELYES VÁLASZ (referencia): {gold}
A HELYES JOGSZABÁLYHELY SZÖVEGE:
{gp}
{ctx}
A RENDSZER VÁLASZA:
{ans}

Értékeld:
- "helyesseg": 1-5. 5 = tartalmilag teljesen egyezik a referenciával; 4 = lényegében helyes, apró hiány; 3 = részben helyes; 2 = főleg hibás; 1 = hibás, kitalált vagy elutasítás.
- "forrashu": 1, ha a válasz minden tartalmi állítása alátámasztható a megadott szövegrészekkel (a rendszernek átadott kontextussal), 0 ha van benne a kontextusban nem szereplő állítás. {fh_note}
Kimenet kizárólag JSON: {{"helyesseg": n, "forrashu": 0 vagy 1}}"""


def judge():
    chunks = {c["id"]: c for v in "B" for c in load_chunks(v)}
    tests = {t["id"]: t for t in TEST}
    for name in GEN:
        a = [json.loads(l) for l in open(R / f"answers_{name}.jsonl")]
        p = R / f"judge_{name}.jsonl"
        done = {json.loads(l)["id"] for l in open(p)} if p.exists() else set()
        def one(x):
            t = tests[x["id"]]
            if t["kategoria"] == "nincs":
                return dict(id=x["id"], helyesseg=5 if REFUSAL in x["valasz"] else 1, forrashu=1)
            gp = "\n".join(chunks[i]["szoveg"] for i in t["arany_chunkok"])
            ctx = ("A RENDSZERNEK ÁTADOTT KONTEXTUS:\n" + "\n".join(f"[{h}]" for h in x["kontextus"]) + "\n(részletes szöveg lent)\n" ) if False else ""
            if x["kontextus"]:
                full = ret_text(name, x["id"])
                ctx = "A RENDSZERNEK ÁTADOTT KONTEXTUS:\n" + full
                note = ""
            else:
                ctx = ""; note = "(Kontextus nélküli válasz: a 'forrashu' legyen 1, ha az állítások összhangban vannak a helyes jogszabályhely szövegével, különben 0.)"
            prompt = JP.format(q=t["kerdes"], gold=t["arany_valasz"], gp=gp, ctx=ctx, ans=x["valasz"], fh_note=note)
            for _ in range(3):
                try:
                    r = requests.post(JUDGE + "/chat/completions", json=dict(model="m", temperature=0, max_tokens=60,
                        messages=[{"role": "user", "content": prompt}], chat_template_kwargs={"enable_thinking": False}), timeout=300).json()
                    j = json.loads(re.search(r"\{.*?\}", r["choices"][0]["message"]["content"], re.S).group(0))
                    return dict(id=x["id"], helyesseg=int(j["helyesseg"]), forrashu=int(j["forrashu"]))
                except Exception as e: err = e
            return dict(id=x["id"], helyesseg=None, forrashu=None, hiba=str(err))
        todo = [x for x in a if x["id"] not in done]
        with ThreadPoolExecutor(3) as ex, open(p, "a") as f:
            for j in ex.map(one, todo):
                f.write(json.dumps(j) + "\n"); f.flush()
        print("bírálat kész", name, flush=True)


_ret = None
def ret_text(name, qid):
    global _ret
    if _ret is None: _ret = json.load(open(R / "retrieval.json"))
    return ctxb(_ret[name.split("_k")[0] if name.endswith("_k8") else name][qid][:8 if name.endswith("_k8") else 5])


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["retr", "gen", "judge"]):
        {"retr": retrieval, "gen": gen, "judge": judge}[st]()
