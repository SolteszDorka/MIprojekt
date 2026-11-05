"""Metrikák, bootstrap CI, táblák és ábrák. python src/report.py -> results/summary.json, figs/*.png"""
import json, sys, random
from math import log2
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from rag import ROOT, load_chunks, hit_gold, cit_ok, citations, REFUSAL
from run_eval import RET, GEN

R = ROOT / "results"; F = ROOT / "figs"; F.mkdir(exist_ok=True)
T = [json.loads(l) for l in open(ROOT / "eval/testset.jsonl")]
ANS = [t for t in T if t["arany_helyek"]]
RT = json.load(open(R / "retrieval.json"))
random.seed(1); rng = np.random.default_rng(1)


def ranks(sys_, t, k=10):
    return [hit_gold(h, t["arany_helyek"]) for h in RT[sys_][t["id"]][:k]]


def metrics(sys_):
    rec = {k: [] for k in (1, 3, 5, 10)}; mrr, ndcg = [], []
    for t in ANS:
        r = ranks(sys_, t)
        for k in rec: rec[k].append(float(any(r[:k])))
        mrr.append(next((1 / (i + 1) for i, x in enumerate(r) if x), 0.0))
        idcg = sum(1 / log2(i + 2) for i in range(min(len(t["arany_helyek"]), 10)))
        cov, dcg = set(), 0.0  # egy arany helyet csak egyszer jutalmazunk
        for i, h in enumerate(RT[sys_][t["id"]][:10]):
            new = {j for j, g in enumerate(t["arany_helyek"]) if hit_gold(h, [g])} - cov
            if new: dcg += 1 / log2(i + 2); cov |= new
        ndcg.append(dcg / idcg)
    return {**{f"R@{k}": v for k, v in rec.items()}, "MRR@10": mrr, "nDCG@10": ndcg}


def ci(x, n=5000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pci(a, b, n=5000):  # páros bootstrap: a - b
    d = np.asarray(a, float) - np.asarray(b, float); m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


corpus = {(c["forras"], c["cikk"], c["bekezdes"]) for c in load_chunks("B") if c["tipus"] == "rendelkezo"}
arts = {(a, b) for a, b, _ in corpus}


def exists(c): return (c[0], c[1]) in arts and (c[2] is None or (c[0], c[1], c[2]) in corpus)


out = {"retrieval": {}, "generation": {}, "paired": {}}
M = {s: metrics(s) for s in RET}
for s, m in M.items():
    out["retrieval"][s] = {k: {"mean": float(np.mean(v)), "ci": ci(v)} for k, v in m.items()}
for a, b in [("S4_hibrid+rerank", "S1_bm25"), ("S4_hibrid+rerank", "S3_hibrid"), ("S3_hibrid", "S1_bm25")]:
    for k in ("R@5", "MRR@10"):
        out["paired"][f"{a} - {b} {k}"] = pci(M[a][k], M[b][k])

G = {}
for s in GEN:
    ans = {json.loads(l)["id"]: json.loads(l) for l in open(R / f"answers_{s}.jsonl")}
    jd = {json.loads(l)["id"]: json.loads(l) for l in open(R / f"judge_{s}.jsonl")}
    rows = []
    for t in T:
        a, j = ans[t["id"]], jd[t["id"]]
        cs = citations(a["valasz"]); refused = REFUSAL in a["valasz"]
        r = dict(id=t["id"], kat=t["kategoria"], helyes=j["helyesseg"], forrashu=j["forrashu"], refused=refused, n_cit=len(cs))
        if t["arany_helyek"]:
            r["cit_ok"] = sum(cit_ok(c, t["arany_helyek"]) for c in cs)
            r["cit_halluc"] = sum(not exists(c) for c in cs)
            r["cit_any"] = float(r["cit_ok"] > 0)
            if a["kontextus"]:
                ctxs = " | ".join(a["kontextus"])
                r["cit_not_in_ctx"] = sum(f"{c[0]} {c[1]}." not in ctxs for c in cs)
        rows.append(r)
    G[s] = rows
    A = [r for r in rows if r["kat"] != "nincs"]; N = [r for r in rows if r["kat"] == "nincs"]
    ok = [float(r["helyes"] >= 4) for r in A if r["helyes"] is not None]
    tot_c = sum(r["n_cit"] for r in A)
    g = dict(helyes_atlag=float(np.mean([r["helyes"] for r in A if r["helyes"]])),
             helyes_arany=float(np.mean(ok)), helyes_ci=ci(ok),
             forrashu=float(np.mean([r["forrashu"] for r in A if r["forrashu"] is not None])),
             cit_precision=sum(r["cit_ok"] for r in A) / max(1, tot_c),
             cit_any=float(np.mean([r["cit_any"] for r in A])),
             cit_halluc=sum(r["cit_halluc"] for r in A) / max(1, tot_c),
             cit_per_valasz=tot_c / len(A),
             refusal_unans=float(np.mean([r["refused"] for r in N])),
             false_refusal=float(np.mean([r["refused"] for r in A])))
    if s != "B0_keresés_nélkül":
        g["cit_not_in_ctx"] = sum(r.get("cit_not_in_ctx", 0) for r in A) / max(1, tot_c)
    by = {}
    for r in A: by.setdefault(r["kat"], []).append(float(r["helyes"] >= 4))
    g["kategoria_helyes"] = {k: float(np.mean(v)) for k, v in by.items()}
    out["generation"][s] = g
for a, b in [("S4_hibrid+rerank", "B0_keresés_nélkül"), ("S4_hibrid+rerank", "S1_bm25"), ("S3_hibrid", "S1_bm25")]:
    for key in ("helyes", "forrashu"):
        xa = [float((r[key] >= 4) if key == "helyes" else r[key]) for r in G[a] if r["kat"] != "nincs"]
        xb = [float((r[key] >= 4) if key == "helyes" else r[key]) for r in G[b] if r["kat"] != "nincs"]
        out["paired"][f"gen {a} - {b} {key}"] = pci(xa, xb)
# kategória szerinti R@5 (S4)
cat = {}
for t, v in zip(ANS, M["S4_hibrid+rerank"]["R@5"]): cat.setdefault(t["kategoria"], []).append(v)
out["retrieval_S4_kategoria_R5"] = {k: float(np.mean(v)) for k, v in cat.items()}
json.dump(out, open(R / "summary.json", "w"), ensure_ascii=False, indent=1)
json.dump(G, open(R / "per_question.json", "w"), ensure_ascii=False)

# ---- ábrák ----
C = {"bm": "#c0504d", "de": "#8c8c8c", "hy": "#4f81bd", "re": "#1f3b73", "b0": "#bfbfbf"}
names = ["S1_bm25", "S2_dense", "S3_hibrid", "S4_hibrid+rerank"]; lab = ["BM25", "Dense\n(BGE-M3)", "Hibrid\n(RRF)", "Hibrid +\nreranker"]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for a, k in zip(ax, ("R@5", "MRR@10")):
    v = [out["retrieval"][n][k]["mean"] for n in names]
    e = [[v[i] - out["retrieval"][n][k]["ci"][0] for i, n in enumerate(names)], [out["retrieval"][n][k]["ci"][1] - v[i] for i, n in enumerate(names)]]
    a.bar(lab, v, color=[C["bm"], C["de"], C["hy"], C["re"]], yerr=e, capsize=3)
    a.set_ylim(0.5, 1); a.set_title(k); a.spines[["top", "right"]].set_visible(False)
    for i, x in enumerate(v): a.text(i, x + .035, f"{x:.2f}", ha="center", fontsize=8)
    a.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(F / "retrieval.png", dpi=200); plt.close()

gn = ["B0_keresés_nélkül", "S1_bm25", "S3_hibrid", "S4_hibrid+rerank"]; gl = ["Keresés\nnélkül (B0)", "BM25-RAG\n(B1)", "Hibrid", "Hibrid +\nreranker"]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for a, (k, t) in zip(ax, [("helyes_arany", "Helyes válasz (≥4/5)"), ("forrashu", "Forráshű válasz"), ("cit_precision", "Hivatkozás-pontosság")]):
    v = [out["generation"][n][k] for n in gn]
    if k == "forrashu": v[0] = 0  # B0-nál nem értelmezhető
    a.bar(gl, v, color=[C["b0"], C["bm"], C["hy"], C["re"]]); a.set_ylim(0, 1.08); a.set_title(t, fontsize=10)
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7.5)
    for i, x in enumerate(v): a.text(i, x + .02, "n. a." if (k == "forrashu" and i == 0) else f"{x:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(F / "generation.png", dpi=200); plt.close()

an = ["A_hibrid", "A_hibrid+rerank", "S3_hibrid", "S4_hibrid+rerank", "C_hibrid", "C_hibrid+rerank"]
fig, ax = plt.subplots(figsize=(7.5, 3.2)); w = .38
for j, (k, col) in enumerate([("R@5", C["hy"]), ("MRR@10", C["re"])]):
    ax.bar(np.arange(6) + (j - .5) * w, [out["retrieval"][n][k]["mean"] for n in an], w, label=k, color=col)
ax.set_xticks(range(6)); ax.set_xticklabels(["A cikk\nreranker nélkül", "A cikk\n+ reranker", "B bekezdés\nreranker nélkül", "B bekezdés\n+ reranker", "C ablak\nreranker nélkül", "C ablak\n+ reranker"], fontsize=7.5)
ax.set_ylim(0.5, 1); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(F / "ablation.png", dpi=200); plt.close()
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
