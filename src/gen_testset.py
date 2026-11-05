"""Tesztkérdések félautomata generálása (nagy lokális modellel), kézi átnézésre.
Kimenet: eval/testset_raw.jsonl  -> a válogatás után eval/testset.jsonl"""
import json, random, re, sys, requests
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from rag import load_chunks, ROOT

BIG = "http://localhost:8082/v1"
random.seed(7)
B = [c for c in load_chunks("B") if c["tipus"] == "rendelkezo" and 250 < len(c["text"]) < 1600]
NAME = {"NIS2": "a NIS2-irányelv", "DORA": "a DORA-rendelet", "Kibertv.": "a magyar kiberbiztonsági törvény (2024. évi LXIX.)"}

RULES = ("Hétköznapi, természetes magyar nyelven fogalmazz, ahogy egy cég compliance-felelőse kérdezne. "
         "NE használd a szövegrész jellegzetes szavait, írd át más szavakkal. A kérdés legyen önállóan érthető "
         "(ne hivatkozz 'a szövegre'), és ha több jogszabály is szóba jöhet, nevezd meg, melyikről kérdezel. "
         "Csak olyat kérdezz, amire a megadott szövegrész(ek) egyértelműen választ adnak. "
         "A válasz 1-3 mondat, tényszerű, a szövegből. "
         'Kimenet kizárólag JSON: {"kerdes": "...", "arany_valasz": "..."}')


def ask(prompt):
    r = requests.post(BIG + "/chat/completions", json=dict(model="m", temperature=0.3, max_tokens=500,
                      messages=[{"role": "user", "content": prompt}], chat_template_kwargs={"enable_thinking": False}), timeout=300).json()
    t = r["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{.*\}", t, re.S).group(0))


def gold(c): return [c["forras"], c["cikk"], c["bekezdes"]]


def passage(c): return f"[{c['hivatkozas']}] {c['text']}"


def make(kat, cs, extra, qid):
    p = f"Jogszabályi szövegrész(ek) {NAME[cs[0]['forras']] if len(cs) == 1 else ''}:\n" + "\n\n".join(map(passage, cs)) + f"\n\nFeladat: {extra}\n{RULES}"
    try: j = ask(p)
    except Exception as e: print("hiba", e); return None
    return dict(id=qid, kategoria=kat, kerdes=j["kerdes"].strip(), arany_valasz=j["arany_valasz"].strip(),
                arany_helyek=[gold(c) for c in cs], arany_chunkok=[c["id"] for c in cs],
                jogszabaly="+".join(sorted({c["forras"] for c in cs})))


def pick(pred, n):
    pool = [c for c in B if pred(c)]; random.shuffle(pool)
    out, per = [], {}
    for c in pool:  # forrásonként kiegyensúlyozva, cikkenként legfeljebb 1
        if per.get((c["forras"], c["cikk"])) or sum(1 for o in out if o["forras"] == c["forras"]) >= n // 3 + 2: continue
        per[(c["forras"], c["cikk"])] = 1; out.append(c)
        if len(out) >= n: break
    return out


def main():
    items = []
    num = re.compile(r"\d+\s*(óra|nap|hónap|%|százalék|EUR|euró|forint|Ft|év|munkanap)|\d{4,}", re.I)
    for c in pick(lambda c: num.search(c["text"]), 52):
        items.append(make("teny", [c], "Írj EGY konkrét ténykérdést (határidő, összeg, fogalom, kötelezett), amire a válasz egy adat vagy rövid megállapítás.", f"t{len(items):03d}"))
    for c in pick(lambda c: re.search(r"hatály|alkalmaz|nem terjed ki|kivéve|minősül|mikro|kisvállalkozás|tartozik", c["text"]) and c["forras"] != "x", 30):
        items.append(make("hatály", [c], "Írj egy hatályra vonatkozó kérdést: kire vonatkozik a szabály, vagy egy konkrét típusú szervezetre vonatkozik-e (igen/nem + indoklás).", f"t{len(items):03d}"))
    # több helyet érintő: azonos cikk két különböző bekezdése
    arts = {}
    for c in B:
        if c["bekezdes"]: arts.setdefault((c["forras"], c["cikk"]), {}).setdefault(c["bekezdes"], c)
    keys = [k for k, v in arts.items() if len(v) >= 3]; random.shuffle(keys)
    for k in keys[:22]:
        ps = list(arts[k].values()); a, b = random.sample(ps, 2)
        items.append(make("tobb", [a, b], "Írj EGY kérdést, amelynek teljes megválaszolásához MINDKÉT szövegrészre szükség van (pl. 'milyen lépésekből áll', 'mit kell tennie és mikorra').", f"t{len(items):03d}"))
    # jogszabályok közötti: dense hasonlóság (NIS2/DORA <-> Kibertv.)
    E = np.load(ROOT / "data" / "emb_B.npy"); allB = load_chunks("B"); idx = {c["id"]: i for i, c in enumerate(allB)}
    hu = [c for c in B if c["forras"] == "Kibertv."]; eu = [c for c in B if c["forras"] != "Kibertv."]
    S = np.array([E[idx[c["id"]]] for c in hu]) @ np.array([E[idx[c["id"]]] for c in eu]).T
    pairs = sorted(((S[i, j], i, j) for i in range(len(hu)) for j in range(len(eu))), reverse=True)
    used, got = set(), 0
    for s, i, j in pairs:
        if hu[i]["cikk"] in used or eu[j]["cikk"] in used or s < .62: continue
        used |= {hu[i]["cikk"], eu[j]["cikk"]}
        items.append(make("kozott", [eu[j], hu[i]], "Írj EGY összehasonlító kérdést, amely mindkét jogszabályra vonatkozik (pl. 'ugyanaz-e a szabály a NIS2-ben és a magyar törvényben', 'mit ír elő a két jogszabály').", f"t{len(items):03d}"))
        got += 1
        if got >= 22: break
    items = [i for i in items if i]
    with open(ROOT / "eval" / "testset_raw.jsonl", "w") as f:
        for i in items: f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(len(items))


if __name__ == "__main__": main()
