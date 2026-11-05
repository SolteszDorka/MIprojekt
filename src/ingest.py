"""Jogszabályok feldolgozása: HTML -> darabok (A cikk-, B bekezdés-, C fix ablakos).
Futtatás: python src/ingest.py   (kimenet: data/chunks_{A,B,C}.jsonl)"""
import json, re, warnings
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
D = Path(__file__).resolve().parent.parent / "data"
WS = re.compile(r"[\s   ]+")
MAXC = 1100  # B: ennél hosszabb bekezdést pontonként bontunk (kb. 400 token)


def norm(t): return WS.sub(" ", t).strip()


def hiv(src, cikk, bek, unit):
    s = f"{src} {cikk}. {unit}"
    return s + (f" ({bek}) bek." if bek else "")


def split_points(intro, points):
    """Hosszú bekezdés: a bevezető mondat minden darab elejére kerül."""
    out, cur = [], []
    for p in points:
        if cur and len(intro) + sum(map(len, cur)) + len(p) > MAXC:
            out.append(intro + " " + " ".join(cur)); cur = []
        cur.append(p)
    if cur: out.append(intro + " " + " ".join(cur))
    return out


def mk(src, unit, cikk, bek, cim, tipus, intro, points):
    body = intro + (" " + " ".join(points) if points else "")
    parts = split_points(intro, points) if len(body) > MAXC and len(points) > 1 else [body]
    return [dict(forras=src, cikk=cikk, bekezdes=bek, cim=cim, tipus=tipus,
                 hivatkozas=hiv(src, cikk, bek, unit), part=i, text=p)
            for i, p in enumerate(parts)]


def mk_annex(src, name, text):
    return dict(forras=src, cikk=None, bekezdes=None, cim=name, tipus="melleklet",
                hivatkozas=f"{src} {name}", part=0, text=norm(text))


def parse_eu(path, src):
    s = BeautifulSoup(open(path, "rb"), "lxml")
    out = []
    for r in s.find_all("div", id=re.compile(r"^rct_\d+$")):
        n = r["id"][4:]
        out.append(dict(forras=src, cikk=None, bekezdes=n, cim="", tipus="preambulum",
                        hivatkozas=f"{src} ({n}) preambulumbekezdés", part=0, text=norm(r.get_text(" "))))
    for a in s.find_all("div", id=re.compile(r"^art_\d+$")):
        num = norm(a.find("p", class_="oj-ti-art").get_text()).split(".")[0]
        t = a.find("div", class_="eli-title")
        cim = norm(t.get_text(" ")) if t else ""
        pars = a.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"))
        if not pars:  # egybekezdéses cikk
            ps = [p for p in a.find_all("p", class_="oj-normal") if not p.find_parent("table")]
            intro = norm(" ".join(p.get_text(" ") for p in ps))
            pts = [norm(tb.get_text(" ")) for tb in a.find_all("table")]
            out += mk(src, "cikk", num, None, cim, "rendelkezo", intro, pts)
            continue
        for d in pars:
            b = str(int(d["id"].split(".")[1]))
            ps = [p for p in d.find_all("p") if not p.find_parent("table")]
            intro = re.sub(r"^\(\d+\)\s*", "", norm(" ".join(p.get_text(" ") for p in ps)))
            pts = [norm(tb.get_text(" ")) for tb in d.find_all("table")]
            out += mk(src, "cikk", num, b, cim, "rendelkezo", intro, pts)
    for a in s.find_all("div", id=re.compile(r"^anx_")):
        tt = a.find("p", class_="oj-doc-ti")
        name = norm(tt.get_text()) if tt else a["id"]
        rows = [norm(t.get_text(" ")) for t in a.find_all(["p", "tr"]) if t.name == "tr" or not t.find_parent("tr")]
        cur = ""
        for r in filter(None, rows):
            if len(cur) + len(r) > MAXC and cur:
                out.append(mk_annex(src, name, cur)); cur = ""
            cur += " " + r
        if cur.strip(): out.append(mk_annex(src, name, cur))
    return out


SZ = re.compile(r"^(\d+(?:/[A-Z])?)\.\s?§\s*")
BEK = re.compile(r"^\((\d+[a-z]?)\)\s*")


def parse_hu(path, src):
    s = BeautifulSoup(open(path, encoding="utf-8"), "lxml")
    for x in s.select("sup.fnSup, .footnote"): x.decompose()
    j = s.find(class_="jogszabaly")
    out, cim, par, mell, sz = [], "", None, None, None
    for pr in j.find_all("div", class_="preambulumbekezdes"):
        t = norm(pr.get_text(" ")); n = re.match(r"\[(\d+)\]", t)
        out.append(dict(forras=src, cikk=None, bekezdes=n.group(1) if n else "", cim="", tipus="preambulum",
                        hivatkozas=f"{src} preambulum", part=0, text=t))

    def flush():
        nonlocal par
        if par and par["intro"]:
            out.extend(mk(src, "§", par["sz"], par["bek"], par["cim"], "rendelkezo", par["intro"], par["pts"]))
        par = None

    for c in [c for c in j.children if getattr(c, "name", None) == "div"]:
        cl = c.get("class") or []
        t = norm(c.get_text(" "))
        if "alcim" in cl: cim = t
        elif "mellekletCimke" in cl: flush(); mell = t
        elif "mellekletPont" in cl and mell:
            if out and out[-1]["tipus"] == "melleklet" and out[-1]["cim"] == mell and len(out[-1]["text"]) + len(t) < MAXC:
                out[-1]["text"] += " " + t
            else: out.append(mk_annex(src, mell, t))
        elif ("bekezdesNyito" in cl or "szakasz" in cl) and not mell:
            if not t: continue
            flush()
            m = SZ.match(t)
            if m: sz, t = m.group(1), t[m.end():]
            if sz is None: continue
            b = BEK.match(t)
            par = dict(sz=sz, bek=b.group(1) if b else None, cim=cim, intro=BEK.sub("", t, 1), pts=[])
        elif par and any(k in cl for k in ("betusPontNyito", "szamosPontNyito", "ketbetusAlPont", "betusAlPont", "bekezdesZaro", "pontZaro")):
            par["pts"].append(t)
    flush()
    return out


def window(chs, size=150, step=100):
    """C változat: fix ablakok a normatív szövegen (szó alapú, ≈300 token)."""
    out = []
    for src in sorted({c["forras"] for c in chs}):
        words, owner = [], []
        for c in chs:
            if c["forras"] != src or c["tipus"] != "rendelkezo": continue
            w = c["text"].split(); words += w; owner += [(c["cikk"], c["bekezdes"])] * len(w)
        for i in range(0, max(1, len(words) - size + step), step):
            seg = words[i:i + size]
            if len(seg) < 30: break
            atoms = sorted(set(owner[i:i + size]), key=str)
            out.append(dict(forras=src, tipus="ablak", cim="", hivatkozas=f"{src} szövegrész",
                            atoms=[list(a) for a in atoms], text=" ".join(seg), cikk=None, bekezdes=None, part=i))
    return out


def article_level(chs):
    g = {}
    for c in chs:
        if c["tipus"] == "rendelkezo": g.setdefault((c["forras"], c["cikk"]), []).append(c)
    out = []
    for (src, cikk), cs in g.items():
        text = " ".join((f"({c['bekezdes']}) " if c["bekezdes"] else "") + c["text"] for c in cs)
        out.append(dict(forras=src, cikk=cikk, bekezdes=None, cim=cs[0]["cim"], tipus="rendelkezo", part=0,
                        hivatkozas=hiv(src, cikk, None, "§" if src == "Kibertv." else "cikk"), text=text,
                        atoms=[list(a) for a in sorted({(c["cikk"], c["bekezdes"]) for c in cs}, key=str)]))
    return out + [c for c in chs if c["tipus"] != "rendelkezo"]


def finish(chs, prefix):
    for i, c in enumerate(chs):
        c["id"] = f"{prefix}{i:05d}"
        c.setdefault("atoms", [[c["cikk"], c["bekezdes"]]] if c["tipus"] == "rendelkezo" else [])
        if c["tipus"] == "ablak": c["szoveg"] = f"{c['forras']}: {c['text']}"
        else:
            h = " – ".join(x for x in [c["forras"], c["hivatkozas"].replace(c["forras"] + " ", "", 1), c["cim"]] if x)
            c["szoveg"] = f"{h}: {c['text']}"
    return chs


if __name__ == "__main__":
    R = D / "raw"
    B = (parse_eu(R / "32022L2555.html", "NIS2") + parse_eu(R / "32022R2554.html", "DORA")
         + parse_hu(R / "kibertv.html", "Kibertv."))
    B = finish(B, "B")
    A = finish(article_level(B), "A")
    C = finish(window(B) + [c for c in B if c["tipus"] != "rendelkezo"], "C")
    for name, ch in zip("ABC", (A, B, C)):
        with open(D / f"chunks_{name}.jsonl", "w") as f:
            for c in ch: f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(name, len(ch), "darab")
    from collections import Counter
    print(sorted(Counter((c["forras"], c["tipus"]) for c in B).items()))
