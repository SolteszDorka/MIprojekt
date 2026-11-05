"""Szerkeszthető PPTX a prezentációhoz (python-pptx)."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
R = Path(__file__).resolve().parent.parent
o = json.load(open(R / "results/summary.json")); G = o["generation"]; RP = o["retrieval"]
pc = lambda x: f"{100*x:.0f}%"
b0, s1, s3, s4 = (G[k] for k in ("B0_keresés_nélkül", "S1_bm25", "S3_hibrid", "S4_hibrid+rerank"))
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
NAVY = RGBColor(0x1f, 0x3b, 0x73)


def slide(title, lines, img=None, img_w=7, side=None):
    below = None
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(.6), Inches(.35), Inches(12), Inches(.9)).text_frame
    t.text = title; r = t.paragraphs[0].runs[0]; r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = NAVY
    x, w = .6, 12
    if img:
        pic = s.shapes.add_picture(str(R / "figs" / img), Inches(.6), Inches(1.4), width=Inches(img_w))
        if img_w >= 11: x, w, below = .6, 12, 1.4 + pic.height / 914400 + .2
        else: x, w = img_w + .9, 13.333 - img_w - 1.5
    if lines:
        top = below or (1.5 if not img else 1.7)
        tf = s.shapes.add_textbox(Inches(x), Inches(top), Inches(w), Inches(5)).text_frame; tf.word_wrap = True
        for i, l in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = l; p.font.size = Pt(22 if not img else (16 if below else 18)); p.space_after = Pt(8)
    return s


slide("Kérdés-válasz rendszer jogszabályokhoz", ["Magyarul feltett kérdés → válasz + melyik cikkből, §-ból", "Forrás: NIS2-irányelv, DORA-rendelet, 2024. évi LXIX. törvény (kiberbiztonság)",
      "Miért? A nyelvi modellek jogi kérdésre magabiztosan, de gyakran rossz helyre hivatkozva válaszolnak.", "Soltész Dorka · Farkas Attila · 2026. ősz"])
slide("Adat: 3 jogszabály, 100 tesztkérdés", ["1175 bekezdésszintű darab (NIS2: 258, DORA: 386, Kibertv.: 531)", "Darabolás: egy bekezdés + fejléc (pl. „NIS2 – 23. cikk (4) bek.”)",
      "100 kérdés, AI-val készítve és ellenőrizve (nem manuálisan): 40 tény · 20 hatály · 15 több hely · 15 jogszabályok közötti · 10 nem megválaszolható",
      "Feladat: a helyes bekezdés megtalálása és a válasz megfogalmazása"])
slide("Hogyan működik?", ["Qwen3.5-9B saját gépen · BGE-M3 · BM25 (magyar szótövezés) · Qdrant · bge-reranker-v2-m3"], img="architecture.png", img_w=12)
slide("Mihez mérek?", ["B0  Qwen keresés nélkül (fejből válaszol)", "B1  Csak kulcsszavas (BM25) keresés + Qwen", "S3  Hibrid keresés (jelentés + kulcsszó) + Qwen", "S4  Hibrid + reranker + Qwen (teljes rendszer)",
      "Ugyanaz a modell, ugyanaz az utasítás, 5 szövegrész.", "Mérés: Recall@5, MRR; válasz helyessége, forráshűsége, hivatkozása (nagy modell pontoz; 30 válasz ellenőrzése: 90% egyezés, κ = 0,77)."])
slide("Eredmény 1: a keresés pontossága", [f"Recall@5: {pc(RP['S1_bm25']['R@5']['mean'])} (BM25) → {pc(RP['S4_hibrid+rerank']['R@5']['mean'])} (hibrid + reranker)", "+13 százalékpont, statisztikailag is biztos"], img="retrieval.png", img_w=7.5)
slide("Eredmény 2: a válaszok minősége", [f"Helyes válasz: keresés nélkül {pc(b0['helyes_arany'])} · BM25-RAG {pc(s1['helyes_arany'])} · hibrid RAG {pc(s3['helyes_arany'])} · hibrid + reranker {pc(s4['helyes_arany'])}",
      f"Keresés nélkül a hivatkozások {pc(b0['cit_halluc'])}-a nem létező helyre mutat; a nem megválaszolható kérdések felére kitalált választ ad. RAG-gal: 10/10 helyes elutasítás (fő konfigurációk).",
      f"Hibrid RAG-nál a válaszok {pc(s3['forrashu'])}-a forráshű."], img="generation.png", img_w=12)
slide("Abláció: darabolás és reranker", ["Bekezdésszintű darabolás: legjobb sorrend (MRR) és hivatkozás", "Fix ablak: hivatkozás-pontosság csak 26%", "Reranker: mindenhol javítja az MRR-t",
      f"Meglepetés: reranker után a válaszok gyengébbek ({pc(s4['helyes_arany'])} vs {pc(s3['helyes_arany'])}): kiszorítja a szomszédos bekezdéseket; 8 szövegrésszel: {pc(G['S4_hibrid+rerank_k8']['helyes_arany'])}"], img="ablation.png", img_w=6.5)
slide("Tanulságok", ["A keresés nélküli modell jogi kérdésre megbízhatatlan", "Hibrid keresés + forrásmegjelölés: 4% → 87% helyes válasz", "A jó keresési metrika nem garantálja a jó választ",
      "Nehéz: jogszabályok közötti, több bekezdéses kérdések", "Korlát: 100 kérdés, AI-val ellenőrzött kérdéssor, LLM-bíráló, egy futás", "Kód: github.com/SolteszDorka/MIprojekt"])
prs.save(str(R / "docs/RAG_kiberjog_prezentacio.pptx"))
