"""Prezentáció (PDF, 16:9, 8 dia, kb. 4,5 perc)."""
import json
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from pathlib import Path
R = Path(__file__).resolve().parent.parent; FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("D", FD + "DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("B", FD + "DejaVuSans-Bold.ttf"))
o = json.load(open(R / "results/summary.json")); G = o["generation"]; RP = o["retrieval"]
pc = lambda x: f"{100*x:.0f}%"
W, H = 960, 540; NAVY = colors.HexColor("#1f3b73"); ORG = colors.HexColor("#c28a1e")
c = canvas.Canvas(str(R / "docs/RAG_kiberjog_prezentacio.pdf"), pagesize=(W, H))
c.setTitle("KiberJog-RAG prezentáció")
n = [0]


def slide(title, sec):
    if n[0]: c.showPage()
    n[0] += 1; c.setFillColor(colors.white); c.rect(0, 0, W, H, 0, 1)
    c.setFillColor(NAVY); c.rect(0, H - 6, W, 6, 0, 1)
    c.setFont("B", 30); c.drawString(50, H - 75, title)
    c.setFont("D", 11); c.setFillColor(colors.grey); c.drawRightString(W - 40, 22, f"{n[0]} / 8 · {sec}")


def text(lines, x=50, y=H - 130, size=19, gap=34):
    for l in lines:
        c.setFillColor(colors.HexColor("#222222")); c.setFont("B" if l.startswith("*") else "D", size)
        c.drawString(x, y, l.lstrip("*")); y -= gap


def pic(name, x, y, w):
    im = ImageReader(str(R / "figs" / name)); iw, ih = im.getSize(); c.drawImage(im, x, y, w, w * ih / iw, mask="auto")


def big(x, y, num, lab, col=NAVY):
    c.setFillColor(col); c.setFont("B", 46); c.drawCentredString(x, y, num); c.setFillColor(colors.HexColor("#444444")); c.setFont("D", 14); c.drawCentredString(x, y - 24, lab)


b0, s1, s3, s4 = (G[k] for k in ("B0_keresés_nélkül", "S1_bm25", "S3_hibrid", "S4_hibrid+rerank"))
slide("Kérdés-válasz rendszer jogszabályokhoz", "Bevezetés")
text(["Magyarul feltett kérdés → válasz + melyik cikkből, §-ból", "", "Forrás: NIS2-irányelv, DORA-rendelet,", "2024. évi LXIX. törvény (kiberbiztonság)", "", "Miért? A nyelvi modellek jogi kérdésre", "magabiztosan, de gyakran rossz helyre hivatkozva válaszolnak.", "", "Soltész Dorka · Farkas Attila · 2026. ősz"], size=21)
slide("Adat: 3 jogszabály, 100 tesztkérdés", "Adat")
text(["• 1175 bekezdésszintű darab (NIS2: 258, DORA: 386, Kibertv.: 531)", "• Darabolás: egy bekezdés + fejléc (pl. „NIS2 – 23. cikk (4) bek.”)", "• 100 kérdés, AI-val készítve és ellenőrizve (nem manuálisan):", "    40 tény · 20 hatály · 15 több hely · 15 jogszabályok közötti", "    10 nem megválaszolható, (helyes: „nem tudok válaszolni”)", "• Egy kérdés megoldása: a helyes bekezdés megtalálása"], size=19)
slide("Hogyan működik?", "Módszer")
pic("architecture.png", 100, 110, 760)
text(["Qwen3.5-9B saját gépen · BGE-M3 · BM25 (magyar szótövezés) · Qdrant · bge-reranker-v2-m3"], y=70, size=15)
slide("Mihez mérek?", "Alapvonal")
text(["*B0  Qwen keresés nélkül (fejből válaszol)", "*B1  Csak kulcsszavas (BM25) keresés + Qwen", "*S3  Hibrid keresés (jelentés + kulcsszó) + Qwen", "*S4  Hibrid + reranker + Qwen (teljes rendszer)", "", "Ugyanaz a modell, ugyanaz az utasítás, 5 szövegrész.", "Mérés: találati pontosság (Recall@5, MRR) és a válasz", "helyessége, forráshűsége, hivatkozása (nagy modell pontoz,", "30 válasz kézi ellenőrzése: 90% egyezés, κ = 0,77)."], size=20)
slide("Eredmény 1: a keresés pontossága", "Eredmény")
pic("retrieval.png", 40, 95, 620)
big(790, 330, pc(RP["S1_bm25"]["R@5"]["mean"]) + " → " + pc(RP["S4_hibrid+rerank"]["R@5"]["mean"]), "Recall@5: BM25 → hibrid + reranker")
text(["+13 százalékpont,", "statisztikailag is biztos"], x=700, y=250, size=14, gap=22)
slide("Eredmény 2: a válaszok minősége", "Eredmény")
big(180, 380, pc(b0["helyes_arany"]), "helyes válasz: keresés nélkül", colors.HexColor("#999999"))
big(400, 380, pc(s1["helyes_arany"]), "BM25-RAG", colors.HexColor("#c0504d"))
big(620, 380, pc(s3["helyes_arany"]), "hibrid RAG", NAVY)
big(820, 380, pc(s4["helyes_arany"]), "hibrid + reranker", NAVY)
text([f"• Keresés nélkül a hivatkozások {pc(b0['cit_halluc'])}-a nem létező helyre mutat, a nem megválaszolható", "  kérdések felére kitalált választ ad. RAG-gal: 10/10 helyes elutasítás (fő konfigurációk).", f"• Hibrid RAG-nál a válaszok {pc(s3['forrashu'])}-a forráshű (minden állítás a szövegrészekből jön)."], y=250, size=16, gap=28)
slide("Abláció: darabolás és reranker", "Eredmény")
pic("ablation.png", 40, 130, 520)
text(["• Bekezdésszintű darabolás: legjobb", "  sorrend (MRR) és hivatkozás", "• Fix ablak: hivatkozás-pontosság", "  csak 26% (több bekezdést kever)", "• Reranker: mindenhol javítja az MRR-t", "• Meglepetés: a reranker után a válaszok", f"  gyengébbek ({pc(s4['helyes_arany'])} vs {pc(s3['helyes_arany'])}):", "  kiszorítja a szomszédos bekezdéseket", f"  8 szövegrésszel: {pc(G['S4_hibrid+rerank_k8']['helyes_arany'])}"], x=585, y=H - 135, size=14, gap=26)
slide("Tanulságok", "Összegzés")
text(["• A keresés nélküli modell jogi kérdésre megbízhatatlan", "• A hibrid keresés + forrásmegjelölés: 4% → 87% helyes válasz", "• A jó keresési metrika nem garantálja a jó választ", "• Nehéz: jogszabályok közötti, több bekezdéses kérdések", "• Korlát: 100 kérdés, LLM-bíráló, egy futás", "", "Kód: github.com/SolteszDorka/MIprojekt"], size=21)
c.save()
