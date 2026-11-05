"""Dokumentáció (PDF) összeállítása a results/ mappa tényleges méréseiből."""
import json, sys, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, Preformatted)

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
FD = "/usr/share/fonts/truetype/dejavu/"
for n, f in [("D", "DejaVuSans.ttf"), ("D-B", "DejaVuSans-Bold.ttf"), ("D-I", "DejaVuSans-Oblique.ttf"), ("M", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFontFamily("D", normal="D", bold="D-B", italic="D-I", boldItalic="D-B")
o = json.load(open(ROOT / "results/summary.json")); st = json.load(open(ROOT / "results/chunk_stats.json"))
mc = json.load(open(ROOT / "results/manual_check.json")); PQ = json.load(open(ROOT / "results/per_question.json"))
RP = o["retrieval"]; GN = o["generation"]; PR = o["paired"]
NAVY = colors.HexColor("#1f3b73"); GREY = colors.HexColor("#f2f4f8")
base = dict(fontName="D", fontSize=9.2, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
P = ParagraphStyle("p", **base)
H1 = ParagraphStyle("h1", fontName="D-B", fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, textColor=NAVY)
H2 = ParagraphStyle("h2", fontName="D-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=NAVY)
CAP = ParagraphStyle("c", fontName="D-I", fontSize=8, leading=10.5, alignment=1, spaceAfter=8, textColor=colors.HexColor("#444444"))
BUL = ParagraphStyle("b", parent=P, leftIndent=14, bulletIndent=3, spaceAfter=2)
SM = ParagraphStyle("s", fontName="D", fontSize=8, leading=10.5)
SMB = ParagraphStyle("sb", parent=SM, fontName="D-B")
CODE = ParagraphStyle("code", fontName="M", fontSize=7.4, leading=9.6, backColor=GREY, borderPadding=5, spaceAfter=8)


def pc(x, d=1): return f"{100 * x:.{d}f}".replace(".", ",") + "%"
def f2(x): return f"{x:.3f}".replace(".", ",")
def pp(x): return f"{100 * x:+.1f}".replace(".", ",").replace("-", "−") + " százalékpont"
def cir(c): return f"[{100 * c[0]:.1f}; {100 * c[1]:.1f}]".replace(".", ",")
def p(t, s=P): return Paragraph(t, s)
def bl(items): return [Paragraph(i, BUL, bulletText="•") for i in items]


def tbl(rows, widths, hl=None, head=1):
    rows = [[Paragraph(str(c), SMB if (i < head or j == 0 and False) else SM) for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=head)
    s = [("BACKGROUND", (0, 0), (-1, head - 1), GREY), ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#c8cdd8")),
         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if hl is not None: s.append(("BACKGROUND", (0, hl), (-1, hl), colors.HexColor("#e3ecfa")))
    t.setStyle(TableStyle(s)); return t


def img(name, w):
    from PIL import Image as I
    a, b = I.open(ROOT / "figs" / name).size
    return Image(str(ROOT / "figs" / name), width=w * cm, height=w * cm * b / a)


def bootstrap(a, b, n=5000):
    r = np.random.default_rng(3); d = np.array(a, float) - np.array(b, float)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1); return d.mean(), [np.percentile(m, 2.5), np.percentile(m, 97.5)]


def corr(s): return [float(r["helyes"] >= 4) for r in PQ[s] if r["kat"] != "nincs"]


s3s4 = bootstrap(corr("S3_hibrid"), corr("S4_hibrid+rerank")); k8 = bootstrap(corr("S4_hibrid+rerank_k8"), corr("S4_hibrid+rerank"))
S = []
add = S.append
# ---------------- címoldal ----------------
add(Spacer(1, 2.2 * cm))
add(p("FÉLÉVES FELADAT – DOKUMENTÁCIÓ", ParagraphStyle("x", fontName="D-B", fontSize=10, textColor=colors.HexColor("#c28a1e"))))
add(p("Kérdés-válasz rendszer kiberbiztonsági jogszabályokhoz", ParagraphStyle("t", fontName="D-B", fontSize=22, leading=27, textColor=NAVY, spaceAfter=8)))
add(p("Visszakereséssel kiegészített szöveggenerálás (RAG) a NIS2-irányelv, a DORA-rendelet és a magyar kiberbiztonsági törvény alapján, Qwen3.5-9B modellel", ParagraphStyle("st", fontName="D", fontSize=11.5, leading=16, alignment=0, spaceAfter=18)))
add(tbl([["Készítette", "Soltész Dorka"], ["Neptun-kód", "[kitöltendő]"], ["Tantárgy / oktató", "[kitöltendő] / Farkas Attila"],
         ["Feladattípus", "RAG-rendszer saját dokumentumokra, kész Transformer-komponensekből"], ["Félév", "2026/27. tanév, őszi félév"],
         ["A kód elérhetősége", "GitHub: <b>https://github.com/SolteszDorka/MIprojekt</b> (teljes, futtatható projekt)"],
         ["Dátum", "2026. november 4."]], [4.2, 12.3], head=0))
add(Spacer(1, .8 * cm))
add(p("Összefoglaló", H2))
add(p(f"A féléves feladatban egy olyan rendszert építettem, amely magyarul feltett kérdésekre három kiberbiztonsági jogszabály szövege alapján válaszol, és minden válaszban megadja, melyik cikkből vagy paragrafusból dolgozott. A forrásokat bekezdésenként bontottam fel, és Qdrant adatbázisban kétféleképpen indexeltem: jelentésalapú (BGE-M3) és kulcsszavas (BM25) kereséshez. A kérdéshez illő részeket a két keresés összefésülésével, majd egy reranker modellel választja ki a rendszer, a választ pedig a saját gépen futó Qwen3.5-9B fogalmazza meg."))
add(p(f"A rendszert két egyszerűbb alapvonalhoz mértem: a keresés nélkül válaszoló Qwen-modellhez és egy csak BM25-öt használó RAG-hoz, 100 tesztkérdésen. <b>Eredmények:</b> a keresés nélküli modell a válaszok mindössze {pc(GN['B0_keresés_nélkül']['helyes_arany'], 0)}-át adta helyesen, és a hivatkozásainak {pc(GN['B0_keresés_nélkül']['cit_halluc'], 0)}-a nem létező jogszabályhelyre mutatott. A BM25-RAG {pc(GN['S1_bm25']['helyes_arany'], 0)}-ot, a hibrid keresésre épülő RAG {pc(GN['S3_hibrid']['helyes_arany'], 0)}-ot ért el. A találati pontosság (Recall@5) a BM25 {pc(RP['S1_bm25']['R@5']['mean'])}-os értékéről a hibrid + reranker összeállítással {pc(RP['S4_hibrid+rerank']['R@5']['mean'])}-ra nőtt. Egy váratlan eredmény: a reranker javította a találatok sorrendjét, de a végső válaszok minőségét nem, mert kiszorította a szomszédos bekezdéseket (6.3 pont)."))
add(PageBreak())
# ---------------- 1 ----------------
add(p("1. A feladat", H1)); add(p("1.1 Háttér", H2))
add(p("Az Európai Unió 2022 decemberében két, a kiberbiztonságot érintő jogszabályt fogadott el. A NIS2-irányelv ((EU) 2022/2555) a kiemelten kritikus és egyéb kritikus ágazatokban működő szervezeteknek ír elő kockázatkezelési és incidensbejelentési kötelezettségeket. Magyarországon ezt elsősorban a Magyarország kiberbiztonságáról szóló 2024. évi LXIX. törvény (a továbbiakban: Kibertv.) ülteti át. A DORA-rendelet ((EU) 2022/2554) a pénzügyi szervezetek digitális működési rezilienciáját szabályozza, és 2025. január 17-től közvetlenül alkalmazandó."))
add(p("Aki egy szervezetnél ezekkel a szabályokkal foglalkozik, gyakran egészen konkrét kérdésre keres választ: mennyi idő alatt kell bejelenteni egy incidenst, mely szervezetekre vonatkozik a törvény. A válasz többnyire egy-két bekezdésben benne van a szövegben, de ezeket meg kell találni három, részben átfedő jogszabályban. Egy általános célú nyelvi modell gyorsan ad választ, de jogi kérdésekben pontatlan, és hajlamos nem létező cikkszámra hivatkozni (ezt a 6.2 pont számokkal is megmutatja). Ezért a cél az, hogy a rendszer csak a jogszabályszövegre támaszkodjon, és minden állításnál jelezze a forrást, így a felhasználó ellenőrizni tudja a választ."))
add(p("1.2 A feladat megfogalmazása", H2))
add(KeepTogether(bl(["<b>Bemenet:</b> egy magyar nyelvű, természetes megfogalmazású kérdés.",
    "<b>Kimenet:</b> rövid magyar válasz és az alapjául szolgáló jogszabályhelyek, pl. „NIS2 23. cikk (4) bek.” vagy „Kibertv. 1. § (1) bek.”.",
    "<b>Elutasítás:</b> ha a három jogszabály alapján nem lehet válaszolni, a rendszernek ezt ki kell mondania."])))
add(p("A feladat típusa visszakereséssel kiegészített szöveggenerálás (retrieval-augmented generation, RAG): a rendszer először megkeresi a kérdéshez illő szövegrészeket, majd ezeket adja át a nyelvi modellnek a válasz megírásához. A modell így nem a saját „emlékezetéből” dolgozik, hanem a konkrét szövegből."))
add(p("1.3 Hatókör", H2))
add(KeepTogether(bl(["Csak a három megnevezett jogszabály szerepel; a végrehajtási rendeletek (pl. 418/2024. Korm. rendelet) és a DORA technikai standardjai nincsenek benne.",
    "Rögzített időállapot: az EU-jogszabályok a Hivatalos Lapban megjelent szövegei (letöltés: 2026. 11. 04.), a Kibertv. a Nemzeti Jogszabálytár 2026. 10. 01-i hatályos állapota.",
    "A rendszer nem ad jogi tanácsot, csak a releváns szöveg megtalálásában és összefoglalásában segít."])))
# ---------------- 2 ----------------
add(p("2. Az adatok", H1)); add(p("2.1 Források és feldolgozás", H2))
add(tbl([["Jogszabály", "Forrás", "Mit tartalmaz a korpusz"],
         ["NIS2-irányelv, (EU) 2022/2555", "EUR-Lex / Kiadványhivatal, magyar XHTML", "46 cikk, 258 bekezdésszintű darab, I–II. melléklet (29 darab), 144 preambulumbekezdés"],
         ["DORA-rendelet, (EU) 2022/2554", "EUR-Lex / Kiadványhivatal, magyar XHTML", "64 cikk, 386 bekezdésszintű darab, 106 preambulumbekezdés"],
         ["2024. évi LXIX. törvény (Kibertv.)", "Nemzeti Jogszabálytár (njt.hu), HTML", "102 §, 531 bekezdésszintű darab, 3 melléklet, 4 preambulumbekezdés"]],
        [4.6, 5.2, 6.7]))
add(p("1. táblázat. A felhasznált jogszabályok", CAP))
add(p("Mindhárom forrás szerkezetes HTML, ezért nem PDF-ből, hanem HTML-ből dolgoztam (BeautifulSoup). Az uniós szövegben a cikkek, bekezdések és pontok azonosítóval ellátott elemek, a njt.hu szövegében a § és a bekezdés számát a szöveg elejéről olvassa ki a program. A lábjegyzeteket eltávolítottam. A szöveg szerzői jogi szempontból szabadon felhasználható: az EUR-Lex jogi dokumentumai forrásmegjelöléssel újrahasznosíthatók (2011/833/EU határozat), a magyar jogszabályszöveg pedig hivatalos közlemény, amelyet a szerzői jog nem véd (Szjt. 1. § (4))."))
add(p("2.2 Darabolás", H2))
add(p("Alapbeállításban egy darab egy bekezdés a hozzá tartozó pontokkal együtt. Ha egy bekezdés 1100 karakternél hosszabb, pontonként bontottam tovább, és minden darab elejére odaírtam a bekezdés bevezető mondatát. Minden darab elé rövid fejléc kerül („NIS2 – 23. cikk (4) bek. – Jelentéstételi kötelezettség: …”), amely a keresést is segíti, és a modell ebből másolja a hivatkozást. A preambulumbekezdések külön típusként bekerültek az adatbázisba, de a keresés kizárja őket (Qdrant-szűrő), mert a válaszban a rendelkező részre kell hivatkozni."))
add(tbl([["Változat", "Egy darab", "Darabszám", "Átlagos hossz (token)"],
         ["A – cikkszintű", "egy teljes cikk vagy §", f"{st['A']['db']}", f"{st['A']['atlag']:.0f} (max. {st['A']['max']})"],
         ["B – bekezdésszintű (alap)", "egy bekezdés, szükség esetén pontcsoport", f"{st['B']['db']}", f"{st['B']['atlag']:.0f} (max. {st['B']['max']})"],
         ["C – fix ablak", "150 szó (≈ 280 token) ablak, 50 szó átfedéssel, szerkezettől függetlenül", f"{st['C']['db']}", f"{st['C']['atlag']:.0f} (max. {st['C']['max']})"]],
        [4, 6.6, 2.3, 3.6], hl=2))
add(p("2. táblázat. A vizsgált darabolási módok (a darabszám csak a rendelkező részt és az ablakokat számolja; a tokenszám a BGE-M3 tokenizerével készült)", CAP))
add(p("2.3 A tesztkérdések", H2))
add(p("A kiértékeléshez 100 magyar nyelvű tesztkérdés készült. <b>A kérdéssort AI segítségével állítottam össze és ellenőriztem, manuális (emberi) ellenőrzésen nem esett át.</b> A folyamat: egy nagy helyi modell (Qwen3.5-122B-A10B) egy-egy kiválasztott jogszabályi szövegrészből hétköznapi nyelven megfogalmazott kérdést és referenciaválaszt írt, az arany jogszabályhellyel együtt; a 126 generált kérdésből 90-et egy AI-asszisztens választott ki (az érthetetlen, túl részletkérdés jellegű vagy több helyes választ megengedő kérdések kimaradtak); a 10 megválaszolhatatlan kérdést szintén AI-asszisztens írta. Minden kérdés arany jogszabályhelye bekezdésszintű; a keresés találatnak számít, ha a visszaadott darab tartalmazza az arany bekezdést."))
add(tbl([["Kategória", "Példa", "db"],
         ["Ténykérdés (határidő, összeg, fogalom)", "Mennyi ideig rendelhet el ideiglenes letiltást a Központ vezetője?", "40"],
         ["Hatály: kire vonatkozik", "A DORA mentesít-e kis méretű befektetési vállalkozásokat az IKT-kockázatkezelés alól?", "20"],
         ["Több helyet érintő kérdés", "Milyen feltételeket kell tartalmaznia a kritikus funkciót érintő szerződésnek?", "15"],
         ["Jogszabályok közötti kérdés", "A NIS2 és a magyar törvény is előírja-e az információcsere-csoporthoz való csatlakozás bejelentését?", "15"],
         ["Nem megválaszolható", "Hány órán belül kell bejelenteni az adatvédelmi incidenst a NAIH-nak a GDPR szerint?", "10"]], [5, 10.4, 1.1]))
add(p("3. táblázat. A tesztkérdések összetétele (a teljes lista: eval/testset.jsonl)", CAP))
# ---------------- 3 ----------------
add(p("3. A rendszer felépítése", H1))
add(p("Az indexelést egyszer, előre végzem el: a jogszabályokat feldarabolom, a darabokat vektorokká alakítom, és a Qdrant adatbázisban tárolom. A második rész minden kérdésnél lefut: megkeresi a kérdéshez illő darabokat, kiválasztja közülük a legjobb ötöt, és ezekből a nyelvi modell megírja a választ."))
add(img("architecture.png", 16)); add(p("1. ábra. Fent az előre elvégzett indexelés, lent egy kérdés feldolgozása", CAP))
add(p("3.1 Indexelés és hibrid keresés", H2))
add(KeepTogether(bl([
    "<b>Sűrű (dense) vektor:</b> BAAI/bge-m3 (kb. 568 millió paraméteres, többnyelvű enkóder, 1024 dimenzió, koszinusz-hasonlóság). Akkor is megtalálja a releváns részt, ha a kérdés más szavakat használ, mint a jogszabály.",
    "<b>Ritka (sparse) BM25-vektor:</b> a szavakat Snowball magyar szótövezővel (PyStemmer) alakítottam szótővé, így a „bejelentés”, „bejelentését”, „bejelentéssel” ugyanaz a kifejezés. A BM25 dokumentum-oldali súlyát (k1 = 1,2; b = 0,75) magam számoltam, az IDF-et a Qdrant ritka vektorára beállított <font name='M'>Modifier.IDF</font> adja.",
    "<b>Hibrid keresés:</b> mindkét keresés az 50 legjobb darabot adja vissza, a két listát reciprokrang-fúzióval (RRF, k = 60) fésüli össze a Qdrant Query API-ja: RRF(d) = Σ 1 / (k + rang(d)). Az első 20 elem megy tovább a rerankerhez.",
    "<b>Reranker:</b> BAAI/bge-reranker-v2-m3 (cross-encoder): a kérdést és a jelöltet együtt olvassa, és relevanciapontszámot ad. A 20 jelöltből az első 5 kerül a nyelvi modell elé."])))
add(p("3.2 Válaszgenerálás", H2))
add(p("A választ a Qwen3.5-9B állítja elő (9 milliárd paraméteres dekóder Transformer). Az eredeti Qwen/Qwen3.5-9B modell unsloth által készített 8 bites kvantált GGUF változatát (Q8_0, 9,5 GB) használtam, llama.cpp szerverrel, helyben, a gondolkodási (thinking) mód kikapcsolásával és 0 hőmérséklettel, hogy a futások megismételhetők legyenek. Az utasítás szövege a repóban (src/rag.py) található; lényege: csak a megadott szövegrészek alapján válaszolj, legfeljebb 5 mondatban, minden állítás után add meg a forrást szögletes zárójelben, és ha nem lehet válaszolni, mondd ki: „A megadott jogszabályhelyek alapján erre nem tudok válaszolni.” A hivatkozásokat a program a válaszból reguláris kifejezéssel nyeri ki, és ellenőrzi."))
add(tbl([["Modell", "Típus", "Szerep"],
         ["BAAI/bge-m3", "enkóder (XLM-RoBERTa alapú), bi-encoder", "a kérdést és a darabokat külön-külön vektorrá alakítja"],
         ["BAAI/bge-reranker-v2-m3", "enkóder, cross-encoder", "a kérdés–darab párokat együtt olvasva újrarangsorol"],
         ["Qwen3.5-9B (Q8_0)", "dekóder (causal), kb. 9 Mrd paraméter", "a szövegrészekből magyar választ és hivatkozásokat generál"],
         ["Qwen3.5-122B-A10B (Q4)", "dekóder, helyi szerveren", "csak a teszthalmaz készítéséhez és a válaszok bírálatához"]], [4.2, 5.6, 6.7]))
add(p("4. táblázat. A Transformer-alapú komponensek (finomhangolás nincs, a feladat kész komponensek láncba fűzése és mérése)", CAP))
add(p("3.3 Futtatási környezet", H2))
add(p("Minden helyben futott: Ubuntu, NVIDIA RTX PRO 6000 Blackwell GPU (a memória nagy része más szolgáltatásoké, a rendszernek kb. 15 GB jutott), Qdrant 1.20.0 Docker-konténerben (kiberjog_A/B/C gyűjtemények), BGE-M3 és reranker fp16-ban, a Qwen3.5-9B egy külön llama.cpp-konténerben. Egy kérdés megválaszolása (keresés + reranker + generálás) néhány másodperc."))
# ---------------- 4 ----------------
add(p("4. Mihez mérem: az alapvonalak", H1))
add(p("Mindhárom változat ugyanazt a modellt (Qwen3.5-9B), ugyanazt az utasítást és ugyanannyi (5) szövegrészt használja; a különbség csak abban van, hogyan kerülnek a szövegrészek a modell elé."))
add(tbl([["Változat", "Keresés", "Mit mutat meg"],
         ["B0 – Qwen keresés nélkül", "nincs; a modell fejből válaszol és hivatkozik", "mennyit tud a modell magától; mennyire hivatkozik rossz helyre"],
         ["B1 – BM25-RAG (S1)", "csak kulcsszavas keresés, első 5 találat", "mit ad egy egyszerű, olcsó keresés"],
         ["Dense (S2)", "csak BGE-M3, első 5 találat", "mennyit tesz hozzá a BM25 a hibridhez"],
         ["Hibrid (S3)", "BGE-M3 + BM25, RRF, első 5", "reranker nélküli hibrid"],
         ["Teljes rendszer (S4)", "hibrid + reranker, első 5", "a javasolt megoldás"]], [4.4, 5.6, 6.5], hl=5))
add(p("5. táblázat. Az összehasonlított változatok", CAP))
# ---------------- 5 ----------------
add(p("5. A kiértékelés módja", H1))
add(p("5.1 A keresés mérése", H2))
add(KeepTogether(bl([
    "<b>Recall@k</b> (k = 1, 3, 5, 10): azon kérdések aránya, amelyeknél az első k találat között van legalább egy helyes jogszabályhely.",
    "<b>MRR@10:</b> az első helyes találat helyezésének reciproka, kérdésekre átlagolva (1/rang, 0 ha a top 10-ben nincs találat).",
    "<b>nDCG@10:</b> a rangsor minősége; több helyes hely esetén azt is méri, hány van elöl (egy helyes helyet csak egyszer jutalmaz).",
    "A 10 nem megválaszolható kérdés nem számít bele, mert ezeknél nincs helyes hely (90 kérdés). A konfidenciaintervallumok 5000 mintás bootstrappel készültek, két rendszer különbségére páros bootstrappel."])))
add(p("5.2 A válaszok mérése", H2))
add(p("A válaszokat egy másik, nagyobb helyi modell (Qwen3.5-122B-A10B) pontozta a referenciaválasz és a helyes jogszabályszöveg ismeretében. A bíráló nem azonos a válaszadó modellel, hogy ne legyen önpreferencia-torzítás. A módszer a következő mutatókat adja:"))
add(tbl([["Mutató", "Hogyan számolom", "Típus"],
         ["Helyes válasz", "a bíráló 1–5-ös pontszáma; ≥ 4 = helyes", "LLM-bíráló"],
         ["Forráshű", "minden állítás alátámasztható-e a modellnek átadott szövegrészekkel (B0-nál nem értelmezhető)", "LLM-bíráló"],
         ["Hivatkozás-pontosság", "a válaszban megadott hivatkozások közül hány esik egybe valamelyik helyes hellyel", "automatikus (regex)"],
         ["Van helyes hivatkozás", "legalább egy helyes hivatkozás van a válaszban", "automatikus"],
         ["Kitalált hivatkozás", "olyan cikk/§/bekezdés, amely a korpuszban nem létezik", "automatikus"],
         ["Elutasítás", "a nem megválaszolható kérdésnél kimondja-e, hogy nem tud válaszolni", "automatikus"]], [3.6, 9.4, 3.5]))
add(p("6. táblázat. A válaszok értékelési szempontjai", CAP))
add(p(f"<b>Kézi ellenőrzés.</b> A bírálat megbízhatóságának méréséhez 30 véletlenül kiválasztott választ (a hat rendszerváltozatból egyenletesen) megpontoztam a bírálótól függetlenül. A „helyes / nem helyes” döntésben az egyezés {pc(mc['egy_kategoria'], 0)} (27/30), Cohen-féle κ = {mc['kappa']:.2f}; a pontos 1–5-ös pontszámban {pc(mc['pontos_egyezes'], 0)}. Az eltérések a határesetekben voltak (pl. olyan válasz, amely kimondja, hogy nem tud válaszolni, de közben a helyes tartalmat is leírja). A kézi pontozást az elkészítés során AI-asszisztens végezte, a szerzői ismétlés javasolt (7. fejezet)."))
add(p("5.3 Ablációk", H2))
add(p("Egyszerre egy elemet változtatok: (a) a darabolás három módja (A, B, C), (b) a reranker ki/be, (c) a modell elé adott szövegrészek száma (5 helyett 8), (d) csak sűrű, csak BM25 és hibrid keresés. A keresési metrikákat mind a nyolc keresési konfigurációra, a válaszokat hat konfigurációra és egy plusz (top-8) változatra is kiszámoltam."))
# ---------------- 6 ----------------
add(p("6. Eredmények", H1)); add(p("6.1 A keresés pontossága", H2))
names = [("S1_bm25", "BM25 (B1)"), ("S2_dense", "Csak BGE-M3 (sűrű)"), ("S3_hibrid", "Hibrid (RRF), reranker nélkül"), ("S4_hibrid+rerank", "Hibrid + reranker (teljes rendszer)")]
rows = [["Keresés", "R@1", "R@3", "R@5", "R@10", "MRR@10", "nDCG@10"]]
for k, n in names: rows.append([n] + [pc(RP[k][m]["mean"]) for m in ("R@1", "R@3", "R@5", "R@10")] + [f2(RP[k]["MRR@10"]["mean"]), f2(RP[k]["nDCG@10"]["mean"])])
add(tbl(rows, [5.7, 1.7, 1.7, 1.7, 1.7, 2, 2], hl=4))
add(p("7. táblázat. Keresési metrikák a 90 megválaszolható kérdésen (bekezdésszintű darabolás)", CAP))
add(img("retrieval.png", 13.5)); add(p("2. ábra. Recall@5 és MRR@10, 95%-os bootstrap konfidenciaintervallummal", CAP))
a = PR["S4_hibrid+rerank - S1_bm25 R@5"]; b = PR["S4_hibrid+rerank - S1_bm25 MRR@10"]; c = PR["S4_hibrid+rerank - S3_hibrid R@5"]; d = PR["S3_hibrid - S1_bm25 R@5"]
add(p(f"A teljes rendszer Recall@5 értéke {pp(a[0])}-tal jobb a BM25-nél (95%-os CI: {cir(a[1])}), az MRR {f2(b[0])}-tal (CI: [{f2(b[1][0])}; {f2(b[1][1])}]). Ez a különbség statisztikailag is biztos. A hibrid keresés önmagában {pp(d[0])}-tal javít a BM25-höz képest. A reranker hozzáadása a hibrid keresés tetejére kisebb, de egyirányú javulást ad (Recall@5: {pp(c[0])}, CI: {cir(c[1])}; MRR@10: {f2(RP['S3_hibrid']['MRR@10']['mean'])} → {f2(RP['S4_hibrid+rerank']['MRR@10']['mean'])}), ez a 90 kérdésen önmagában még nem tekinthető biztosnak, mert a konfidenciaintervallum átfogja a nullát. A sűrű keresés egyedül is erős (R@5 = {pc(RP['S2_dense']['R@5']['mean'])}), a BM25 hozzáadása főleg a rangsor elejét és a Recall@10-et javítja (R@10: {pc(RP['S2_dense']['R@10']['mean'])} → {pc(RP['S3_hibrid']['R@10']['mean'])}). Ez egybevág azzal a várakozással, hogy a magyar toldalékolás miatt a kulcsszavas keresés önmagában gyenge, de a pontos kifejezéseknél hasznos kiegészítés."))
add(p("6.2 A válaszok minősége", H2))
gn = [("B0_keresés_nélkül", "B0 – Qwen keresés nélkül"), ("S1_bm25", "B1 – BM25-RAG"), ("S3_hibrid", "Hibrid, reranker nélkül"), ("S4_hibrid+rerank", "Hibrid + reranker (S4)")]
rows = [["Változat", "Helyes válasz", "Forráshű", "Hiv.-pontosság", "Van helyes hiv.", "Kitalált hiv.", "Elutasít (nincs válasz)"]]
for k, n in gn:
    g = GN[k]; rows.append([n, pc(g["helyes_arany"]), pc(g["forrashu"]) if k != "B0_keresés_nélkül" else "n. a.*", pc(g["cit_precision"]), pc(g["cit_any"]), pc(g["cit_halluc"]), pc(g["refusal_unans"], 0)])
add(tbl(rows, [4.6, 2, 1.9, 2.2, 2.1, 1.8, 1.9], hl=4))
add(p("8. táblázat. A válaszok értékelése (90 megválaszolható kérdés; az utolsó oszlop a 10 megválaszolhatatlan kérdésen). *B0-nál a forráshűség nem értelmezhető, mert a modell nem kap szövegrészeket", CAP))
add(img("generation.png", 16)); add(p("3. ábra. Helyes válaszok, forráshűség és hivatkozás-pontosság a négy fő változatra", CAP))
x = PR["gen S4_hibrid+rerank - B0_keresés_nélkül helyes"]; y = PR["gen S3_hibrid - S1_bm25 helyes"]; z = PR["gen S4_hibrid+rerank - S1_bm25 helyes"]
add(p(f"<b>Keresés nélkül a modell használhatatlan jogi kérdésekre:</b> a válaszok {pc(GN['B0_keresés_nélkül']['helyes_arany'], 0)}-a helyes, a hivatkozások {pc(GN['B0_keresés_nélkül']['cit_halluc'], 0)}-a nem létező jogszabályhelyre mutat, és az összes megadott hivatkozás {pc(GN['B0_keresés_nélkül']['cit_precision'], 1)}-a esik a helyes helyre. A megválaszolhatatlan kérdések felére is kitalált választ adott. Ezzel szemben az öt fő RAG-konfiguráció mind a 10 ilyen kérdést elutasította, és nem hivatkozott nem létező helyre (kivéve az A változatnál 2 hivatkozást; a top-8 változat 9 elutasítást adott a 10-ből)."))
add(p(f"<b>A keresés minősége átível a válaszok minőségébe:</b> a BM25-RAG helyes válaszai {pc(GN['S1_bm25']['helyes_arany'], 0)}-ot, a hibrid keresésű RAG-é {pc(GN['S3_hibrid']['helyes_arany'], 0)}-ot tesznek ki ({pp(y[0])}, CI: {cir(y[1])}). A BM25-nél a modell az esetek {pc(GN['S1_bm25']['false_refusal'], 0)}-ában tévesen elutasította a válaszadást, mert a keresés nem hozta elő a megfelelő részt; a hibridnél ez {pc(GN['S3_hibrid']['false_refusal'], 0)}. A hivatkozás-pontosság (kb. {pc(GN['S3_hibrid']['cit_precision'], 0)}) alacsonynak látszik, de ez főleg abból ered, hogy a modell több szomszédos bekezdésre is hivatkozik, miközben a tesztben egyetlen arany hely szerepel; a „van legalább egy helyes hivatkozás” mutató a hibridnél {pc(GN['S3_hibrid']['cit_any'], 0)}."))
add(p("6.3 A reranker váratlan hatása a válaszokra", H2))
add(p(f"A reranker javította a keresési metrikákat, a válaszok minősége mégis rosszabb lett ({pc(GN['S4_hibrid+rerank']['helyes_arany'])} az S3 {pc(GN['S3_hibrid']['helyes_arany'])}-ával szemben, a különbség {pp(s3s4[0])}, páros CI: {cir(s3s4[1])}). A hibás esetek átnézése megmutatta az okot: a reranker a jelöltek közül a kérdéshez leginkább hasonló szöveget részesíti előnyben, ezért az első ötbe gyakran ugyanannak a hosszú bekezdésnek több darabja kerül be, a szomszédos bekezdések (amelyek a teljes válaszhoz kellenének) pedig kiszorulnak. Többbekezdéses kérdéseknél emiatt a modell hiányos szövegből dolgozik, és gyakrabban utasítja el a választ ({pc(GN['S4_hibrid+rerank']['false_refusal'], 0)} szemben a hibrid {pc(GN['S3_hibrid']['false_refusal'], 0)}-ával). Ha a reranker után nyolc darabot adunk a modellnek, a helyes válaszok aránya {pc(GN['S4_hibrid+rerank_k8']['helyes_arany'])}-ra nő ({pp(k8[0])} az öt darabos változathoz képest, CI: {cir(k8[1])}). Tanulság: a keresési metrika (Recall@5) és a válasz minősége nem mozog együtt automatikusan; a rerankert a modell elé adott szövegrészek számával együtt kell hangolni, és érdemes lenne egy bekezdés darabjait a reranker után összevonni."))
add(p("6.4 Ablációk: darabolás és reranker", H2))
rows = [["Darabolás", "Reranker", "R@5", "MRR@10", "Helyes válasz", "Van helyes hiv.", "Hiv.-pontosság"]]
for ch, nm in (("A", "A – cikkszintű"), ("B", "B – bekezdésszintű"), ("C", "C – fix ablak")):
    for rr in (True, False):
        key = {"B": "S4_hibrid+rerank" if rr else "S3_hibrid"}.get(ch, f"{ch}_hibrid+rerank" if rr else f"{ch}_hibrid")
        gk = key if key in GN else None
        gcells = [pc(GN[gk]["helyes_arany"]), pc(GN[gk]["cit_any"]), pc(GN[gk]["cit_precision"])] if gk else ["–", "–", "–"]
        rows.append([nm, "van" if rr else "nincs", pc(RP[key]["R@5"]["mean"]), f2(RP[key]["MRR@10"]["mean"])] + gcells)
add(tbl(rows, [4, 1.7, 1.6, 1.8, 2.5, 2.6, 2.3]))
add(p("9. táblázat. A darabolás és a reranker hatása (a válaszminőség azoknál a konfigurációknál szerepel, amelyekre lefutott a generálás)", CAP))
add(img("ablation.png", 12.5)); add(p("4. ábra. Darabolási módok és a reranker hatása a keresésre", CAP))
add(KeepTogether(bl([
    f"<b>Darabolás:</b> a keresési mutatók közel vannak egymáshoz (R@5: {pc(RP['A_hibrid+rerank']['R@5']['mean'], 0)}–{pc(RP['S4_hibrid+rerank']['R@5']['mean'], 0)}), de az MRR a bekezdésszintű darabolásnál a legjobb ({f2(RP['S4_hibrid+rerank']['MRR@10']['mean'])} szemben az A {f2(RP['A_hibrid+rerank']['MRR@10']['mean'])}-os és a C {f2(RP['C_hibrid+rerank']['MRR@10']['mean'])}-es értékével). A fix ablak találati pontossága csaknem ugyanolyan jó, de a hivatkozás-pontossága a leggyengébb ({pc(GN['C_hibrid+rerank']['cit_precision'], 0)} szemben a B {pc(GN['S4_hibrid+rerank']['cit_precision'], 0)}-ával; helyes hivatkozás a válaszok {pc(GN['C_hibrid+rerank']['cit_any'], 0)}-ában, a B-nél {pc(GN['S4_hibrid+rerank']['cit_any'], 0)}-ában), mert az ablak több bekezdést kever, és a modell nem tudja, melyikből vette az állítást. A cikkszintű darabokat a modell hosszabb, zajosabb szöveggel kapja ({st['A']['atlag']:.0f} token átlagosan), ami az első találatok pontosságát rontja (R@1 {pc(RP['A_hibrid+rerank']['R@1']['mean'], 0)} szemben {pc(RP['S4_hibrid+rerank']['R@1']['mean'], 0)}).",
    f"<b>Reranker:</b> mindhárom darabolásnál javítja az MRR-t ({RP['A_hibrid']['MRR@10']['mean']:.2f} → {RP['A_hibrid+rerank']['MRR@10']['mean']:.2f}; {RP['S3_hibrid']['MRR@10']['mean']:.2f} → {RP['S4_hibrid+rerank']['MRR@10']['mean']:.2f}; {RP['C_hibrid']['MRR@10']['mean']:.2f} → {RP['C_hibrid+rerank']['MRR@10']['mean']:.2f}), a Recall@10 viszont nem következetesen nő (A: {pc(RP['A_hibrid']['R@10']['mean'],0)} → {pc(RP['A_hibrid+rerank']['R@10']['mean'],0)}; B: {pc(RP['S3_hibrid']['R@10']['mean'],0)} → {pc(RP['S4_hibrid+rerank']['R@10']['mean'],0)}; C: {pc(RP['C_hibrid']['R@10']['mean'],0)} → {pc(RP['C_hibrid+rerank']['R@10']['mean'],0)}), vagyis a reranker főleg a rangsor elejét rendezi át, a jelölthalmazt nem bővíti."])))
add(p("6.5 Kategóriák és hibaelemzés", H2))
cat = {"teny": "Ténykérdés", "hatály": "Hatály", "tobb": "Több helyet érintő", "kozott": "Jogszabályok közötti"}
rows = [["Kategória", "R@5 (S4)", "Helyes válasz: B0", "B1 (BM25)", "Hibrid", "S4"]]
for k, n in cat.items():
    rows.append([n, pc(o["retrieval_S4_kategoria_R5"][k], 0)] + [pc(GN[s]["kategoria_helyes"][k], 0) for s in ("B0_keresés_nélkül", "S1_bm25", "S3_hibrid", "S4_hibrid+rerank")])
add(tbl(rows, [4.6, 2.2, 3, 2.4, 2.2, 2.1]))
add(p("10. táblázat. Helyes válaszok kategóriánként", CAP))
add(p("A hibákat három csoportba sorolva: (1) a keresés nem hozta elő a helyes részt (az S4-nél a megválaszolható kérdések kb. 4–5%-a); (2) a helyes rész ott volt, de a modell rosszul értelmezte vagy elutasította a választ; (3) a tartalom jó, de a hivatkozás nem a várt helyre mutat. A legnehezebbek a jogszabályok közötti kérdések: ezeknél a modellnek két külön jogszabály szövegét kell összevetnie, és a rendszer gyakran csak az egyik oldalról hoz részt (hibrid: 73%, S4: 40% helyes). Gyakori hiba, hogy a NIS2 és a Kibertv. hasonló szabályai felcserélődnek. Kisebb hibatípus, hogy a modell kimondja az elutasítást, miközben a helyes tartalmat is leírja (ez a kézi ellenőrzés eltéréseinek fő forrása)."))
# ---------------- 7 ----------------
add(p("7. Korlátok és továbbfejlesztés", H1))
add(KeepTogether(bl([
    "<b>Kis tesztkészlet.</b> 90 megválaszolható kérdésen a néhány százalékpontos különbségek véletlenek is lehetnek; ezért adtam meg bootstrap konfidenciaintervallumot, és az S3–S4 keresési különbséget nem tekintem bizonyítottnak.",
    "<b>Félautomata teszthalmaz.</b> A kérdéseket nagy modell generálta, a válogatást és az ellenőrzést AI-asszisztens végezte, emberi (manuális) ellenőrzés nem történt. A kérdések megfogalmazása így hasonlít a modellek nyelvezetéhez, és egyes referenciaválaszok az eredeti szöveg pontatlan összefoglalásai lehetnek. A teljes lista a repóban van (eval/testset.jsonl).",
    "<b>LLM-bíráló.</b> A pontozást nagy modell végezte; a 30 elemű kézi ellenőrzés jó egyezést mutat (κ = 0,77), de kis minta, és az ellenőrzést is AI-asszisztens végezte, nem független emberi értékelő.",
    "<b>Egy futás.</b> A modell 0 hőmérsékleten fut, így egy futás reprodukálható, de a prompt kis módosítása is mozgathat néhány százalékpontot.",
    "<b>Időállapot és szűk forráskör.</b> A jogszabályok változnak; a végrehajtási rendeletek és a DORA technikai standardjai hiányoznak.",
    "<b>Belső hivatkozások.</b> A jogszabályok gyakran hivatkoznak egymásra („a 21. cikkben említett…”); a rendszer ezeket nem követi.",
    "<b>Továbbfejlesztés:</b> a reranker utáni összevonás (egy bekezdés darabjai egy egységként), jogszabályra szűrés a kérdésben megnevezett jogszabályra, jogszabályok közötti kérdésekre külön keresés jogszabályonként."])))
# ---------------- 8 ----------------
add(p("8. A megvalósítás elérhetősége és futtatása", H1))
add(p("A teljes kód, a tesztkérdések és a mérési eredmények a <font name='M'>kiberjog-rag/</font> mappában találhatók. <b>Repólink: https://github.com/SolteszDorka/MIprojekt.</b> A repó tartalma:"))
add(tbl([["Fájl", "Tartalom"],
         ["src/ingest.py", "HTML → darabok (A, B, C változat), metaadatok"],
         ["src/rag.py", "Qdrant-indexelés, BM25/dense/hibrid keresés, reranker, generálás, hivatkozás-ellenőrzés"],
         ["src/gen_testset.py", "tesztkérdések félautomata generálása"],
         ["src/run_eval.py", "keresés, válaszgenerálás, bírálat (újraindítható)"],
         ["src/report.py, src/build_doc.py", "metrikák, bootstrap, ábrák; ez a dokumentum"],
         ["eval/testset.jsonl", "a 100 tesztkérdés, arany jogszabályhelyekkel"],
         ["results/", "keresési találatok, válaszok, bírálatok, summary.json"],
         ["README.md", "telepítés és futtatás lépései"]], [5, 11.5]))
add(p("11. táblázat. A repó tartalma", CAP))
add(p("Futtatás röviden: <font name='M'>python src/ingest.py</font> → Qdrant indítása (Docker) és a <font name='M'>build_index</font> hívás a README szerint → <font name='M'>python src/run_eval.py</font> → <font name='M'>python src/report.py</font>. A nagy modellek (Qwen3.5-9B GGUF, BGE-M3, reranker) nincsenek a repóban; a README megadja a letöltésük módját."))
add(p("9. A leadási követelmények teljesülése", H1))
add(tbl([["Követelmény", "Hol található"], ["Mi a feladat?", "1. fejezet"], ["Milyen adaton?", "2. fejezet (NIS2, DORA, 2024. évi LXIX. törvény; 100 tesztkérdés)"],
         ["Milyen modellel vagy megközelítéssel?", "3. fejezet (RAG; BGE-M3, BM25, bge-reranker-v2-m3, Qwen3.5-9B)"], ["Milyen eredménnyel?", "6. fejezet"],
         ["Mihez mérve (alapvonal)?", "4. fejezet (Qwen keresés nélkül; BM25-RAG)"], ["Hol érhető el a kód?", "8. fejezet"],
         ["Prezentáció (legfeljebb 5 perc)", "külön fájl (PDF)"]], [6, 10.5]))
add(p("12. táblázat. A dokumentáció és a leadási követelmények", CAP))
add(p("Hivatkozások", H2))
refs = ["Az Európai Parlament és a Tanács (EU) 2022/2555 irányelve (NIS2). EUR-Lex: https://eur-lex.europa.eu/legal-content/HU/TXT/?uri=CELEX:32022L2555",
        "Az Európai Parlament és a Tanács (EU) 2022/2554 rendelete (DORA). EUR-Lex: https://eur-lex.europa.eu/legal-content/HU/TXT/?uri=CELEX:32022R2554",
        "2024. évi LXIX. törvény Magyarország kiberbiztonságáról. Nemzeti Jogszabálytár: https://njt.hu/jogszabaly/2024-69-00-00",
        "Lewis, P. et al. (2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.",
        "Chen, J. et al. (2024): BGE M3-Embedding. arXiv:2402.03216.",
        "Robertson, S., Zaragoza, H. (2009): The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR 3(4).",
        "Cormack, G. V. et al. (2009): Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR 2009.",
        "Qdrant dokumentáció – Hybrid Queries: https://qdrant.tech/documentation/concepts/hybrid-queries/",
        "BAAI/bge-reranker-v2-m3 modellkártya: https://huggingface.co/BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen3.5-9B modellkártya: https://huggingface.co/Qwen/Qwen3.5-9B"]
for i, r in enumerate(refs, 1): add(Paragraph(r, ParagraphStyle("r", parent=SM, leftIndent=14, bulletIndent=0, spaceAfter=2), bulletText=f"{i}."))


def foot(c, d):
    c.saveState(); c.setFont("D", 7.5); c.setFillColor(colors.HexColor("#666666"))
    if d.page > 1: c.drawString(2 * cm, 1.2 * cm, "Kérdés-válasz rendszer kiberbiztonsági jogszabályokhoz – dokumentáció")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(d.page)); c.restoreState()


out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "RAG_kiberjog_dokumentacio.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                  title="Kérdés-válasz rendszer kiberbiztonsági jogszabályokhoz", author="Soltész Dorka").build(S, onFirstPage=foot, onLaterPages=foot)
print(out)
