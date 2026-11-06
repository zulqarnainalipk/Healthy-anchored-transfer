"""Builds pipeline.html (journal-style pipeline figure) and renders fig4_pipeline.png / .pdf with headless Chrome.

Style follows common journal figures: flat pale boxes with thin outlines, serif labels, thin orthogonal arrows with
small sharp heads, real photographs and real data panels. Icons are Lucide (ISC licence), inlined as symbols.
Panels (images) come from analysis/48_pipeline_panels.py. Usage: python figures/fig4_pipeline/build_pipeline.py
"""

import json
import os
import re
import subprocess
import time

from PIL import Image
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Path to a Chrome or Chromium executable; set the environment variable CHROME_PATH to override the default.
CHROME = os.environ.get("CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
INK, TEAL, AMBER = "#0072B2", "#009E73", "#E69F00"   # Okabe-Ito (Wong 2011): blue, bluish green, orange
TEXT, MUTED, DARKAMBER, RED = "#1A1A1A", "#555555", "#B87A00", "#D55E00"


def symbol(name):
    txt = (HERE / "assets" / "icons" / f"{name}.svg").read_text(encoding="utf-8")
    inner = re.search(r"<svg[^>]*>(.*)</svg>", txt, re.S).group(1)
    return f'<symbol id="i-{name}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{inner}</symbol>'


def icon(name, x, y, s, color):
    return f'<use href="#i-{name}" x="{x}" y="{y}" width="{s}" height="{s}" style="color:{color}"/>'


def img(x, y, w, f):
    im = Image.open(HERE / 'assets' / f)
    h = w * im.height / im.width
    return f'<image x="{x}" y="{y}" width="{w}" height="{h:.1f}" href="assets/{f}"/>', h


def photo(x, y, s, f, color):
    return (f'<image x="{x}" y="{y}" width="{s}" height="{s}" href="assets/{f}" preserveAspectRatio="xMidYMid slice"/>'
            f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="none" stroke="{color}" stroke-width="1.5"/>')


def arrow(d, color=INK, dash=False, mk="a"):
    da = ' stroke-dasharray="5 3.5"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{da} marker-end="url(#{mk})"/>'


def box(x, y, w, h, fill, stroke, sw=1.2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def txt(x, y, s, size=15, anchor="start", weight="400", fill=TEXT, italic=False):
    size = round(size * 0.92, 1)
    st = ' font-style="italic"' if ("μ" in s or "TV" in s) else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{st}>{s}</text>'


def slabs(x, y):
    out = []
    for i in range(3):
        d = i * 30
        out.append(f'<polygon points="{x},{y+d+10} {x+40},{y+d+22} {x+40},{y+d+32} {x},{y+d+20}" fill="#8FBBD9" stroke="{INK}" stroke-width="1"/>')
        out.append(f'<polygon points="{x+40},{y+d+22} {x+120},{y+d+10} {x+120},{y+d+20} {x+40},{y+d+32}" fill="#B8D6EA" stroke="{INK}" stroke-width="1"/>')
        out.append(f'<polygon points="{x},{y+d+10} {x+80},{y+d} {x+120},{y+d+10} {x+40},{y+d+22}" fill="{"#F9DFA6" if i == 0 else "#E3F0F8"}" stroke="{INK}" stroke-width="1"/>')
    return "".join(out)


def build():
    s = []
    s.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="980" viewBox="0 0 1800 980" font-family="Arial,Helvetica,sans-serif">
<defs>
<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M0,1 L10,5 L0,9 L2.6,5 Z" fill="{INK}"/></marker>
<marker id="at" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M0,1 L10,5 L0,9 L2.6,5 Z" fill="{TEAL}"/></marker>
<marker id="aa" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M0,1 L10,5 L0,9 L2.6,5 Z" fill="#B87A00"/></marker>
{"".join(symbol(n) for n in ("brain-circuit", "network", "filter", "circle-check", "octagon-x", "snowflake", "tag", "sigma"))}
</defs>
<rect width="1800" height="980" fill="#FFFFFF"/>''')

    # ---- panel letters and titles
    for x, letter, title in ((30, "a", "Inputs"), (560, "b", "Shared frozen encoder"), (850, "c", "Feature spaces"), (1215, "d", "Healthy-anchored map"), (1500, "e", "Decision")):
        s.append(txt(x, 38, letter, 24, weight="700") + txt(x + 22, 38, title, 20))

    # ---- a. inputs
    s.append(box(30, 56, 470, 400, "#E8F6F1", TEAL))
    s.append(txt(46, 84, "Source crop: tomato (labelled)", 17, weight="700"))
    s.append(txt(46, 116, "Healthy leaves", 15, italic=True))
    for i, f in enumerate(("src_h1", "src_h2", "src_h3")):
        s.append(photo(46 + i * 112, 126, 100, f + ".jpg", TEAL))
    s.append(txt(390, 182, "n ≥ 50", 15, italic=True))
    s.append(txt(46, 268, "Diseased leaves (one model per class)", 15, italic=True))
    for i, (f, l) in enumerate((("src_d1", "early blight"), ("src_d2", "septoria"), ("src_d3", "leaf mold"))):
        s.append(photo(46 + i * 112, 278, 100, f + ".jpg", TEAL))
        s.append(txt(96 + i * 112, 398, l, 14, "middle"))
    s.append(txt(46, 432, "Gaussian mixture per disease class, shrunk covariance", 13.5, italic=True, fill="#555555"))

    s.append(box(30, 490, 470, 430, "#FDF3DF", AMBER))
    s.append(txt(46, 518, "Target crop: fig (healthy photographs only)", 17, weight="700"))
    s.append(txt(46, 550, "Healthy leaves, no disease labels", 15, italic=True))
    for i, f in enumerate(("tgt_h1", "tgt_h2", "tgt_h3")):
        s.append(photo(46 + i * 112, 560, 100, f + ".jpg", AMBER))
    s.append(txt(390, 616, "n ≥ 50", 15, italic=True))
    s.append(icon("tag", 392, 628, 28, "#8A97A8") + '<line x1="388" y1="660" x2="424" y2="626" stroke="#D55E00" stroke-width="2.5"/>')
    s.append(txt(46, 702, "New field leaf to classify (unlabelled)", 15, italic=True))
    s.append(photo(46, 712, 150, "tgt_new.jpg", AMBER))
    s.append(txt(216, 790, "disease of this leaf", 14, fill="#555555", italic=True))
    s.append(txt(216, 808, "is not given at any step", 14, fill="#555555", italic=True))

    # ---- b. encoder (centre)
    s.append(box(560, 250, 250, 480, "#F5F7FA", INK))
    s.append(txt(685, 282, "DINOv2 ViT-B/14", 16, "middle", weight="700"))
    s.append('<image x="580" y="312" width="210" height="33" href="assets/tokens.png"/>')
    s.append(txt(685, 364, "image cut into patch tokens", 13.5, "middle", italic=True))
    s.append(arrow("M685,372 V396"))
    s.append(slabs(625, 408))
    s.append(txt(756, 484, "× 12", 15, italic=True))
    s.append(txt(685, 516, "transformer blocks, weights frozen", 13.5, "middle", italic=True))
    s.append(arrow("M685,524 V548"))
    s.append('<image x="600" y="552" width="170" height="64" href="assets/feature_strip.png"/>' + f'<rect x="600" y="552" width="170" height="64" fill="none" stroke="{INK}" stroke-width="1"/>')
    s.append(txt(685, 636, "1,536 feature values per image", 13.5, "middle", italic=True))
    s.append(arrow("M685,644 V664"))
    s.append(box(612, 668, 146, 44, "#D5EEE6", TEAL))
    s.append(icon("filter", 620, 678, 22, INK))
    s.append(txt(648, 688, "LDA projection", 14, weight="700") + txt(648, 704, "16-d, source only", 12.5, italic=True))

    # arrows a -> b (orthogonal)
    s.append(arrow("M500,256 H530 V380 H558", TEAL, mk="at"))
    s.append(arrow("M500,700 H530 V600 H558", "#B87A00", mk="aa"))

    # ---- c. feature spaces (framed real plots)
    s.append(box(850, 56, 330, 420, "#E8F6F1", TEAL))
    s.append(txt(866, 82, "Source: tomato", 16, weight="700"))
    im, h = img(864, 94, 302, "space_source.png")
    s.append(im)
    y0 = 94 + h + 22
    leg = [("#00634A", "healthy", 866), ("#009E73", "early blight", 938), ("#4DC4A1", "septoria", 1034), ("#99DCC6", "leaf mold", 1100)]
    for c, l, x in leg:
        s.append(f'<circle cx="{x+5}" cy="{y0-5}" r="5" fill="{c}"/>' + txt(x + 15, y0, l, 13.5))
    s.append(txt(1015, y0 + 26, "real DINOv2 features, 16-d projection, 2-D view", 13.5, "middle", italic=True, fill="#555555"))
    s.append(txt(1015, y0 + 44, "ellipses: 2σ contours of the class Gaussians", 13.5, "middle", italic=True, fill="#555555"))

    s.append(box(850, 500, 330, 420, "#FDF3DF", AMBER))
    s.append(txt(866, 526, "Target: fig", 16, weight="700"))
    im, h = img(864, 538, 302, "space_target.png")
    s.append(im)
    y1 = 538 + h + 22
    s.append(f'<circle cx="871" cy="{y1-5}" r="5" fill="{AMBER}"/>' + txt(882, y1, "healthy (blue: Gaussian)", 13.5))
    s.append(f'<circle cx="1100" cy="{y1-5}" r="5" fill="none" stroke="#D55E00" stroke-width="1.8"/>' + txt(1111, y1, "new leaves", 13.5))
    s.append(f'<path d="M866,{y1+20} H896" stroke="{TEAL}" stroke-width="2" stroke-dasharray="6 3.5"/>' + txt(904, y1 + 25, "tomato models pushed through T", 13.5))
    s.append(txt(1015, y1 + 50, "new leaves: fig anthracnose, unseen when fitting", 13.5, "middle", italic=True, fill="#555555"))

    s.append(arrow("M810,380 H830 V256 H848", TEAL, mk="at"))
    s.append(arrow("M810,600 H830 V680 H848", "#B87A00", mk="aa"))

    # ---- d. map (real heat map of T)
    s.append(box(1215, 290, 250, 400, "#F5F7FA", INK))
    im, h = img(1235, 304, 210, "map_mini.png")
    s.append(im)
    y2 = 304 + h + 18
    s.append(txt(1340, y2, "Gaussian of source healthy → target healthy", 13, "middle", italic=True, fill="#555555"))
    im, h2 = img(1236, y2 + 14, 76, "map_T.png")
    s.append(im)
    s.append(txt(1274, y2 + 14 + h2 + 16, "T (16 × 16)", 13, "middle", italic=True))
    s.append(box(1322, y2 + 20, 130, 58, "#FFFFFF", "#C9D3E0", 1))
    s.append(txt(1387, y2 + 44, "x ↦ μ<tspan font-size=\"10\" dy=\"4\">B</tspan><tspan dy=\"-4\"> + T(x − μ</tspan><tspan font-size=\"10\" dy=\"4\">A</tspan><tspan dy=\"-4\">)</tspan>", 15.5, "middle", italic=True))
    s.append(txt(1387, y2 + 66, "push-through", 12.5, "middle", italic=True, fill="#555555"))
    yb = y2 + 14 + h2 + 46
    s.append(txt(1340, yb, "hypothesis N: disease effects commute", 13, "middle", italic=True, fill="#555555"))
    s.append(txt(1340, yb + 17, "with the host map (validated, not assumed)", 13, "middle", italic=True, fill="#555555"))
    s.append(txt(1340, yb + 40, "needs ≥ 50 healthy images", 13.5, "middle", weight="700"))
    s.append(arrow("M1180,266 H1198 V380 H1213", TEAL, mk="at"))
    s.append(arrow("M1180,710 H1198 V560 H1213", "#B87A00", mk="aa"))
    s.append(arrow("M1340,692 V770 H1182", dash=True))

    # ---- e. decision (real posterior and real discriminator)
    post = json.loads((HERE / "assets" / "posterior.json").read_text())["probabilities"]
    agr = json.loads((HERE / "assets" / "agreement.json").read_text())
    top = max(range(3), key=lambda i: post[i])
    s.append(box(1500, 56, 270, 420, "#FFFFFF", INK))
    s.append(txt(1635, 86, "Classify by likelihood", 16, "middle", weight="700"))
    s.append(txt(1635, 106, "under the pushed disease models", 13.5, "middle", italic=True, fill="#555555"))
    for i, v in enumerate(post):
        y = 142 + i * 52
        s.append(txt(1522, y, f"Class {i+1}", 14) + txt(1748, y, f"{v:.2f}", 14, "end", italic=True)
                 + f'<rect x="1522" y="{y+8}" width="226" height="12" fill="#ECEFF4"/><rect x="1522" y="{y+8}" width="{226*v:.1f}" height="12" fill="{AMBER if i == top else "#9FB3CC"}"/>')
    s.append(box(1522, 308, 226, 40, "#F9E3B8", AMBER))
    s.append(txt(1635, 334, f"predicted: class {top+1}", 16, "middle", weight="700"))
    s.append(txt(1635, 374, "probabilities of one real fig leaf under", 13.5, "middle", italic=True, fill="#555555"))
    s.append(txt(1635, 392, "the three pushed tomato models", 13.5, "middle", italic=True, fill="#555555"))
    s.append(txt(1635, 416, "no label of this leaf is used", 13.5, "middle", italic=True, fill="#555555"))

    s.append(box(1500, 500, 270, 420, "#FFFFFF", INK))
    s.append(txt(1635, 530, "Agreement check", 16, "middle", weight="700"))
    s.append(txt(1635, 548, "same disease from two source crops,", 13.5, "middle", fill="#555555") + txt(1635, 565, "both pushed to the target", 13.5, "middle", fill="#555555"))
    im, h = img(1550, 566, 170, "disc_hist.png")
    s.append(im)
    yy = 566 + h + 20
    s.append(txt(1635, yy, f"a = {agr['accuracy']:.2f}  →  TV ≥ {agr['tv_bound']:.2f}  (Wilson bound)", 15, "middle", italic=True))
    s.append(txt(1635, yy + 20, f"tomato vs potato early blight, {agr['test_images']} held-out images", 12.5, "middle", italic=True, fill="#555555"))
    refuse = agr["tv_bound"] > 0.5
    s.append(box(1522, 782, 108, 86, "#D5EEE6", TEAL, 1.2 if refuse else 2.2))
    s.append(icon("circle-check", 1557, 790, 34, TEAL) + txt(1576, 846, "answer", 14, "middle", weight="700") + txt(1576, 861, "bound small", 12, "middle", italic=True))
    s.append(box(1640, 782, 108, 86, "#F9E3B8", AMBER, 2.2 if refuse else 1.2))
    s.append(icon("octagon-x", 1675, 790, 34, "#B87A00") + txt(1694, 846, "refuse", 14, "middle", weight="700") + txt(1694, 861, "bound large", 12, "middle", italic=True))
    s.append(txt(1635, 888, f"this pair: {'refuse' if refuse else 'answer'} (bound {agr['tv_bound']:.2f})", 13.5, "middle", italic=True, fill="#555555"))
    s.append(txt(1635, 906, "large bound: the two sources disagree", 13, "middle", italic=True, fill="#555555"))

    s.append(arrow("M1467,400 H1484 V270 H1498"))
    s.append(arrow("M1467,560 H1484 V700 H1498", dash=True))

    # ---- legend
    s.append(f'<rect x="30" y="948" width="16" height="16" fill="{TEAL}"/>' + txt(54, 961, "source crop (labelled)", 14.5))
    s.append(f'<rect x="250" y="948" width="16" height="16" fill="{AMBER}"/>' + txt(274, 961, "target crop (healthy images only)", 14.5))
    s.append(f'<rect x="540" y="948" width="16" height="16" fill="{INK}"/>' + txt(564, 961, "transport and decision", 14.5))
    s.append(f'<path d="M760,956 H792" stroke="{TEAL}" stroke-width="2" stroke-dasharray="6 3.5"/>' + txt(800, 961, "pushed source disease model", 14.5))
    s.append(txt(1770, 961, "Photographs: PlantVillage, fig dataset. Real features; crop pair illustrative. Icons: Lucide (ISC). Palette: Okabe-Ito.", 12.5, "end", italic=True, fill="#555555"))
    s.append("</svg>")
    return "\n".join(s)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("step 1/4: building the SVG/HTML page from the panels in assets/")
    html = ('<!doctype html><html><head><meta charset="utf-8"><title>HATC pipeline</title>'
            '<style>html,body{margin:0;background:#fff}svg{display:block}</style></head><body>' + build() + "</body></html>")
    (HERE / "pipeline.html").write_text(html, encoding="utf-8")
    url = (HERE / "pipeline.html").as_uri()
    log(f"step 2/4: rendering the PNG with headless Chrome ({CHROME})")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1800,980", "--virtual-time-budget=4000", f"--screenshot={HERE / 'fig4_pipeline.png'}", url],
                   capture_output=True)
    log("step 3/4: rendering the PDF with headless Chrome")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=4000",
                    f"--print-to-pdf={HERE / 'fig4_pipeline.pdf'}", url], capture_output=True)
    log(f"step 4/4: done, built {HERE / 'fig4_pipeline.png'}")


if __name__ == "__main__":
    main()
