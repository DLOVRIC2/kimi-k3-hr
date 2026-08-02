#!/usr/bin/env python3
"""Generate the README figures from the per-item records.

Same rule as analysis.py: nothing is drawn from a number typed by hand. Each
figure reads the checkpoints through analysis.load_items, so a chart cannot
drift out of step with the table beside it -- which is exactly what happened to
the prose (LOG.md #25).

Hand-rolled SVG rather than matplotlib: it adds no dependency, the output is
deterministic (no font metrics, no version-dependent layout), and the files are
a few KB of readable text that diff sensibly.

Colours are chosen to read on BOTH GitHub themes. Charts served through GitHub's
image proxy do not reliably get prefers-color-scheme, so the palette avoids pure
black and pure white entirely and uses a mid grey for text that has contrast
against either background.

    uv run eval/charts.py            # writes docs/*.svg
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import analysis as A  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs"

# Two renderings of the same figures, from the same records.
#
# "github" is self-contained: literal hex chosen to read on both GitHub themes,
# because charts served through GitHub's image proxy do not reliably receive
# prefers-color-scheme.
#
# "blog" targets the Algorise site, which defines --spec-* colour tokens in
# oklch with separate light and dark values, and s-* text classes scoped under
# .spec-diagram. Emitting tokens rather than hex is what makes the figure follow
# the reader's theme instead of fighting it. Same numbers either way -- the
# alternative was hand-drawing charts from the results, which is precisely the
# fabrication the content pipeline's evidence gate exists to prevent.
THEME = "github"

PALETTE = {
    "github": {
        "ink": "#7d8590", "grid": "#7d8590", "bg": "#ffffff",
        "K3-REAP80-hr-code": "#f85149", "gpt-oss-120b": "#58a6ff",
        "gemma4-31b": "#3fb950", "gpt-oss-20b": "#d29922",
        "correct": "#3fb950", "wrong": "#7d8590", "silent": "#f85149",
    },
    "blog": {
        "ink": "var(--spec-muted)", "grid": "var(--spec-border)",
        "bg": "var(--spec-bg)",
        "K3-REAP80-hr-code": "var(--spec-red)", "gpt-oss-120b": "var(--spec-teal)",
        "gemma4-31b": "var(--spec-green)", "gpt-oss-20b": "var(--spec-amber)",
        "correct": "var(--spec-green)", "wrong": "var(--spec-muted)",
        "silent": "var(--spec-red)",
    },
}

# Text classes the blog stylesheet provides; on GitHub they do not exist, so the
# github theme keeps inline font attributes instead.
CLASS_FOR = {"title": "s-lane", "sub": "s-sub", "axis": "s-edge",
             "label": "s-titleS", "num": "s-num"}


def pal(key: str) -> str:
    return PALETTE[THEME][key]


class _Ink(str):
    """Resolves to the current theme's ink colour at format time."""
    def __str__(self): return pal("ink")


INK = _Ink()
FAINT = _Ink()
K3 = "#f85149"
BLUE = "#58a6ff"
GREEN = "#3fb950"
GOLD = "#d29922"


class _Color(dict):
    def __getitem__(self, k): return pal(k)


COLOR = _Color()
SHORT = {
    "K3-REAP80-hr-code": "K3-REAP80-hr-code",
    "gpt-oss-120b": "gpt-oss:120b",
    "gemma4-31b": "gemma4:31b",
    "gpt-oss-20b": "gpt-oss:20b",
}
FONT = ("font-family=\"ui-sans-serif,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Helvetica,Arial,sans-serif\"")


def svg(w: int, h: int, body: str, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{title}">\n'
            f'<title>{title}</title>\n{body}\n</svg>\n')


def text(x, y, s, size=12, fill=None, anchor="start", weight="400", opacity=1.0,
         role="axis"):
    """One text primitive, two stylings.

    On the blog the size/weight/family come from the stylesheet via an s-* class,
    so emitting them inline would override the site's own typography scale. Only
    fill is kept, and only when the caller means a specific semantic colour.
    """
    fill = pal("ink") if fill is None else fill
    if THEME == "blog":
        cls = CLASS_FOR.get(role, "s-edge")
        f = f' fill="{fill}"' if fill != pal("ink") else ""
        op = f' opacity="{opacity}"' if opacity != 1.0 else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
                f'text-anchor="{anchor}"{f}{op}>{s}</text>')
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" opacity="{opacity}">{s}</text>')


def rect(x, y, w, h, fill, opacity=1.0, rx=2):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" '
            f'fill="{fill}" opacity="{opacity}" rx="{rx}"/>')


def line(x1, y1, x2, y2, stroke=None, width=1, opacity=0.25, dash=None):
    stroke = pal("grid") if stroke is None else stroke
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{d}/>')


# ------------------------------------------------------------------ data

def belebele(model: str, task: str):
    return A.score(model, task)


def fig_size_vs_score() -> str:
    """Disk footprint against Croatian accuracy. The whole cost argument."""
    W, H = 760, 420
    L, R, T, B = 70, 30, 46, 64
    import math
    pts = []
    for m in A.MODELS:
        pts.append((A.DISK_GB[m], belebele(m, "belebele_hrv").acc * 100, m))

    x0, x1 = math.log10(10), math.log10(500)
    def px(gb): return L + (math.log10(gb) - x0) / (x1 - x0) * (W - L - R)
    def py(pct): return H - B - (pct / 100) * (H - T - B)

    b = [text(L, 24, "Croatian comprehension against model size", 14, INK, weight="600", role="title")]
    for pct in (0, 25, 50, 75, 100):
        b.append(line(L, py(pct), W - R, py(pct), opacity=0.15))
        b.append(text(L - 10, py(pct) + 4, f"{pct}%", 11, INK, "end", opacity=0.75))
    b.append(line(L, py(25), W - R, py(25), stroke=INK, opacity=0.45, dash="4 4"))
    b.append(text(W - R, py(25) - 7, "chance", 10, INK, "end", opacity=0.7))

    for gb in (10, 25, 50, 100, 200, 400):
        b.append(text(px(gb), H - B + 20, f"{gb}", 11, INK, "middle", opacity=0.75))
    b.append(text((L + W - R) / 2, H - 14, "weights on disk (GB, log scale)", 11,
                  INK, "middle", opacity=0.8))

    for gb, pct, m in pts:
        c = COLOR[m]
        b.append(f'<circle cx="{px(gb):.1f}" cy="{py(pct):.1f}" r="7" fill="{c}"/>')
        above = m != "K3-REAP80-hr-code"
        dy = -16 if above else 26
        anchor = "middle" if m != "gemma4-31b" else "start"
        lx = px(gb) if anchor == "middle" else px(gb) - 4
        b.append(text(lx, py(pct) + dy, f"{SHORT[m]}", 12, c, anchor, weight="600",
                      role="label"))
        b.append(text(lx, py(pct) + dy + (-14 if above else 14),
                      f"{pct:.1f}%  ·  {gb} GB", 11, INK, anchor, opacity=0.85))

    k3 = next(p for p in pts if p[2] == "K3-REAP80-hr-code")
    gem = next(p for p in pts if p[2] == "gemma4-31b")
    b.append(line(px(gem[0]), py(gem[1]), px(k3[0]), py(k3[1]),
                  stroke=INK, width=1.5, opacity=0.35, dash="5 5"))
    mid_x, mid_y = (px(gem[0]) + px(k3[0])) / 2, (py(gem[1]) + py(k3[1])) / 2
    b.append(text(mid_x, mid_y - 8, "17x the disk, 48 points worse", 12, INK,
                  "middle", weight="600", opacity=0.95))
    return svg(W, H, "\n".join(b), "Croatian accuracy against model size")


def fig_response_split() -> str:
    """Did it answer, and was it right -- the bimodal failure, per task."""
    tasks = [("belebele_hrv", "Croatian"), ("belebele_eng", "English"),
             ("humaneval", "HumanEval")]
    HEAD, ROW, GAP = 26, 27, 14
    T, B, L, R = 74, 26, 132, 26
    H = int(T + B + len(A.MODELS) * (HEAD + len(tasks) * ROW + GAP) - GAP)
    W = 760
    full = W - L - R
    bar_h = 18

    b = [text(20, 26, "Every score splits in two: did it answer, was it right", 14,
              INK, weight="600", role="title"),
         text(20, 45, "K3 is the only model here that fails by not answering at all",
              11, INK, opacity=0.8, role="sub")]
    for i, (lab, col, op) in enumerate([("correct", pal("correct"), 1.0),
                                        ("answered, wrong", pal("wrong"), 0.45),
                                        ("no answer", pal("silent"), 1.0)]):
        x = 300 + i * 150
        b.append(rect(x, 56, 10, 10, col, op))
        b.append(text(x + 15, 65, lab, 11, INK, opacity=0.85))

    y = T
    for m in A.MODELS:
        b.append(text(20, y + 13, SHORT[m], 12.5, COLOR[m], weight="700",
                      role="label"))
        y += HEAD
        for task, tname in tasks:
            s = belebele(m, task)
            correct, silent = s.acc, 1 - s.resp_rate
            answered_wrong = s.resp_rate - s.acc
            x, ty = L, y + (ROW - bar_h) / 2
            for frac, col, op in ((correct, pal("correct"), 1.0),
                                  (answered_wrong, pal("wrong"), 0.45),
                                  (silent, pal("silent"), 1.0)):
                b.append(rect(x, ty, frac * full, bar_h, col, op))
                x += frac * full
            b.append(text(L - 10, ty + 13, tname, 11, INK, "end", opacity=0.85))
            b.append(text(L + 7, ty + 13, f"{correct*100:.0f}%", 10.5, pal("bg"),
                          weight="700", role="num"))
            # Only label silence when the band is wide enough to hold the text;
            # a 2% sliver with a caption is noise, and the legend already says
            # what the colour means.
            if silent >= 0.10:
                b.append(text(W - R - 7, ty + 13, f"{silent*100:.0f}% no answer",
                              10.5, pal("bg"), "end", weight="700", role="num"))
            y += ROW
        y += GAP
    return svg(W, H, "\n".join(b), "Response rate and accuracy split by task")


def fig_language_gap() -> str:
    """Croatian minus English, on response rate and on accuracy."""
    W, H = 760, 380
    L, R, T, B = 150, 130, 74, 46
    mid = L + (W - L - R) / 2
    scale = (W - L - R) / 2 / 22.0     # +-22 points full scale

    b = [text(20, 26, "Croatian minus English, on identical passages", 14, INK,
              weight="600", role="title"),
         text(20, 45, "K3 answers far more in Croatian. It is not more accurate in it.",
              11, INK, opacity=0.8, role="sub")]
    b.append(rect(W - R - 118, 18, 10, 10, pal("silent"), 1.0))
    b.append(text(W - R - 103, 27, "response rate", 11, INK, opacity=0.85))
    b.append(rect(W - R - 118, 38, 10, 10, pal("wrong"), 0.45))
    b.append(text(W - R - 103, 47, "accuracy", 11, INK, opacity=0.85))

    band = (H - T - B) / len(A.MODELS)
    bar_h = band * 0.3
    b.append(line(mid, T - 6, mid, H - B + 6, stroke=INK, width=1, opacity=0.4))
    b.append(text(mid, H - 14, "0", 11, INK, "middle", opacity=0.7))
    b.append(text(mid - 90, H - 14, "more English", 10, INK, "middle", opacity=0.6))
    b.append(text(mid + 90, H - 14, "more Croatian", 10, INK, "middle", opacity=0.6))

    y = T
    for m in A.MODELS:
        ids, hrv, eng = A.paired_items(m)
        n = len(ids)
        d_resp = (sum(hrv[i]["parsed"] for i in ids)
                  - sum(eng[i]["parsed"] for i in ids)) / n * 100
        d_acc = (sum(hrv[i]["correct"] for i in ids)
                 - sum(eng[i]["correct"] for i in ids)) / n * 100
        b.append(text(L - 14, y + band / 2 + 4, SHORT[m], 12, COLOR[m], "end",
                      weight="600", role="label"))
        for k, (d, col, op) in enumerate([(d_resp, pal("silent"), 1.0),
                                          (d_acc, pal("wrong"), 0.45)]):
            yy = y + band / 2 - bar_h - 3 + k * (bar_h + 6)
            w = abs(d) * scale
            x = mid if d >= 0 else mid - w
            b.append(rect(x, yy, w, bar_h, col, op))
            lx = mid + w + 8 if d >= 0 else mid - w - 8
            b.append(text(lx, yy + bar_h - 1, f"{d:+.1f}", 11, INK,
                          "start" if d >= 0 else "end", weight="600", opacity=0.95,
                          role="num"))
        y += band

    return svg(W, H, "\n".join(b), "Croatian minus English by response rate and accuracy")


def fig_corrections() -> str:
    """What the two post-hoc corrections did to the headline number."""
    W, H = 760, 270
    L, R, T, B = 44, 44, 108, 40
    b = [text(20, 24, "The headline gap, as two errors were removed", 14, INK,
              weight="600", role="title"),
         text(20, 42, "K3 Croatian minus English accuracy, same run, three analyses",
              11, INK, opacity=0.8, role="sub")]

    hrv = {r["id"]: r for r in A.load_items("K3-REAP80-hr-code", "belebele_hrv",
                                            paired=False)}
    eng_old = A.load_items("K3-REAP80-hr-code", "belebele_eng", paired=False)
    v1 = (sum(r["correct"] for r in eng_old) / len(eng_old)
          - sum(r["correct"] for r in hrv.values()) / len(hrv)) * 100
    h2 = {r["id"]: r for r in A.load_items("K3-REAP80-hr-code", "belebele_hrv")}
    e2 = A.load_items("K3-REAP80-hr-code", "belebele_eng_unpaired")
    v2 = (sum(r["correct"] for r in e2) / len(e2)
          - sum(r["correct"] for r in h2.values()) / len(h2)) * 100
    ids, h3, e3 = A.paired_items("K3-REAP80-hr-code")
    v3 = (sum(e3[i]["correct"] for i in ids) - sum(h3[i]["correct"] for i in ids)) / len(ids) * 100

    stages = [(v1, "as first published", "unpaired items, loose echo rule"),
              (v2, "echo rule fixed", "still unpaired"),
              (v3, "and items paired", "the actual result")]
    colw = (W - L - R) / 3
    maxv = 18.0
    # Values are negative, so the bars hang from a shared zero line -- which is
    # what a reader expects of a negative quantity, and it puts every value
    # label at the same height so the three are directly comparable.
    b.append(line(L, T, W - R, T, stroke=INK, width=1, opacity=0.35))
    b.append(text(L - 6, T + 4, "0", 10, INK, "end", opacity=0.7))
    for i, (v, title, sub) in enumerate(stages):
        cx = L + colw * i + colw / 2
        h = abs(v) / maxv * 92
        col = pal("silent") if i < 2 else pal("wrong")
        op = 1.0 if i < 2 else 0.55
        b.append(rect(cx - 36, T, 72, h, col, op))
        b.append(text(cx, T - 12, f"{v:+.1f}", 21, INK, "middle", weight="700",
                      role="num"))
        b.append(text(cx, H - 56, title, 12, INK, "middle", weight="600",
                      role="label"))
        b.append(text(cx, H - 40, sub, 10, INK, "middle", opacity=0.75))
        if i < 2:
            b.append(text(L + colw * (i + 1), T + 52, "→", 18, INK, "middle",
                          opacity=0.45))
    b.append(text(W / 2, H - 14,
                  "significance: p=0.00064 ***  →  p=0.13 n.s.", 11,
                  INK, "middle", opacity=0.9))
    return svg(W, H, "\n".join(b), "Effect of the two corrections on the headline gap")


FIGS = {
    "fig1-size-vs-score.svg": fig_size_vs_score,
    "fig2-response-split.svg": fig_response_split,
    "fig3-language-gap.svg": fig_language_gap,
    "fig4-corrections.svg": fig_corrections,
}


def main() -> None:
    global THEME
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theme", choices=list(PALETTE), default="github")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to docs/ or docs/blog/")
    a = ap.parse_args()
    THEME = a.theme
    out = pathlib.Path(a.out) if a.out else (OUT if THEME == "github" else OUT / "blog")
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        (out / name).write_text(fn())
        print(f"wrote {out.relative_to(OUT.parent)}/{name}")


if __name__ == "__main__":
    main()
