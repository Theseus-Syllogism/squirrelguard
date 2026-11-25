#!/usr/bin/env python3
"""Build the public documents: Markdown -> designed HTML -> PDF (US Letter).

    python3 docs/public/tools/build_docs.py            # FEATURE-GUIDE.md -> build/FEATURE-GUIDE.html + .pdf

What the build adds to the plain Markdown (the .md stays readable on its own):
  layout      a margin column for section numbers, icons and labels; text at a readable line length beside it;
              tables, figures and listings use the full width
  navigation  a contents page and chapter openers with page numbers, page numbers on cross-references, a running
              header with the chapter, PDF bookmarks; landscape diagrams get landscape pages
  marks       status chips with a symbol (not colour alone), Measured / Limits / Safeguards labels in the margin,
              notices with icons, licence chips, key caps, terminal listings, software logos and letter tiles

Needs: pip install markdown==3.7 beautifulsoup4 pypdf, and a Chromium/Chrome binary (set CHROME if it is not found).
Fonts: Inter and JetBrains Mono if installed (both SIL OFL), else the system sans-serif and monospace.
The PDF is printed twice: the first print tells which page every heading lands on, the second carries the numbers.
"""
import datetime
import glob
import html
import os
import pathlib
import re
import shutil
import subprocess
import sys

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
GUTTER = "1.42in"

CSS = r"""
@page { size: Letter; margin: 0.92in 0.7in 0.86in 0.7in;
  @top-left { content: "squirrel-guard  ·  Feature and Capability Guide"; font: 7.5pt "SG Inter", Inter, sans-serif; color: #5B6670;
    vertical-align: bottom; padding-bottom: 7pt; border-bottom: .6pt solid #C9D1D9; }
  @top-center { content: ""; border-bottom: .6pt solid #C9D1D9; }
  @top-right { font: 600 7.5pt "SG Inter", Inter, sans-serif; color: #1F3B57; vertical-align: bottom; padding-bottom: 7pt;
    border-bottom: .6pt solid #C9D1D9; }
  @bottom-left { content: "%(footer)s"; font: 7.5pt "SG Inter", Inter, sans-serif; color: #5B6670; vertical-align: top; padding-top: 8pt; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 600 7.5pt "SG Inter", Inter, sans-serif; color: #1F3B57;
    vertical-align: top; padding-top: 8pt; } }
@page cover { margin: 0; @top-left { content: none; border: 0 } @top-center { content: none; border: 0 }
  @top-right { content: none; border: 0 } @bottom-left { content: none } @bottom-right { content: none } }
%(pages)s
:root { --navy: #1F3B57; --ink: #1B1F24; --ink2: #3D4853; --grey: #5B6670; --line: #D5DBE1; --tint: #F3F6F9;
  --accent: #A8480F; --accent-tint: #FBEFE6; --blue: #2E6DA4; --green: #2E7D45; --amber: #9A5B00; --gutter: %(gutter)s; }
* { box-sizing: border-box; }
html { font-family: "SG Inter", Inter, "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; color: var(--ink); line-height: 1.5;
  font-feature-settings: "calt", "cv05", "cv08"; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; }
main { max-width: 7.1in; margin: 0 auto; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 7pt; }
a { color: var(--blue); text-decoration: none; }
strong, b { font-weight: 600; }
ul, ol { margin: 0 0 7pt; padding-left: 15pt; }
li { margin-bottom: 2.5pt; }
li::marker { color: var(--accent); font-weight: 600; }
hr { display: none; }
code { font: 8.2pt "SG Mono", "JetBrains Mono", "DejaVu Sans Mono", monospace; background: var(--tint); padding: .5pt 3pt; border-radius: 3px;
  border: .5pt solid #E3E8ED; overflow-wrap: anywhere; }
img.logo { height: 1.3em; width: auto; vertical-align: -.27em; margin-right: 2.5pt; }
img.ico { height: 1.25em; width: auto; vertical-align: -.25em; }

/* ---- the margin column: everything in a chapter starts after the gutter unless it is "full" ---- */
.sec > * { margin-left: var(--gutter); }
.sec > table, .sec > figure, .sec > .spec, .sec > .co, .sec > h3, .sec > h4.full, .sec > .ch-head, .sec > .full,
.sec > .toc, .sec > ol.flow, .sec > .term { margin-left: 0; }

/* ---- chapter opener ---- */
.sec { break-before: page; }
.ch-head { display: grid; grid-template-columns: var(--gutter) 1fr auto; align-items: end; padding: 5pt 0 9pt;
  border-bottom: 2.2pt solid var(--navy); margin-bottom: 12pt; }
.ch-num { font: 300 40pt/0.9 "SG Inter Display", "Inter Display", "SG Inter", Inter, sans-serif; color: var(--accent); letter-spacing: -.02em; }
.ch-head h2 { font: 700 21pt/1.12 "SG Inter Display", "Inter Display", "SG Inter", Inter, sans-serif; color: var(--navy); margin: 0; letter-spacing: -.012em; }
.ch-ico { width: 34pt; height: 34pt; border-radius: 50%%; background: var(--tint); display: grid; place-items: center; }
.ch-ico img { width: 19pt; height: 19pt; }
.front .ch-head { grid-template-columns: 1fr auto; }
p.lead { font-size: 11.2pt; line-height: 1.45; color: var(--ink2); margin-bottom: 10pt; }
.ch-toc { margin: 4pt 0 6pt var(--gutter); padding: 7pt 0 3pt; border-top: .6pt solid var(--line); border-bottom: .6pt solid var(--line);
  columns: 2; column-gap: 20pt; font-size: 8.4pt; }
.ch-toc a { display: flex; gap: 6pt; color: var(--ink2); padding: 1.2pt 0; break-inside: avoid; }
.ch-toc .n { color: var(--accent); font-weight: 600; min-width: 20pt; font-feature-settings: "tnum"; }
.ch-toc .t { flex: 1; }
.ch-toc .pg, .toc .pg { color: var(--grey); font-feature-settings: "tnum"; }

/* ---- section headings: number and icon in the margin ---- */
h3, h4 { color: var(--navy); line-height: 1.22; break-after: avoid; }
h3 { display: grid; grid-template-columns: var(--gutter) 1fr; align-items: center; font-size: 13pt; font-weight: 700;
  margin: 17pt 0 7pt; padding-top: 11pt; border-top: .6pt solid var(--line); letter-spacing: -.005em; }
.ch-head + h3, .lead + h3, .ch-toc + h3, .adm + h3 { border-top: 0; padding-top: 0; margin-top: 12pt; }
.h-side { display: flex; align-items: center; justify-content: space-between; padding-right: 11pt; }
.h-num { font: 600 11pt "SG Inter", Inter, sans-serif; color: var(--accent); font-feature-settings: "tnum"; }
.h-ico { width: 21pt; height: 21pt; border-radius: 5pt; background: var(--tint); display: grid; place-items: center; }
.h-ico img { width: 12.5pt; height: 12.5pt; }
h4 { font-size: 10.2pt; margin: 11pt 0 4pt; }
h3#marks-used-in-this-guide { break-before: page; border-top: 0; padding-top: 0; margin-top: 0; }

/* ---- feature summary strip under a section heading ---- */
.spec { display: flex; gap: 0; margin: 0 0 9pt; border-top: 1.4pt solid var(--navy); border-bottom: .6pt solid var(--line);
  background: var(--tint); break-inside: avoid; break-after: avoid; }
.spec > div { padding: 5.5pt 10pt 6pt; border-left: .6pt solid var(--line); flex: 1 1 0; min-width: 0; }
.spec > div:first-child { border-left: 0; flex: 0 0 var(--gutter); }
.spec > div.long { flex-grow: 2.2; }
.spec dt { font-size: 6.6pt; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: var(--grey);
  display: flex; align-items: center; gap: 3.5pt; margin-bottom: 2pt; }
.spec dt svg { width: 8.5pt; height: 8.5pt; }
.spec dd { margin: 0; font-size: 8.5pt; line-height: 1.35; }
.spec .st { margin-bottom: 1.5pt; }
dl.spec { padding: 0; }

/* ---- Measured / Limits / Safeguards: label in the margin, text beside a rule ---- */
.co { display: grid; grid-template-columns: var(--gutter) 1fr; margin: 7pt 0 8pt; }
.co-l { font-size: 7pt; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; display: flex; gap: 4.5pt;
  align-items: center; align-self: start; padding-top: 2.2pt; color: var(--c); }
.co-l svg { width: 10.5pt; height: 10.5pt; flex: none; }
.co-b { border-left: 1.8pt solid var(--c); padding: 1pt 0 1pt 9pt; }
.co-b > :last-child { margin-bottom: 0; }
.co-measured { --c: #2E6DA4; } .co-limits { --c: #9A5B00; } .co-safe { --c: #2E7D45; } .co-gate { --c: #44546A; }
.co-todo { --c: #6B4FBB; }

/* ---- notices ---- */
.adm { display: grid; grid-template-columns: 17pt 1fr; gap: 0 5pt; padding: 8pt 11pt 8pt 9pt; margin-top: 8pt; margin-bottom: 10pt;
  border-radius: 4pt; border: .6pt solid var(--b); background: var(--bg); break-inside: avoid; }
.adm > svg { width: 12.5pt; height: 12.5pt; margin-top: 1pt; }
.adm p { margin: 0 0 4pt; } .adm > div > :last-child { margin-bottom: 0; }
.adm-important { --b: #E3C48F; --bg: #FFF7E8; } .adm-note { --b: #B9DCC4; --bg: #EEF8F1; }
.adm-notice { --b: #E3C48F; --bg: #FFF7E8; } .adm-info { --b: #BFD4EA; --bg: #EEF4FB; }

/* ---- tables ---- */
table { border-collapse: collapse; width: 100%%; margin: 5pt 0 11pt; font-size: 8.5pt; line-height: 1.38; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { text-align: left; font-size: 6.8pt; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--navy);
  padding: 4pt 6pt 4pt; border-bottom: 1.4pt solid var(--navy); vertical-align: bottom; }
td { border-bottom: .6pt solid var(--line); padding: 4.2pt 6pt; vertical-align: top; }
tbody tr:nth-child(even) td { background: #F8FAFB; }
td:first-child, th:first-child { padding-left: 2pt; }
td img.ico { height: 13pt; }
td.c-ico { width: 20pt; padding-right: 0; }
td.num, th.num { text-align: right; font-feature-settings: "tnum"; }
table.facts { width: auto; min-width: 70%%; margin-left: var(--gutter); }
table.facts thead { display: none; }
table.facts td:first-child { width: 1.25in; color: var(--grey); white-space: nowrap; font-weight: 600; }
table.facts tbody tr:nth-child(even) td { background: none; }
.sec > table.narrow { margin-left: var(--gutter); width: calc(100%% - var(--gutter)); }
.xr { white-space: nowrap; color: var(--grey); font-size: .92em; font-feature-settings: "tnum"; }

/* ---- chips ---- */
.st, .lic, .idf, .tag { display: inline-flex; align-items: center; gap: 3pt; padding: .8pt 6pt .8pt 4.5pt; border-radius: 9pt;
  font-size: 7.4pt; font-weight: 600; white-space: nowrap; line-height: 1.45; vertical-align: 1pt; }
.st svg, .lic svg, .idf svg { width: 8.3pt; height: 8.3pt; flex: none; }
.st-prod { background: #E3F2E8; color: #1B5730; } .st-prov { background: #E1ECF8; color: #1C4873; }
.st-exp { background: #FFF1D9; color: #734600; } .st-off { background: #EBEEF1; color: #444E57; }
.st-new { background: #ECE8FA; color: #47348A; }
.lic { border-radius: 3pt; padding: .5pt 4.5pt; font-family: "SG Mono", "JetBrains Mono", monospace; font-size: 7pt; font-weight: 600; }
.lic-ok { background: #E3F2E8; color: #1B5730; } .lic-copyleft { background: #E1ECF8; color: #1C4873; }
.lic-nc { background: #FFE9D6; color: #7A3200; } .lic-unk { background: #EBEEF1; color: #444E57; }
.idf { padding-left: 0; background: none; font-size: 8.3pt; font-weight: 500; }
.idf-yes { color: #7A3200; font-weight: 600; } .idf-no { color: var(--grey); } .idf-ind { color: #1C4873; }
.tag { border-radius: 3pt; padding: .3pt 4pt; background: var(--navy); color: #fff; font-size: 6.6pt; letter-spacing: .07em;
  font-weight: 700; vertical-align: 1.2pt; }
.tag-file { background: #fff; color: var(--navy); border: .7pt solid var(--navy); font-family: "SG Mono", "JetBrains Mono", monospace; letter-spacing: 0; }
.tag-proto { background: #44546A; }
.mk { display: inline-flex; align-items: center; gap: 3.5pt; font-size: 6.8pt; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
  margin-right: 7pt; white-space: nowrap; }
.mk svg { width: 9.5pt; height: 9.5pt; }
.mk-measured { color: #2E6DA4; } .mk-limits { color: #9A5B00; } .mk-safe { color: #2E7D45; }
kbd { display: inline-block; font: 600 7.4pt "SG Mono", "JetBrains Mono", monospace; color: var(--ink); background: #fff; border: .7pt solid #AEB8C2;
  border-bottom-width: 1.8pt; border-radius: 3pt; padding: 0 3.5pt; min-width: 12pt; text-align: center; line-height: 1.5; vertical-align: .6pt; }

/* ---- figures ---- */
figure { margin: 9pt 0 12pt; break-inside: avoid; }
figure img { max-width: 100%%; border: .6pt solid var(--line); border-radius: 3pt; display: block; margin: 0 auto; }
figure img[src*="screens/"] { max-height: 3.72in; width: auto; box-shadow: 0 1pt 5pt #1F3B5718; }
figure.tall img[src*="screens/"] { max-height: 6.4in; }
figure.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; align-items: start; }
figure.pair figcaption { grid-column: 1 / -1; }
figcaption { display: grid; grid-template-columns: var(--gutter) 1fr; font-size: 8.1pt; line-height: 1.4; color: var(--ink2); margin-top: 6pt; }
.fig-n { font-size: 7pt; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--accent); padding-top: 1pt; }
figure.wide { margin: 0; break-inside: avoid; }
figure.wide img { width: 100%%; height: auto; max-height: 6.0in; object-fit: contain; border: 0; }
figure.wide figcaption { margin-top: 8pt; }

/* ---- listings ---- */
.term { margin: 7pt 0 10pt; border: .6pt solid #C3CCD5; border-radius: 4pt; overflow: hidden; break-inside: avoid; }
.term-h { display: flex; align-items: center; gap: 5pt; font-size: 6.8pt; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
  color: var(--navy); background: #E9EEF3; padding: 3pt 8pt; border-bottom: .6pt solid #C3CCD5; }
.term-h svg { width: 9pt; height: 9pt; }
.term pre { margin: 0; padding: 7pt 9pt; font: 7.5pt/1.5 "SG Mono", "JetBrains Mono", "DejaVu Sans Mono", monospace; white-space: pre-wrap;
  overflow-wrap: anywhere; background: #FAFBFC; color: var(--ink2); }
.term pre code { font: inherit; background: none; border: 0; padding: 0; }
.term .pr { color: var(--accent); font-weight: 700; user-select: none; }
.term .cmd { color: var(--ink); font-weight: 600; }
.term .cm { color: #6B7680; font-style: italic; }
.term.screen pre { background: #14202E; color: #D5DEE8; font-size: 7.1pt; line-height: 1.32; white-space: pre; }
.term.screen .term-h { background: #1F3B57; color: #fff; border-color: #1F3B57; }

/* ---- step flow ---- */
ol.flow { list-style: none; padding: 0; margin: 9pt 0 12pt; display: grid; grid-template-columns: repeat(%(flowcols)s, 1fr); gap: 7pt 12pt;
  counter-reset: flow; }
ol.flow li { counter-increment: flow; position: relative; margin: 0; padding: 7pt 8pt 7pt; border: .6pt solid var(--line);
  border-top: 1.6pt solid var(--navy); border-radius: 0 0 4pt 4pt; background: #fff; font-size: 8pt; line-height: 1.36; break-inside: avoid; }
ol.flow li::before { content: counter(flow); position: absolute; top: 5pt; right: 7pt; font: 600 8pt "SG Inter", Inter; color: var(--accent); }
ol.flow li:not(:nth-child(%(flowcols)sn))::after { content: "›"; position: absolute; right: -9.5pt; top: 36%%; font: 600 12pt "SG Inter", Inter; color: #8C97A2; }
ol.flow li b:first-of-type { display: block; color: var(--navy); font-size: 8.6pt; margin: 1pt 14pt 2pt 0; }
ol.flow img.logo { height: 13pt; margin-right: 3pt; }

/* ---- contents ---- */
.toc { columns: 2; column-gap: 26pt; margin-top: 4pt; }
.toc-ch { break-inside: avoid; margin-bottom: 6.5pt; }
.toc-ch > a { display: grid; grid-template-columns: 17pt 1fr auto; align-items: baseline; gap: 4pt; color: var(--navy); font-weight: 700;
  font-size: 9.4pt; padding-bottom: 1.6pt; border-bottom: .6pt solid var(--line); margin-bottom: 1.5pt; }
.toc-ch > a .n { color: var(--accent); font-feature-settings: "tnum"; }
.toc-ch .sub a { display: grid; grid-template-columns: 17pt 22pt 1fr auto; gap: 4pt; font-size: 7.8pt; line-height: 1.32; color: var(--ink2); padding: .45pt 0; }
.toc-ch .sub .n { color: var(--grey); font-feature-settings: "tnum"; }
.toc-fig { margin-top: 10pt; }

/* ---- cover ---- */
.cover { page: cover; height: 11in; width: 8.5in; position: relative; overflow: hidden; color: #fff; break-after: page;
  background: #1A3550; display: flex; flex-direction: column; justify-content: space-between; padding: 0.95in 0.85in 0.7in; }
.cover::before { content: ""; position: absolute; inset: 0; background:
  radial-gradient(1200px 700px at 105%% -8%%, #2F6A9E 0%%, #2F6A9E00 62%%), linear-gradient(170deg, #1F3B57 0%%, #172F47 100%%); }
.cover > * { position: relative; }
.cover .band { position: absolute; left: 0; top: 0; bottom: 0; width: .22in; background: #C96A2B; }
.cover .pattern { position: absolute; right: -0.25in; top: 3.35in; width: 4.6in; display: grid; grid-template-columns: repeat(6, 1fr); gap: 15pt 15pt; opacity: .17; }
.cover .pattern svg { width: 100%%; height: auto; }
.cover .mark { width: 58px; height: 58px; }
.cover .kicker { font: 600 9pt "SG Inter", Inter; letter-spacing: .2em; text-transform: uppercase; color: #F0B68A; margin: 20pt 0 8pt; }
.cover h1 { font: 700 44pt/1.02 "SG Inter Display", "Inter Display", "SG Inter", Inter, sans-serif; letter-spacing: -.025em; margin: 0 0 12pt; color: #fff; }
.cover .sub { font-size: 14.5pt; line-height: 1.35; color: #D6E4F0; max-width: 5.1in; }
.cover .facts { display: grid; grid-template-columns: repeat(4, auto); justify-content: start; gap: 0 26pt; margin: 0 0 22pt; }
.cover .facts div { border-top: 1.4pt solid #C96A2B; padding-top: 6pt; }
.cover .facts b { display: block; font: 700 19pt "SG Inter Display", "Inter Display", Inter; letter-spacing: -.01em; }
.cover .facts span { font-size: 8pt; color: #B9CDE0; }
.cover dl { display: grid; grid-template-columns: 1.15in auto; gap: 3pt 12pt; font-size: 9.4pt; margin: 12pt 0 0; }
.cover dt { color: #A9C3DB; } .cover dd { margin: 0; }
.cover .badge { display: inline-flex; align-items: center; gap: 5pt; border: 1.2pt solid #9FE0B5; color: #9FE0B5; padding: 2pt 9pt 2pt 7pt;
  border-radius: 4pt; font-weight: 700; letter-spacing: .12em; font-size: 8.6pt; }
.cover .badge svg { width: 10pt; height: 10pt; }
.cover .fine { font-size: 7.6pt; color: #A9C3DB; max-width: 5.6in; margin: 12pt 0 0; line-height: 1.45; }
.cover .stack { display: flex; gap: 9pt; align-items: center; margin-top: 14pt; }
.cover .stack img { height: 15pt; width: auto; filter: brightness(0) invert(1); opacity: .78; }

/* ---- screen view ---- */
@media screen {
  body { background: #E9EDF1; }
  main { background: #fff; padding: 0.55in 0.7in 0.7in; margin: 0 auto 28px; box-shadow: 0 1px 8px #0002; max-width: 8.5in; }
  .cover { margin: 24px auto 0; height: 10.2in; }
  .sec { margin-top: 44pt; } .sec:first-child { margin-top: 0; }
  figure.wide img { height: auto; }
  .xr, .ch-toc .pg, .toc .pg { display: none; }
}
"""

FONTS = [  # (family, file, weight, style): TrueType outlines, so the PDF embeds real fonts (the system's Inter is CFF, which
           # Chromium can only embed as Type 3 glyph programs)
    ("SG Inter", "Inter-Regular.woff2", 400, "normal"), ("SG Inter", "Inter-Italic.woff2", 400, "italic"),
    ("SG Inter", "Inter-Medium.woff2", 500, "normal"), ("SG Inter", "Inter-SemiBold.woff2", 600, "normal"),
    ("SG Inter", "Inter-Bold.woff2", 700, "normal"), ("SG Inter Display", "InterDisplay-Light.woff2", 300, "normal"),
    ("SG Inter Display", "InterDisplay-SemiBold.woff2", 600, "normal"), ("SG Inter Display", "InterDisplay-Bold.woff2", 700, "normal"),
    ("SG Mono", "JetBrainsMono-Regular.ttf", 400, "normal"), ("SG Mono", "JetBrainsMono-Italic.ttf", 400, "italic"),
    ("SG Mono", "JetBrainsMono-SemiBold.ttf", 600, "normal"), ("SG Mono", "JetBrainsMono-Bold.ttf", 700, "normal"),
]
FONT_CSS = "\n".join(f'@font-face {{ font-family: "{fam}"; src: url("fonts/{f}"); font-weight: {w}; font-style: {s}; }}'
                     for fam, f, w, s in FONTS)

ICON_CACHE: dict[str, str] = {}


def icon(name: str, colour: str = "currentColor", width: float = 2.0) -> str:
    """A Lucide icon as inline SVG in the given stroke colour."""
    if name not in ICON_CACHE:
        raw = (ROOT / "img" / "icons" / f"{name}.svg").read_text()
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
        raw = re.sub(r'\s(class|width|height)="[^"]*"', "", raw, count=3)
        ICON_CACHE[name] = raw
    return (ICON_CACHE[name].replace('stroke="#1F3B57"', f'stroke="{colour}"')
            .replace('stroke-width="2"', f'stroke-width="{width}"').replace("<svg", '<svg aria-hidden="true"', 1))


STATUS_ICON = {"st-prod": "circle-check", "st-prov": "circle-dot", "st-exp": "circle-dashed", "st-off": "circle-off", "st-new": "package"}
CALLOUTS = [(r"measured", "measured", "gauge", "Measured"), (r"limits?\b", "limits", "triangle-alert", "Limits"),
            (r"safeguards?", "safe", "shield-check", "Safeguards"), (r"gates?\b", "gate", "lock", "Gates"),
            (r"not (yet )?(done|validated)", "todo", "circle-dashed", "Not yet")]
SPEC_ICON = {"runs on": "server", "model": "brain", "models": "brain", "methods": "wrench", "method": "wrench", "code": "file-text",
             "front ends": "monitor", "needs": "plug", "stored": "database", "format": "package", "verify with": "badge-check"}
ADMON = {"important": ("adm-important", "triangle-alert", "#9A5B00"), "notice": ("adm-notice", "scale", "#9A5B00"),
         "caution": ("adm-important", "triangle-alert", "#9A5B00"), "note": ("adm-note", "info", "#2E7D45")}
LICENCES = [  # (pattern, class) in priority order
    (r"non-commercial research|non-commercial|academic use only|CC BY-NC 4\.0", "lic-nc"),
    (r"AGPL-3\.0|GPL-3\.0", "lic-copyleft"),
    (r"Apache-2\.0|BSD-3|MIT|ISC|CC0-1\.0|SIL OFL 1\.1|PostgreSQL licence", "lic-ok"),
    (r"not recorded in the project|weight provenance not recorded|not stated|no licence file[^;<]*|to be confirmed by the operator", "lic-unk"),
]
TERM_TITLES = {"console": ("Terminal", "terminal"), "sh": ("Terminal", "terminal"), "screen": ("Terminal UI", "keyboard"),
               "toml": ("Configuration file", "sliders-horizontal"), "tree": ("Package contents", "package"), "text": ("", "file-text")}


def front_matter(text):
    meta = {}
    if text.startswith("---\n"):
        end = text.index("\n---", 4)
        for line in text[4:end].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        text = text[end + 4:].lstrip("\n")
    return meta, text


def frag(markup: str) -> Tag:
    """The first element of an HTML fragment, as a detached tag."""
    return BeautifulSoup(markup, "html.parser").find()


def text_of(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True))


def css_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


class Doc:
    def __init__(self, md_path: pathlib.Path):
        self.meta, text = front_matter(md_path.read_text())
        text = re.sub(r"^# .*\n", "", text, count=1)                   # the cover carries the title
        body = markdown.markdown(text, extensions=["tables", "attr_list", "md_in_html", "toc", "sane_lists", "fenced_code"],
                                 extension_configs={"toc": {"permalink": False}})
        self.soup = BeautifulSoup(f"<main>{body}</main>", "html.parser")
        self.main = self.soup.main
        self.sections: list[dict] = []          # {id, num, title, el, subs: [{id, num, title}]}
        self.figures: list[tuple[str, str]] = []
        self.title = self.meta.get("title", md_path.stem)
        self.short = self.title.split(":")[0].strip()
        self.structure()
        self.decorate()
        self.contents()
        self.check_links()

    # ------------------------------------------------------------------ structure
    def structure(self) -> None:
        """Wrap each h2 and what follows it in a <section>; the part before the first h2 becomes the first one."""
        nodes = [n for n in list(self.main.children) if not (isinstance(n, NavigableString) and not n.strip())]
        cur, k = None, 0
        intro = self.soup.new_tag("section", attrs={"class": "sec front", "id": "about"})
        intro.append(frag('<h2 id="about-this-guide">About this guide</h2>'))
        self.main.clear()
        self.main.append(intro)
        cur = intro
        for n in nodes:
            if isinstance(n, Tag) and n.name == "h2":
                k += 1
                cur = self.soup.new_tag("section", attrs={"class": "sec", "id": f"s{k}"})
                self.main.append(cur)
            if isinstance(n, Tag) and n.name == "hr":
                continue
            cur.append(n)
        for sec in self.main.find_all("section", recursive=False):
            self.head(sec)

    def head(self, sec: Tag) -> None:
        h2 = sec.find("h2", recursive=False)
        img = h2.find("img")
        if img:
            img.extract()
        raw = text_of(h2)
        m = re.match(r"^(\d+)\.\s+(.*)$", raw)
        num, title = (m.group(1), m.group(2)) if m else ("", raw)
        if not num:
            sec["class"] = "sec front"
        info = {"id": h2["id"], "num": num, "title": title, "el": sec, "subs": [], "icon": img["src"] if img else ""}
        self.sections.append(info)
        page = f"p{len(self.sections)}"
        sec["style"] = f"page: {page}"
        info["page"] = page
        head = frag('<header class="ch-head"></header>')
        if num:
            head.append(frag(f'<div class="ch-num">{int(num):02d}</div>'))
        h2.clear()
        h2.append(title)
        wrap = self.soup.new_tag("div")
        h2.replace_with(head)
        wrap.append(h2)
        head.append(wrap)
        if img:
            img["alt"] = ""
            head.append(frag(f'<div class="ch-ico">{img}</div>'))
        # lead paragraph: the first paragraph of a numbered chapter
        nxt = head.find_next_sibling()
        if num and nxt is not None and nxt.name == "p":
            nxt["class"] = "lead"
        # numbered sub-sections
        for h3 in sec.find_all("h3", recursive=False):
            himg = h3.find("img")
            if himg:
                himg.extract()
            t = text_of(h3)
            m3 = re.match(r"^(\d+\.\d+|Appendix [A-Z])\.?\s+(.*)$", t)
            n3, t3 = (m3.group(1), m3.group(2)) if m3 else ("", t)
            info["subs"].append({"id": h3["id"], "num": n3.replace("Appendix ", ""), "title": t3})
            h3.clear()
            side = frag('<span class="h-side"></span>')
            side.append(frag(f'<span class="h-num">{html.escape(n3.replace("Appendix ", ""))}</span>'))
            if himg:
                himg["alt"] = ""
                side.append(frag(f'<span class="h-ico">{himg}</span>'))
            h3.append(side)
            h3.append(frag(f'<span class="h-t">{html.escape(("Appendix " + n3[-1] + ". " if n3.startswith("Appendix") else "") + t3)}</span>'))
        subs = [s for s in info["subs"] if s["num"]]
        first_h3 = sec.find("h3", recursive=False)
        own = {s["id"] for s in subs}
        linked = {a["href"][1:] for el in (first_h3.find_previous_siblings("table") if first_h3 is not None else [])
                  for a in el.select('a[href^="#"]')} & own
        if num and len(subs) >= 3 and len(linked) * 2 < len(subs):      # a chapter with its own overview table needs no second list
            nav = frag('<nav class="ch-toc" aria-label="In this chapter"></nav>')
            for s in subs:
                nav.append(frag(f'<a href="#{s["id"]}"><span class="n">{s["num"]}</span><span class="t">{html.escape(s["title"])}</span>'
                                f'<span class="pg" data-pg="{s["id"]}">00</span></a>'))
            (first_h3.insert_before(nav) if first_h3 is not None else sec.append(nav))

    # ------------------------------------------------------------------ decoration
    def decorate(self) -> None:
        m = self.main
        for img in m.find_all("img"):
            src = img.get("src", "")
            if "logos/" in src:
                img["class"] = "logo"
                for a in ("width", "height"):
                    img.attrs.pop(a, None)
            elif "icons/" in src and img.find_parent(class_=["ch-ico", "h-ico"]) is None:
                img["class"] = "ico"
                for a in ("width", "height"):
                    img.attrs.pop(a, None)
        for span in m.select("span.st"):
            cls = next((c for c in span.get("class", []) if c in STATUS_ICON), None)
            if cls:
                span.insert(0, frag(icon(STATUS_ICON[cls], "currentColor", 2.4)))
        for span in m.select("span.idf"):                              # written by hand in the legend
            if span.find("svg") is None:
                span.insert(0, frag(icon("fingerprint" if "idf-yes" in span["class"] else "link", "currentColor", 2.2)))
        for span in m.select("span.mk"):
            kind = next((c[3:] for c in span["class"] if c.startswith("mk-")), "measured")
            ic = next((c[2] for c in CALLOUTS if c[1] == kind), "gauge")
            span.insert(0, frag(icon(ic, "currentColor", 2.3)))
        self.tables()
        self.callouts()
        self.notices()
        self.figures_()
        self.listings()
        for td in m.select("td"):
            kids = [c for c in td.children if not (isinstance(c, NavigableString) and not c.strip())]
            if len(kids) == 1 and isinstance(kids[0], Tag) and kids[0].name == "img" and "ico" in kids[0].get("class", []):
                td["class"] = "c-ico"
        # cross-references in running text get their page number
        for a in m.select('a[href^="#"]'):
            if a.find_parent(class_=["ch-toc", "toc"]) is not None:
                continue
            a.insert_after(frag(f'<span class="xr">&nbsp;(p.&nbsp;<span data-pg="{a["href"][1:]}">00</span>)</span>'))

    def tables(self) -> None:
        for table in self.main.find_all("table"):
            heads = [text_of(th) for th in table.select("thead th")]
            if heads and not any(heads):                                # empty header row: label | value
                prev = table.find_previous_sibling()
                rows = [(text_of(tr.find_all("td")[0]), tr.find_all("td")[1]) for tr in table.select("tbody tr")]
                if prev is not None and prev.name == "h3" and len(rows) <= 4:
                    dl = frag('<dl class="spec"></dl>')
                    for label, cell in rows:
                        key = label.lower()
                        long = len(text_of(cell)) > 46
                        item = frag(f'<div class="{"long" if long else ""}"></div>')
                        ic = icon(SPEC_ICON[key], "#5B6670", 2.2) if key in SPEC_ICON else ""
                        item.append(frag(f"<dt>{ic}{html.escape(label)}</dt>"))
                        dd = self.soup.new_tag("dd")
                        for c in list(cell.contents):
                            dd.append(c)
                        item.append(dd)
                        dl.append(item)
                    table.replace_with(dl)
                else:
                    table["class"] = "facts"
                continue
            if "Identity-sensitive" in heads:
                col = heads.index("Identity-sensitive")
                for tr in table.select("tbody tr"):
                    td = tr.find_all("td")[col]
                    t = text_of(td)
                    kind, ic = (("idf-yes", "fingerprint") if t.startswith("Yes") else ("idf-ind", "link") if t.startswith("Indirect")
                                else ("idf-no", "circle-off"))
                    inner = "".join(str(c) for c in td.contents)
                    td.clear()
                    td.append(frag(f'<span class="idf {kind}">{icon(ic, "currentColor", 2.2)}<span>{inner}</span></span>'))
            if any(h.startswith("Licence") for h in heads):
                col = next(i for i, h in enumerate(heads) if h.startswith("Licence"))
                for tr in table.select("tbody tr"):
                    td = tr.find_all("td")[col]
                    markup = "".join(str(c) for c in td.contents).replace("<strong>", "").replace("</strong>", "")
                    done: list[str] = []

                    def chip(mo, cls=""):
                        done.append(f'<span class="lic {cls}">{mo.group(0)}</span>')
                        return f"\x00{len(done) - 1}\x00"
                    for pat, cls in LICENCES:
                        markup = re.sub(pat, lambda mo, cls=cls: chip(mo, cls), markup)
                    markup = re.sub(r"\x00(\d+)\x00", lambda mo: done[int(mo.group(1))], markup)
                    td.clear()
                    td.append(BeautifulSoup(markup, "html.parser"))
            if len(heads) == 2 and table.find_parent("section") is not None and "wide" not in table.get("class", []):
                if max((len(text_of(td)) for td in table.select("tbody td")), default=0) < 100:
                    table["class"] = "narrow"

    def callouts(self) -> None:
        for p in self.main.select("section > p"):
            first = next((c for c in p.children if not (isinstance(c, NavigableString) and not c.strip())), None)
            if not (isinstance(first, Tag) and first.name == "strong"):
                continue
            label = text_of(first).rstrip(":.").strip()
            hit = next((c for c in CALLOUTS if re.match(c[0], label, flags=re.I)), None)
            if hit is None or len(label) > 34:
                continue
            _, kind, ic, _name = hit
            first.extract()
            body = frag('<div class="co-b"></div>')
            rest = "".join(str(c) for c in p.contents).strip()
            rest = re.sub(r"^[:.]\s*", "", rest)
            if rest:
                rest = rest[0].upper() + rest[1:] if not rest.startswith("(") else rest
                body.append(BeautifulSoup(f"<p>{rest}</p>", "html.parser"))
            nxt = p.find_next_sibling()
            if not rest and nxt is not None and nxt.name in ("ul", "ol"):
                body.append(nxt.extract())
            box = frag(f'<div class="co co-{kind}"><div class="co-l">{icon(ic, "currentColor", 2.3)}<span>{html.escape(label)}</span></div></div>')
            box.append(body)
            p.replace_with(box)

    def notices(self) -> None:
        for bq in self.main.find_all("blockquote"):
            strong = bq.find("strong")
            word = text_of(strong).rstrip(".:").lower() if strong else ""
            cls, ic, colour = ADMON.get(word, ("adm-info", "info", "#2E6DA4"))
            box = frag(f'<aside class="adm {cls}">{icon(ic, colour, 2.2)}<div></div></aside>')
            for c in list(bq.contents):
                box.div.append(c)
            bq.replace_with(box)
        for div in self.main.select("div.note"):
            box = frag(f'<aside class="adm adm-note">{icon("info", "#2E7D45", 2.2)}<div></div></aside>')
            inner = "".join(str(c) for c in div.contents).strip()
            box.div.append(BeautifulSoup(inner if inner.startswith("<p") else f"<p>{inner}</p>", "html.parser"))
            div.replace_with(box)

    def figures_(self) -> None:
        for fig in self.main.find_all("figure"):
            cap = fig.find("figcaption")
            if cap is None:
                continue
            t = "".join(str(c) for c in cap.contents).strip()
            mo = re.match(r"^Figure (\d+)\.\s*(.*)$", t, flags=re.S)
            if mo:
                fig["id"] = f"figure-{mo.group(1)}"
                self.figures.append((mo.group(1), re.sub(r"<[^>]+>", "", mo.group(2))))
                cap.clear()
                cap.append(BeautifulSoup(f'<span class="fig-n">Figure {mo.group(1)}</span><span>{mo.group(2)}</span>', "html.parser"))
            if "wide" in fig.get("class", []):
                sec = fig.find_parent("section")
                fig["style"] = f"page: {sec['style'].split(':')[1].strip()}w"

    def listings(self) -> None:
        for pre in self.main.find_all("pre"):
            code = pre.find("code")
            if code is None:
                continue
            lang = next((c[9:] for c in code.get("class", []) if c.startswith("language-")), "text")
            title, ic = TERM_TITLES.get(lang, (lang, "file-text"))
            raw = code.get_text()
            if raw.startswith("# title: "):
                first, raw = raw.split("\n", 1)
                title = first[9:].strip()
            lines = []
            for line in raw.rstrip("\n").split("\n"):
                esc = html.escape(line)
                if lang in ("console", "sh") and line.startswith("$ "):
                    cmd, _, comment = esc[2:].partition("   # ")
                    esc = f'<span class="pr">$ </span><span class="cmd">{cmd}</span>' + (f'<span class="cm">   # {comment}</span>' if comment else "")
                elif lang in ("console", "sh", "toml") and line.lstrip().startswith("#"):
                    esc = f'<span class="cm">{esc}</span>'
                lines.append(esc)
            box = frag(f'<div class="term {"screen" if lang == "screen" else ""}"></div>')
            if title:
                box.append(frag(f'<div class="term-h">{icon(ic, "currentColor", 2.2)}<span>{html.escape(title)}</span></div>'))
            box.append(BeautifulSoup("<pre><code>" + "\n".join(lines) + "</code></pre>", "html.parser"))
            pre.replace_with(box)

    # ------------------------------------------------------------------ contents page
    def contents(self) -> None:
        sec = next((s for s in self.sections if s["title"].lower() == "contents"), None)
        if sec is None:
            return
        el = sec["el"]
        for child in list(el.children):
            if isinstance(child, Tag) and "ch-head" in child.get("class", []):
                continue
            child.extract()
        toc = frag('<nav class="toc" aria-label="Contents"></nav>')
        for s in self.sections:
            if s is sec or s["id"] == "about-this-guide":
                continue
            block = frag('<div class="toc-ch"></div>')
            block.append(frag(f'<a href="#{s["id"]}"><span class="n">{s["num"]}</span><span>{html.escape(s["title"])}</span>'
                              f'<span class="pg" data-pg="{s["id"]}">00</span></a>'))
            sub = frag('<div class="sub"></div>')
            for h in s["subs"]:
                if not h["num"]:
                    continue
                sub.append(frag(f'<a href="#{h["id"]}"><span></span><span class="n">{h["num"]}</span><span>{html.escape(h["title"])}</span>'
                                f'<span class="pg" data-pg="{h["id"]}">00</span></a>'))
            if sub.find("a"):
                block.append(sub)
            toc.append(block)
        el.append(toc)

    def check_links(self) -> None:
        ids = {t["id"] for t in self.main.find_all(id=True)}
        bad = sorted({a["href"] for a in self.main.select('a[href^="#"]') if a["href"][1:] not in ids})
        if bad:
            sys.exit("links to sections that do not exist: " + ", ".join(bad))

    # ------------------------------------------------------------------ output
    def cover(self) -> str:
        meta = self.meta
        mark = icon("squirrel", "#FFFFFF", 1.6).replace("<svg", '<svg class="mark"', 1)
        pattern = "".join(icon(n, "#FFFFFF", 1.4) for n in (
            "cctv", "squirrel", "scan-face", "footprints", "car", "shield-check", "route", "users", "file-lock", "bell", "ruler", "boxes",
            "eye", "terminal", "fingerprint", "stamp", "activity", "layers", "lock", "gauge", "person-standing", "video", "database", "scale",
            "map", "brain", "clipboard-check", "hash", "volume-2", "search"))
        two = next((s["el"] for s in self.sections if s["num"] == "2"), None)
        caps = len(two.select("table tbody tr")) if two is not None else 0
        chapters = sum(1 for s in self.sections if s["num"])
        facts = [(str(caps), "capabilities, each with<br>measured limits"), (str(chapters), "chapters"),
                 (str(len(self.figures)), "figures and<br>screenshots"), ("0", "cloud services<br>in the data path")]
        stack = "".join(f'<img src="img/logos/{n}.svg" alt="">' for n in ("python", "docker", "postgresql", "nginx", "onnx", "opencv", "grafana", "mqtt", "linux"))
        return f"""<section class="cover"><div class="band"></div><div class="pattern">{pattern}</div>
  <div>{mark}
    <div class="kicker">{html.escape(self.title.split(':', 1)[1].strip() if ':' in self.title else '')}</div>
    <h1>{html.escape(self.short)}</h1>
    <div class="sub">{html.escape(meta.get('subtitle', ''))}</div>
  </div>
  <div>
    <div class="facts">{''.join(f'<div><b>{a}</b><span>{b}</span></div>' for a, b in facts)}</div>
    <span class="badge">{icon('shield-check', '#9FE0B5', 2.4)}{html.escape(meta.get('classification', 'PUBLIC').upper())}</span>
    <dl>
      <dt>Version</dt><dd>{html.escape(meta.get('version', ''))}</dd>
      <dt>Date</dt><dd>{html.escape(meta.get('date', ''))}</dd>
      <dt>Distribution</dt><dd>Unrestricted. Contains no addresses, credentials, personal data or site imagery.</dd>
    </dl>
    <div class="stack">{stack}</div>
    <p class="fine">This guide describes software capabilities and their measured limits. It is not legal advice. Several
    capabilities depend on third-party model weights with non-commercial licences; see the chapter on third-party models and
    licences. Product logos are trademarks of their owners and identify the software only.</p>
  </div>
</section>"""

    def css(self) -> str:
        rules = []
        for s in self.sections:
            label = (f'{s["num"]}   {s["title"]}' if s["num"] else s["title"])
            rules.append(f'@page {s["page"]} {{ @top-right {{ content: "{css_str(label)}"; }} }}')
            rules.append(f'@page {s["page"]}w {{ size: 11in 8.5in; margin: 0.8in 0.7in 0.78in 0.7in; @top-right {{ content: "{css_str(label)}"; }} }}')
        footer = f'Version {self.meta.get("version", "")}  ·  {self.meta.get("date", "")}  ·  {self.meta.get("classification", "Public")}'
        return CSS % {"pages": "\n".join(rules), "gutter": GUTTER, "footer": css_str(footer), "flowcols": "4"}

    def html(self, pages: dict[str, int] | None = None) -> str:
        soup = BeautifulSoup(str(self.main), "html.parser")
        if pages:
            for el in soup.select("[data-pg]"):
                n = pages.get(el["data-pg"])
                if n is None:
                    el.parent.extract() if "xr" in (el.parent.get("class") or []) else el.clear()
                else:
                    el.string = str(n)
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title>
<meta name="author" content="squirrel-guard project"><meta name="description" content="{html.escape(self.meta.get('subtitle', ''))}">
<style>{FONT_CSS}{self.css()}</style></head>
<body>{self.cover()}{soup}</body></html>"""


def find_chrome():
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    for c in sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell")), reverse=True) + \
            [shutil.which(n) for n in ("chromium", "chromium-browser", "google-chrome", "chrome")]:
        if c and os.path.exists(c):
            return c
    sys.exit("no Chromium found: set CHROME=/path/to/chrome")


def print_pdf(html_path: pathlib.Path, pdf: pathlib.Path) -> None:
    subprocess.run([find_chrome(), "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    "--generate-pdf-document-outline", "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True, timeout=240)


def heading_pages(pdf: pathlib.Path) -> dict[str, int]:
    """Page number of every link target, from the PDF's named destinations."""
    reader = PdfReader(str(pdf))
    out = {}
    for name, dest in reader.named_destinations.items():
        try:
            out[name.lstrip("/")] = reader.get_destination_page_number(dest) + 1
        except Exception:
            pass
    return out


def main():
    BUILD.mkdir(exist_ok=True)
    for md in [ROOT / "FEATURE-GUIDE.md"]:
        doc = Doc(md)
        out_html = ROOT / f"{md.stem}.html"                            # beside the .md so relative image paths work
        pdf = BUILD / f"{md.stem}.pdf"
        out_html.write_text(doc.html())
        print_pdf(out_html, pdf)
        pages = heading_pages(pdf)
        if not pages:
            sys.exit("the PDF has no named destinations: page numbers cannot be filled in (Chromium too old?)")
        out_html.write_text(doc.html(pages))
        print_pdf(out_html, pdf)
        again = heading_pages(pdf)
        moved = [k for k in pages if again.get(k) != pages[k]]
        if moved:
            sys.exit(f"page numbers moved between the two prints ({len(moved)} headings): {moved[:5]}")
        n = len(PdfReader(str(pdf)).pages)
        shutil.move(out_html, BUILD / out_html.name)                   # a record of what was printed (links are relative to docs/public)
        print(f"{datetime.datetime.now():%H:%M:%S} wrote {pdf} ({n} pages, {pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
