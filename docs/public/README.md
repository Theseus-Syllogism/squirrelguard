# squirrel-guard public documentation

Material prepared for public release. It contains **no addresses, hostnames, domains, credentials, personal data
or site imagery**: machines are described by role (camera network, recorder host, GPU server, operator
workstation). Example photos and the 3D replay demo are stock material; screenshots of the reference
installation are redacted in the page before capture (people, faces and rooms blurred, names and camera names
replaced, biometric measurements blurred).

| File | What it is |
|---|---|
| [`FEATURE-GUIDE.md`](FEATURE-GUIDE.md) | Feature and Capability Guide: every feature, its measured performance, limits, safeguards and licences |
| [`FEATURE-GUIDE.pdf`](../../FEATURE-GUIDE.pdf) (repository root) | The same guide as a print-ready PDF (US Letter, 66 pages, bookmarks, page-numbered contents and cross-references) |
| `img/screens/` | Redacted screenshots of the review interfaces (`tools/capture_screens.cjs`), and console and evidence screenshots from test data with example addresses (`tools/capture_console.py`) |
| `img/logos/` | Software logos (Simple Icons, CC0) and generated letter tiles (`tools/make_logos.py`) |
| `fonts/` | Inter and JetBrains Mono (SIL OFL 1.1), embedded in the PDF |
| [`diagrams/squirrel-guard-architecture.drawio`](diagrams/squirrel-guard-architecture.drawio) | Editable draw.io file with four pages (below) |
| `diagrams/*.png`, `diagrams/*.svg` | Exports of each page |

Diagram pages:

1. **Software stack**: every component by layer, coloured by the machine role that runs it.
2. **Event pipeline**: one event from camera frame to alert, with the people and vehicle lane.
3. **Deployment and trust boundaries**: network zones and permitted connections.
4. **Privacy and retention**: retention periods and the rules the software enforces.

## Rebuilding

The diagrams and the PDF are generated, so edit the sources, not the outputs:

```
python3 docs/public/tools/make_diagrams.py --export     # diagrams: .drawio + PNG/SVG (needs the drawio CLI)
python3 docs/public/tools/make_logos.py [--fetch]      # software logos and letter tiles
SG_SCREENSHOT_DIR=/tmp/pages pytest tests/test_evidence_export.py
python3 docs/public/tools/capture_console.py --pages /tmp/pages   # console and evidence screenshots (needs playwright-core)
pip install markdown==3.7 beautifulsoup4 pypdf
python3 docs/public/tools/build_docs.py                 # FEATURE-GUIDE.md -> build/FEATURE-GUIDE.html + .pdf
cp docs/public/build/FEATURE-GUIDE.pdf .   # the PDF lives in the repository root
```

`build_docs.py` prints with headless Chromium (set `CHROME=/path/to/chrome` if it is not found) twice: the first
print gives the page of every heading, the second carries those numbers in the contents and cross-references.
Landscape diagrams get landscape pages.

### Writing conventions the build understands

The Markdown stays plain; the build turns these patterns into the designed elements:

| In the Markdown | In the PDF |
|---|---|
| `## N. Title <img src="img/icons/x.svg">` and a first paragraph | chapter opener with number, icon, lead paragraph and a contents list |
| `### N.M Title <img …>` then a two-column table with an empty header (Status, Runs on, Model…) | section heading with number and icon in the margin, then a summary strip |
| a paragraph starting `**Measured:**`, `**Limits:**`, `**Safeguards:**` | a labelled block with the label in the margin |
| `> **Important.**`, `> **Notice.**`, `> **Note.**` | a notice box with an icon |
| `<span class="st st-prod">…</span>` (`st-prov`, `st-exp`, `st-off`, `st-new`) | a status chip with a symbol |
| a table column named `Identity-sensitive` or `Licence…` | coloured chips for identity and licence |
| a fenced block marked `console`, `toml`, `tree` or `screen`, optional first line `# title: …` | a titled listing; `$` lines are commands |
| `<img src="img/logos/x.svg" width="16">` | a software logo in the text |
| `<kbd>`, `<span class="tag">CLI</span>` | key caps and small tags |

## Before publishing a new version

- Re-check every measured figure against the evaluation notes in `docs/` (the guide quotes them; it does not
  re-measure).
- Search the folder for addresses, hostnames and personal names (`grep -rIiE` over `*.md *.py *.drawio`; the PDF
  with `pdftotext`).
- Do not add camera frames from a real site or of real people. Stock images need their licence recorded in
  Appendix B of the guide.

Icons: [Lucide](https://lucide.dev), ISC licence (`icons/LICENSE-lucide-ISC.txt`). Logos: [Simple Icons](https://simpleicons.org), CC0; trademarks of their owners.
