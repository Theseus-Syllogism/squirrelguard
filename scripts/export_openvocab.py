"""Export an open-vocabulary YOLO (YOLO-World v2 by default) to ONNX with the text vocabulary baked in.

Usage: .venv/bin/python scripts/export_openvocab.py [--vocab scripts/openvocab_vocab.txt] [--model yolov8s-worldv2.pt]
                                                    [--imgsz 480] [--out models/yolov8s-worldv2-sg.onnx]
  --vocab   a vocabulary file (format in scripts/openvocab_vocab.txt) or a comma-separated list ("ladder,knife")
  --model   an Ultralytics YOLO-World .pt (yolov8{s,m,l,x}-worldv2.pt; only yolov8s-worldv2 was evaluated), downloaded
            from the Ultralytics GitHub releases on first use. YOLOE .pt files take the same path but are UNTESTED.
Output: models/<model>-sg.onnx
  input  "images": float32 (1, 3, imgsz, imgsz), RGB 0-1, letterboxed (pad 114). 480 by default: the runtime looks
         at a window around the person, and 480 found the same objects as 640 for ~40 % less CPU (docs/openvocab.md)
  output float32 (1, 4 + n_prompts, anchors): cx, cy, w, h in input pixels, then one sigmoid score per prompt
  metadata: "names" (Ultralytics: prompt index -> prompt text), "sg_labels" (JSON list: the reported label per
            prompt, null for distractors), "sg_vocab" (the vocabulary file text), "sg_source" (model + date)

How it works: YOLO-World scores every box against CLIP text embeddings of the prompts. set_classes() computes those
embeddings once (CLIP ViT-B/32 text tower, downloaded by Ultralytics' `clip` package) and stores them in the model,
so the exported ONNX graph is a plain detector: no CLIP and no tokenizer at runtime. Changing the vocabulary means
running this script again (~1 min on CPU).

Licences: Ultralytics code and the YOLO-World/YOLOE weights they distribute are AGPL-3.0 (same as the other YOLO
models in this project: fine for this private system; distributing it as a product needs an Ultralytics licence).
YOLO-World itself (Tencent AILab, github.com/AILab-CVC/YOLO-World) is GPL-3.0; the CLIP text encoder used only
at export time is MIT (OpenAI).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_vocab(spec: str) -> tuple[list[str], list[str | None], str]:
    """-> (prompts, reported label per prompt or None for distractors, source text)."""
    p = Path(spec)
    text = p.read_text() if p.is_file() else "\n".join(s.strip() for s in spec.split(","))
    prompts, labels = [], []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        prompt, _, label = (s.strip() for s in line.partition("="))
        if prompt.lower() in (q.lower() for q in prompts):
            raise SystemExit(f"duplicate prompt: {prompt!r}")
        prompts.append(prompt)
        labels.append(None if label == "-" else (label or prompt))
    if not any(labels):
        raise SystemExit("vocabulary has no reported labels")
    return prompts, labels, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(ROOT / "scripts" / "openvocab_vocab.txt"))
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=480)   # we run on a window around the person: 480 is enough
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnx
    from ultralytics import YOLO

    prompts, labels, text = read_vocab(args.vocab)
    out = Path(args.out) if args.out else ROOT / "models" / f"{Path(args.model).stem}-sg.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src = Path(args.model)
        if not src.exists() and (out.parent / src.name).exists():
            src = out.parent / src.name               # weights downloaded by an earlier run
        fresh = not src.exists()                      # Ultralytics downloads a bare name into the working directory
        model = YOLO(str(src))
        if "yoloe" in src.name.lower():               # YOLOE: text embeddings passed explicitly
            model.set_classes(prompts, model.get_text_pe(prompts))
        else:                                         # YOLO-World
            model.set_classes(prompts)
        tmp_pt = Path(td) / "vocab.pt"
        model.save(tmp_pt)
        onnx_path = Path(YOLO(str(tmp_pt)).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True,
                                                  dynamic=False))
        m = onnx.load(str(onnx_path))
        meta = {p.key: p for p in m.metadata_props}
        for k, v in (("sg_labels", json.dumps(labels)), ("sg_vocab", text),
                     ("sg_source", f"{src.name} imgsz={args.imgsz} exported {dt.date.today()}")):
            if k in meta:
                meta[k].value = v
            else:
                m.metadata_props.add(key=k, value=v)
        onnx.save(m, str(out))
        if fresh and Path(src.name).exists():
            shutil.move(src.name, out.parent / src.name)   # keep the downloaded .pt next to the other weights
    reported = sorted({lab for lab in labels if lab})
    print(f"exported {out}: {len(prompts)} prompts, {len(reported)} reported labels "
          f"({', '.join(reported)}), {labels.count(None)} distractors")


if __name__ == "__main__":
    main()
