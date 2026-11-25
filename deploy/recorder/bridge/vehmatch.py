"""Vehicle matching that explains itself: plate text, appearance, colour and body type, combined into a band.

  plate_similarity(a, b, conf_a, conf_b) -> {"score", "text"}     fuzzy plate comparison
  explain(new, old, appearance_cos)       -> {"band", "pct", "reasons", "decision"}

Plate comparison counts edits between the two readings. Swapping characters that cameras and readers confuse
(0/O/D/Q, 1/I/L/7, 2/Z, 5/S, 8/B, 6/G, 4/A, U/V) costs less than other swaps, and a character the reader itself
was unsure of (confidence < 0.6) costs less again. A missing first or last character (cut off at the frame edge)
costs half. Score = 1 - cost / length.

The bands are this system's own scale, not error rates: no labelled vehicle data has calibrated them. Standard
library only (numpy not needed); copies live in deploy/recorder/bridge/ and deploy/recorder/persons-ui/.
"""
from __future__ import annotations

CONFUSABLE = [set("0ODQ"), set("1IL7"), set("2Z"), set("5S"), set("8B"), set("6G"), set("4A"), set("UV"), set("MN"), set("CG")]
LOW_CONF = 0.6
NIGHT_WORDS = ("unknown", "infrared", "")
BASIC = {"white": "white", "silver": "grey", "grey": "grey", "gray": "grey", "black": "black", "red": "red", "maroon": "red",
         "orange": "orange", "gold": "yellow", "golden": "yellow", "yellow": "yellow", "beige": "brown", "tan": "brown",
         "brown": "brown", "green": "green", "blue": "blue", "navy": "blue", "purple": "purple", "pink": "red"}


def norm_plate(p: str | None) -> str | None:
    p = "".join(ch for ch in (p or "").upper() if ch.isalnum())
    return p or None


def _sub_cost(x: str, y: str, cx: float, cy: float) -> float:
    if x == y:
        return 0.0
    base = 0.35 if any(x in g and y in g for g in CONFUSABLE) else 1.0
    if min(cx, cy) < LOW_CONF:
        base *= 0.6
    return base


def plate_similarity(a: str | None, b: str | None, conf_a: list | None = None, conf_b: list | None = None) -> dict | None:
    """None when either reading is missing. conf_*: per-character confidence of each reading, when known."""
    a, b = norm_plate(a), norm_plate(b)
    if not a or not b:
        return None
    ca = list(conf_a or [1.0] * len(a))[:len(a)] + [1.0] * max(0, len(a) - len(conf_a or []))
    cb = list(conf_b or [1.0] * len(b))[:len(b)] + [1.0] * max(0, len(b) - len(conf_b or []))
    n, m = len(a), len(b)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + (0.5 if i == 1 or i == n else 1.0)
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + (0.5 if j == 1 or j == m else 1.0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            edge_a = 0.5 if i in (1, n) else 1.0
            edge_b = 0.5 if j in (1, m) else 1.0
            D[i][j] = min(D[i - 1][j] + edge_a, D[i][j - 1] + edge_b, D[i - 1][j - 1] + _sub_cost(a[i - 1], b[j - 1], ca[i - 1], cb[j - 1]))
    cost = D[n][m]
    score = max(0.0, 1.0 - cost / max(n, m))
    if cost == 0:
        text = f"plate {a} read the same both times"
    elif cost <= 0.7:
        text = f"plates {a} and {b} differ only in characters that are easily misread"
    elif cost <= 1.5:
        text = f"plates {a} and {b} differ in about one character"
    else:
        text = f"plates {a} and {b} differ"
    return {"score": round(score, 3), "cost": round(cost, 2), "text": text}


def basic_colour(c: str | None) -> str | None:
    """'dark metallic blue' -> 'blue'; None for unknown or night (infrared) readings."""
    words = str(c or "").lower().replace("/", " ").replace("-", " ").split()
    for w in reversed(words):
        if w in BASIC:
            return BASIC[w]
    return None


def explain(new: dict, old: dict, appearance_cos: float | None) -> dict:
    """new/old: {"plate", "plate_conf" (per char), "colour", "colour_night" (bool), "body"}. Returns the decision with
    reasons. decision: "same" (file under old), "possible" (keep separate, flag), "different"."""
    reasons, pts = [], 0.0
    ps = plate_similarity(new.get("plate"), old.get("plate"), new.get("plate_conf"), old.get("plate_conf"))
    if ps:
        reasons.append({"code": "PLATE", "text": ps["text"], "value": ps["score"]})
        if ps["score"] >= 0.95:
            pts += 4.0
        elif ps["score"] >= 0.85:
            pts += 2.5
        elif ps["score"] >= 0.7:
            pts += 0.5
        else:
            pts -= 4.0                                           # two clear, different plates: different vehicles
    if appearance_cos is not None:
        if appearance_cos >= 0.85:
            pts += 2.0; t = "looks very similar"
        elif appearance_cos >= 0.80:                             # the bridge's match threshold before this module
            pts += 1.4; t = "looks similar"
        elif appearance_cos >= 0.75:
            pts += 0.8; t = "looks fairly similar"
        elif appearance_cos >= 0.65:
            pts += 0.2; t = "looks somewhat similar"
        else:
            pts -= 1.0; t = "looks different"
        reasons.append({"code": "APPEARANCE", "text": f"{t} (appearance {appearance_cos:.2f})", "value": round(appearance_cos, 3)})
    cn, co = basic_colour(new.get("colour")), basic_colour(old.get("colour"))
    if new.get("colour_night") or old.get("colour_night"):
        reasons.append({"code": "COLOUR_UNKNOWN", "text": "colour not compared: night/infrared image"})
    elif cn and co:
        near = {frozenset(("grey", "white")), frozenset(("grey", "black")), frozenset(("brown", "red")), frozenset(("orange", "red")),
                frozenset(("yellow", "orange")), frozenset(("blue", "grey"))}
        if cn == co:
            pts += 0.5; reasons.append({"code": "COLOUR_SAME", "text": f"same colour ({cn})"})
        elif frozenset((cn, co)) in near:
            reasons.append({"code": "COLOUR_NEAR", "text": f"similar colours ({cn} / {co}): lighting can shift these"})
        else:
            pts -= 1.5; reasons.append({"code": "COLOUR_DIFFERENT", "text": f"different colours ({cn} / {co})"})
    bn, bo = (new.get("body") or "").lower() or None, (old.get("body") or "").lower() or None
    if bn and bo:
        if bn == bo:
            pts += 0.3; reasons.append({"code": "BODY_SAME", "text": f"same body type ({bn})"})
        else:
            pts -= 1.0; reasons.append({"code": "BODY_DIFFERENT", "text": f"different body types ({bn} / {bo})"})
    p = 1 / (1 + 2.718281828 ** (-(pts - 1.0)))
    pct = round(100 * p)
    band = "strong" if p >= 0.9 else ("likely" if p >= 0.7 else ("possible" if p >= 0.45 else "unlikely"))
    plate_says_same = ps is not None and ps["score"] >= 0.85
    plate_says_diff = ps is not None and ps["score"] < 0.7
    if plate_says_diff:
        decision = "different"
    elif plate_says_same or band in ("strong", "likely") and (ps is not None or (appearance_cos or 0) >= 0.8):
        decision = "same"
    elif band in ("likely", "possible"):
        decision = "possible"
    else:
        decision = "different"
    return {"band": band, "pct": pct, "points": round(pts, 2), "decision": decision, "reasons": reasons}
