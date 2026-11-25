"""What is a person doing: ST-GCN++ on a 2D keypoint track (COCO-17, as from pose.py) -> NTU RGB+D 120 actions.

Model: PYSKL ST-GCN++ joint stream trained on NTU RGB+D 120 XSub with HRNet 2D keypoints (84.4 % top-1 there),
exported to ONNX by scripts/export_stgcnpp.py. Preprocessing reproduces pyskl's test pipeline
(configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py) in numpy:
  PreNormalize2D        x, y -> (p - size/2) / (size/2) using the image size; joints with score <= 0.01 get x = y = 0
  GenSkeFeat(coco, j)   channels = (x, y, score): the keypoint confidence IS the third input channel
  UniformSample(100)    test mode, seed 255: < 100 frames -> a looped contiguous window; 100..199 -> drop random
                        frames; >= 200 -> one random frame per segment (so a long track is sped up)
  FormatGCNInput(2)     a single person is padded with an all-zero second person
and averages softmax over clips ('prob'), like RecognizerGCN.forward_test. pyskl tests with 10 clips; this
defaults to 3 (~90 ms CPU per clip single-threaded on an i7-14700HX): with < 100 frames the clips are only shifted
copies of the same loop, and on test tracks 3 vs 10 clips moved the top-1 probability by ~0.03.
A whole track goes through as one sample, as pyskl would do it: a 20 s track is squeezed into 100 frames (6x
speed-up), which the model never saw (NTU actions last ~2-5 s). For anything longer than ~5 s use windows().

Our additions (not in pyskl): NaN or missing frames are zero-confidence; interior gaps up to `max_gap_s` are
linearly interpolated and leading/trailing empty frames are dropped; the track is resampled to 30 fps (NTU's rate)
when `fps` is given, so the motion speed matches what the model saw.

LIMITS: NTU is indoor, lab-recorded, mostly front-facing actors 2-4 m from a Kinect, performing one scripted action
per clip. Expect weak transfer to an outdoor camera looking down at a yard: people seen from the side or back,
small (a distant person is a few % of the frame; NTU actors are ~60 % of the frame height), partly occluded, doing nothing in
particular (there is no "walking", "standing" or "lying" class: on test clips side-on walking came out as
"hopping" ~0.3, walking away as "staggering"/"move heavy objects" up to 0.3-0.6, a person lying still as
"play magic cube"). Real falls and staggers did score high (falling 0.8-0.95, staggering 0.83), so use a high
min_prob (>= 0.5) and/or require two consecutive windows. Classes 50-60 and 106-120 are two-person interactions:
with one track the second slot is empty and they are unreliable; pass both people as (2,T,17,3) (pyskl's own
NTU sample of a hug: "hugging other person" 0.64 with both tracks, "touch back" 0.11 with one).
The model always picks something; treat the output as a hint for review, never as an alarm on its own.
Licence: pyskl code Apache-2.0; the weights were trained on NTU RGB+D 120 (non-commercial research terms).
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

NTU120_LABELS: list[str] = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
    "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper", "wear jacket",
    "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket",
    "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet",
    "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)",
    "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together",
    "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)",
    "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other", "put on headphone", "take off headphone",
    "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush (quite)",
    "flick hair", "thumb up", "thumb down", "make ok sign", "make victory sign", "staple book", "counting money",
    "cutting nails", "cutting paper (using scissors)", "snapping fingers", "open bottle", "sniff (smell)",
    "squat down", "toss a coin", "fold paper", "ball up paper", "play magic cube", "apply cream on face",
    "apply cream on hand back", "put on bag", "take off bag", "put something into a bag",
    "take something out of a bag", "open a box", "move heavy objects", "shake fist", "throw up cap/hat",
    "hands up (both hands)", "cross arms", "arm circles", "arm swings", "running on the spot",
    "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn", "stretch oneself", "blow nose",
    "hit other person with something", "wield knife towards other person", "knock over other person (hit with body)",
    "grab other person's stuff", "shoot at other person with a gun", "step on foot", "high-five",
    "cheers and drink", "carry something with other person", "take a photo of other person", "follow other person",
    "whisper in other person's ear", "exchange things with other person", "support somebody with hand",
    "finger-guessing game (playing rock-paper-scissors)",
]
assert len(NTU120_LABELS) == 120

# NTU classes worth surfacing for a front-yard camera -> our own tags
EVENTS_OF_INTEREST: dict[str, str] = {
    "falling": "fall",
    "staggering": "stagger",
    "punching/slapping other person": "fight",
    "kicking other person": "fight",
    "pushing other person": "fight",
    "hit other person with something": "fight",
    "knock over other person (hit with body)": "fight",
    "wield knife towards other person": "weapon",
    "shoot at other person with a gun": "weapon",
    "kicking something": "kick",
    "side kick": "kick",
    "throw": "throw",
    "pickup": "pick_up",
    "drop": "drop",
    "jump up": "jump",
    "hopping (one foot jumping)": "jump",
    "squat down": "crouch",
    "taking a selfie": "photo",
    "take a photo of other person": "photo",
    "grab other person's stuff": "grab",
    "touch other person's pocket": "grab",
}

NTU_FPS = 30.0
CLIP_LEN = 100
SCORE_THR = 0.01        # PreNormalize2D threshold


def events(top5: list[tuple[str, float]], min_prob: float = 0.5) -> list[str]:
    """Top-5 from SkeletonActionRecognizer -> our tags (deduplicated, best first) whose probability >= min_prob.
    Probabilities of classes sharing a tag are summed (e.g. punching + pushing -> 'fight')."""
    acc: dict[str, float] = {}
    for label, p in top5:
        tag = EVENTS_OF_INTEREST.get(label)
        if tag:
            acc[tag] = acc.get(tag, 0.0) + p
    return [t for t, p in sorted(acc.items(), key=lambda kv: -kv[1]) if p >= min_prob]


def _fill_gaps(kp: np.ndarray, max_gap: int) -> np.ndarray:
    """kp (T,17,3): linearly interpolate interior runs of empty frames (all conf <= 0.01) up to max_gap long."""
    present = kp[:, :, 2].max(1) > SCORE_THR
    if not present.any():
        return kp
    idx = np.flatnonzero(present)
    kp = kp.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a)[:, None, None] / (b - a)
            kp[a + 1:b] = kp[a] * (1 - w) + kp[b] * w
    return kp


def _resample(kp: np.ndarray, fps: float, target: float = NTU_FPS) -> np.ndarray:
    """Linear resampling in time of (M,T,17,3) from fps to target fps. A joint next to a missing one (conf <= 0.01)
    takes the nearest sample instead, so it is never dragged towards the (0, 0) of a missing joint."""
    T = kp.shape[1]
    if T < 2 or abs(fps - target) < 0.5:
        return kp
    n = max(2, int(round((T - 1) * target / fps)) + 1)
    t = np.linspace(0, T - 1, n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0)[None, :, None, None]
    a, b = kp[:, i0], kp[:, i1]
    out = a * (1 - w) + b * w
    near = np.where(w >= 0.5, b, a)
    edge = ((a[..., 2] <= SCORE_THR) | (b[..., 2] <= SCORE_THR))[..., None]
    return np.where(edge, near, out).astype(np.float32)


def _uniform_sample_test(num_frames: int, clip_len: int, num_clips: int, seed: int = 255) -> np.ndarray:
    """pyskl UniformSampleFrames._get_test_clips with p_interval=1 (same random call sequence), then mod T."""
    rng = np.random.RandomState(seed)
    all_inds = []
    for i in range(num_clips):
        rng.rand()                          # ratio = 1 (p_interval (1, 1)): the draw still happens
        off = rng.randint(1)                # old_num_frames - num_frames + 1 == 1 -> 0
        if num_frames < clip_len:
            start = i if num_frames < num_clips else i * num_frames // num_clips
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < clip_len * 2:
            basic = np.arange(clip_len)
            pick = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[pick] = 1
            inds = basic + np.cumsum(offset)[:-1]
        else:
            bids = np.array([j * num_frames // clip_len for j in range(clip_len + 1)])
            inds = bids[:clip_len] + rng.randint(np.diff(bids))
        all_inds.append(inds + off)
    return np.mod(np.concatenate(all_inds), num_frames)


def preprocess(kps: np.ndarray, img_size: tuple[int, int], fps: float | None = None, clips: int = 10,
               max_gap_s: float = 1.0) -> np.ndarray | None:
    """(T,17,3) or (M<=2,T,17,3) pixel tracks -> model input (clips, 2, 100, 17, 3), or None if nothing usable."""
    kp = np.asarray(kps, np.float32)
    if kp.ndim == 3:
        kp = kp[None]
    kp = kp[:2].copy()
    bad = ~np.isfinite(kp).all(-1)                       # NaN anywhere in a joint -> missing joint
    kp[bad] = 0.0
    # empty frames: shared trim/fill across persons, keyed on the union of presence
    gap = int(round(max_gap_s * (fps or NTU_FPS)))
    present = (kp[..., 2] > SCORE_THR).any(-1).any(0)
    if not present.any():
        return None
    idx = np.flatnonzero(present)
    kp = kp[:, idx[0]:idx[-1] + 1]
    kp = np.stack([_fill_gaps(p, gap) for p in kp])
    if fps:
        kp = _resample(kp, fps)
    # PreNormalize2D (mode 'fix', img_shape = (h, w))
    w, h = img_size
    out = kp.copy()
    out[..., 0] = (kp[..., 0] - w / 2) / (w / 2)
    out[..., 1] = (kp[..., 1] - h / 2) / (h / 2)
    mask = kp[..., 2] <= SCORE_THR
    out[..., 0][mask] = 0
    out[..., 1][mask] = 0
    # UniformSample (test) + PoseDecode
    T = out.shape[1]
    inds = _uniform_sample_test(T, CLIP_LEN, clips)
    out = out[:, inds]
    # FormatGCNInput(num_person=2, mode='zero')
    if out.shape[0] < 2:
        out = np.concatenate([out, np.zeros_like(out)], 0)
    M, TT, V, C = out.shape
    return np.ascontiguousarray(out.reshape(M, clips, TT // clips, V, C).transpose(1, 0, 2, 3, 4), np.float32)


class SkeletonActionRecognizer:
    def __init__(self, model_path: str, threads: int = 4, clips: int = 3):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.clips = clips

    def probs(self, kps: np.ndarray, img_size: tuple[int, int], fps: float | None = None) -> np.ndarray | None:
        """-> (120,) class probabilities (softmax averaged over clips), or None if the track has no keypoints."""
        x = preprocess(kps, img_size, fps, self.clips)
        if x is None:
            return None
        logits = self.sess.run(None, {self.inp: x})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).mean(0)

    def __call__(self, kps: np.ndarray, img_size: tuple[int, int], fps: float | None = None,
                 k: int = 5) -> list[tuple[str, float]]:
        """kps (T,17,3) x, y, confidence in pixels of an image of img_size (W, H); NaN / zero-confidence frames
        count as missing. A (2,T,17,3) array gives two people (needed for the interaction classes).
        -> top-k [(NTU label, probability)], best first; [] if there is nothing to classify."""
        p = self.probs(kps, img_size, fps)
        if p is None:
            return []
        top = np.argsort(-p)[:k]
        return [(NTU120_LABELS[i], float(p[i])) for i in top]

    def windows(self, kps: np.ndarray, img_size: tuple[int, int], fps: float, win_s: float = 4.0,
                hop_s: float = 2.0, k: int = 5) -> list[tuple[float, list[tuple[str, float]]]]:
        """Sliding windows over a long track (NTU clips are ~2-5 s; pyskl would squeeze a 20 s track into 100
        frames). -> [(start time s, top-k)] per window."""
        kp = np.asarray(kps, np.float32)
        T = kp.shape[-3]
        win, hop = max(1, int(round(win_s * fps))), max(1, int(round(hop_s * fps)))
        starts = list(range(0, max(1, T - win + 1), hop))
        if starts[-1] + win < T:
            starts.append(max(0, T - win))
        res = []
        for s in starts:
            r = self(kp[..., s:s + win, :, :], img_size, fps, k)
            if r:
                res.append((s / fps, r))
        return res
