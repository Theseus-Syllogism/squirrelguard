"""Generate the camera-speaker deterrent sounds (audible, 1.5-8 kHz: what a small camera speaker reproduces).

Writes WAV files; encode for the camera with:  ffmpeg -i x.wav -c:a libopus -b:a 64k -ar 48000 -ac 1 x.opus
Usage: .venv/bin/python scripts/make_deter_sounds.py OUT_DIR
"""
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 48000
rng = np.random.default_rng(7)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "sounds/deter")
out.mkdir(parents=True, exist_ok=True)


def env(n, attack=0.005, release=0.03):
    t = np.arange(n) / RATE
    dur = n / RATE
    return np.clip(np.minimum(t / attack, (dur - t) / release), 0, 1)


def chirp(f0, f1, dur):
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))) * env(n)


def silence(dur):
    return np.zeros(int(dur * RATE))


def bandnoise(dur, lo, hi):
    n = int(dur * RATE)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9) * env(n, 0.002, 0.02)


def sweeps(seed, dur=3.0):
    r = np.random.default_rng(seed)
    parts, total = [], 0
    while total < dur * RATE:
        d = r.uniform(0.06, 0.3)
        parts += [chirp(*r.uniform(1500, 8000, 2), d), silence(r.uniform(0.02, 0.12))]
        total = sum(len(p) for p in parts)
    return np.concatenate(parts)[: int(dur * RATE)]


def screech(dur=1.6):
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    f = 4200 - 2200 * (t / dur) ** 0.7 + 180 * np.sin(2 * np.pi * 9 * t)  # falling whistle with vibrato
    phase = 2 * np.pi * np.cumsum(f) / RATE
    x = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * bandnoise(dur, 2000, 6000)
    return x / np.abs(x).max() * env(n, 0.01, 0.25)


def clatter(dur=2.0):
    parts, total = [], 0
    while total < dur * RATE:
        parts += [bandnoise(rng.uniform(0.02, 0.08), 1500, 7000), silence(rng.uniform(0.03, 0.15))]
        total = sum(len(p) for p in parts)
    return np.concatenate(parts)[: int(dur * RATE)]


sounds = {
    "sweeps_a": sweeps(1),
    "sweeps_b": sweeps(2),
    "screech": np.concatenate([screech(), silence(0.25), screech(1.3)]),
    "clatter": clatter(),
}
for name, x in sounds.items():
    pcm = (np.clip(x, -1, 1) * 0.9 * 32767).astype("<i2")
    with wave.open(str(out / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    print(f"{name}.wav  {len(x) / RATE:.1f}s")
