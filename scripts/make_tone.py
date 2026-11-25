"""Generate a deterrent WAV: a randomised series of frequency sweeps (varying the sound slows habituation).

Usage: .venv/bin/python scripts/make_tone.py OUT.wav --lo 4000 --hi 12000 [--rate 48000] [--seconds 3]

Frequencies above ~20 kHz need --rate 96000 or more AND a speaker that can reproduce them (a piezo
tweeter); laptop and camera speakers roll off well before that, so ultrasonic output from them is
mostly silence. Test with a microphone/spectrum app before trusting it. Ultrasound also reaches dogs,
cats and bats.
"""
import argparse
import wave

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lo", type=float, default=4000)
ap.add_argument("--hi", type=float, default=12000)
ap.add_argument("--rate", type=int, default=48000)
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--volume", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()
if a.hi >= a.rate / 2:
    raise SystemExit(f"--hi {a.hi} Hz is at/above Nyquist for --rate {a.rate}; raise --rate")

rng = np.random.default_rng(a.seed)
chunks, total = [], int(a.seconds * a.rate)
while sum(len(c) for c in chunks) < total:
    dur = rng.uniform(0.08, 0.4)
    n = int(dur * a.rate)
    f0, f1 = rng.uniform(a.lo, a.hi, 2)
    t = np.arange(n) / a.rate
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))  # linear chirp
    env = np.minimum(1, np.minimum(t, dur - t) / 0.01)               # 10 ms fade in/out, no clicks
    chunks.append(np.sin(phase) * env)
    chunks.append(np.zeros(int(rng.uniform(0.02, 0.15) * a.rate)))  # gaps
sig = np.concatenate(chunks)[:total] * a.volume
pcm = (sig * 32767).astype("<i2")
with wave.open(a.out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(a.rate)
    w.writeframes(pcm.tobytes())
print(f"wrote {a.out}: {a.seconds}s, {a.lo:.0f}-{a.hi:.0f} Hz, {a.rate} Hz sample rate")
