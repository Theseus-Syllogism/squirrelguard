"""RFC 3161 trusted timestamps, standard library only.

  token = stamp(sha256_hex, url)    -> Stamp(tsr=DER bytes, gen_time, offset_s, ...)
  info  = parse_response(tsr_bytes) -> status, gen_time, the hash it covers, nonce, serial

Only the hash leaves the machine, never the file. The response is the TSA's signed statement that this hash existed
at gen_time. Checking the TSA's signature needs its certificate chain: `openssl ts -verify` does that (see
verify_openssl); parse_response only reads the content and checks it covers the right hash and nonce.

A copy of this file lives in deploy/recorder/recorder/ (the segment sealer); tests/test_evidence_export.py checks that
the two are identical.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SHA256_OID = bytes.fromhex("0609608648016503040201")          # 2.16.840.1.101.3.4.2.1
TSTINFO_OID = bytes.fromhex("060b2a864886f70d0109100104")     # 1.2.840.113549.1.9.16.1.4
STATUS = {0: "granted", 1: "granted with modifications", 2: "rejection", 3: "waiting", 4: "revocation warning",
          5: "revocation notification"}


class TSAError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ DER
def _len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _tlv(tag: int, body: bytes) -> bytes:
    return bytes([tag]) + _len(len(body)) + body


def _int(v: int) -> bytes:
    b = v.to_bytes(max(1, (v.bit_length() + 8) // 8), "big", signed=True)
    return _tlv(0x02, b)


def request(digest: bytes, nonce: int) -> bytes:
    """TimeStampReq v1 for a SHA-256 digest, with a nonce, asking for the TSA certificate."""
    if len(digest) != 32:
        raise ValueError("SHA-256 digest must be 32 bytes")
    imprint = _tlv(0x30, _tlv(0x30, SHA256_OID + b"\x05\x00") + _tlv(0x04, digest))
    return _tlv(0x30, _int(1) + imprint + _int(nonce) + b"\x01\x01\xff")


def _read(buf: bytes, i: int) -> tuple[int, int, int]:
    """-> (tag, start of value, end of value)."""
    if i + 2 > len(buf):
        raise TSAError("truncated")
    tag, n = buf[i], buf[i + 1]
    i += 2
    if n & 0x80:
        k = n & 0x7F
        if k == 0 or k > 4 or i + k > len(buf):
            raise TSAError("bad length")
        n = int.from_bytes(buf[i:i + k], "big")
        i += k
    if i + n > len(buf):
        raise TSAError("truncated")
    return tag, i, i + n


def _children(buf: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    out, i = [], start
    while i < end:
        t, s, e = _read(buf, i)
        out.append((t, s, e))
        i = e
    return out


def _octets(buf: bytes, t: int, s: int, e: int) -> bytes:
    if t == 0x04:
        return buf[s:e]
    if t == 0x24:                                               # constructed OCTET STRING (BER): concatenate
        return b"".join(_octets(buf, *c) for c in _children(buf, s, e))
    raise TSAError("expected OCTET STRING")


def parse_response(tsr: bytes) -> dict:
    """Read a TimeStampResp. Raises TSAError when it is not one, or carries no token."""
    t, s, e = _read(tsr, 0)
    if t != 0x30:
        raise TSAError("not a timestamp response")
    parts = _children(tsr, s, e)
    st = _children(tsr, parts[0][1], parts[0][2])
    status = int.from_bytes(tsr[st[0][1]:st[0][2]], "big")
    if status not in (0, 1) or len(parts) < 2:
        raise TSAError(f"the timestamp service refused: {STATUS.get(status, status)}")
    ci = _children(tsr, parts[1][1], parts[1][2])                 # ContentInfo: OID, [0] SignedData
    sd = _children(tsr, *ci[1][1:])[0]
    sdc = _children(tsr, sd[1], sd[2])                           # version, digestAlgorithms, encapContentInfo, ...
    enc = _children(tsr, sdc[2][1], sdc[2][2])
    if enc[0][0] != 0x06 or tsr[enc[0][1]:enc[0][2]] != TSTINFO_OID[2:]:
        raise TSAError("the token does not hold timestamp information")
    inner = _children(tsr, enc[1][1], enc[1][2])[0]
    tst = _octets(tsr, *inner)
    _, s2, e2 = _read(tst, 0)
    f = _children(tst, s2, e2)                                   # version, policy, imprint, serial, genTime, ...
    imp = _children(tst, f[2][1], f[2][2])
    hashed = tst[imp[1][1]:imp[1][2]]
    alg = tst[imp[0][1]:imp[0][2]]
    if f[4][0] != 0x18:
        raise TSAError("no generation time in the token")
    gt = tst[f[4][1]:f[4][2]].decode()
    gen = _gentime(gt)
    nonce = None
    for t3, s3, e3 in f[5:]:
        if t3 == 0x02:
            nonce = int.from_bytes(tst[s3:e3], "big", signed=True)
            break
    return {"status": STATUS.get(status, str(status)), "gen_time": gen, "gen_time_text": gt,
            "hash": hashed.hex(), "sha256": alg.startswith(SHA256_OID),
            "serial": int.from_bytes(tst[f[3][1]:f[3][2]], "big", signed=True), "nonce": nonce}


def _gentime(text: str) -> dt.datetime:
    """GeneralizedTime 'YYYYMMDDHHMMSS[.fff]Z' -> aware UTC datetime."""
    if not text.endswith("Z"):
        raise TSAError("generation time is not in UTC")
    body = text[:-1]
    main, _, frac = body.partition(".")
    t = dt.datetime.strptime(main, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    return t + dt.timedelta(microseconds=int((frac + "000000")[:6])) if frac else t


# ------------------------------------------------------------------------------------------------ client
@dataclass
class Stamp:
    tsr: bytes
    url: str
    gen_time: dt.datetime
    hash: str
    serial: int
    offset_s: float          # TSA time minus this machine's time, at the middle of the round trip
    round_trip_s: float


def stamp(sha256_hex: str, url: str, timeout: float = 20) -> Stamp:
    """Ask the TSA at url to timestamp a SHA-256 hash. Raises TSAError with a readable message on failure."""
    digest = bytes.fromhex(sha256_hex)
    nonce = secrets.randbits(63)
    req = urllib.request.Request(url, data=request(digest, nonce), headers={
        "Content-Type": "application/timestamp-query", "Accept": "application/timestamp-reply",
        "User-Agent": "squirrel-guard-evidence"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(1 << 20)
    except Exception as e:                                      # network, HTTP error, timeout
        raise TSAError(f"the timestamp service did not answer ({type(e).__name__})") from None
    t1 = time.time()
    info = parse_response(body)
    if info["hash"] != sha256_hex or not info["sha256"]:
        raise TSAError("the timestamp covers a different hash")
    if info["nonce"] != nonce:
        raise TSAError("the timestamp answer does not match the request (nonce)")
    mid = dt.datetime.fromtimestamp((t0 + t1) / 2, dt.timezone.utc)
    return Stamp(body, url, info["gen_time"], info["hash"], info["serial"],
                 round((info["gen_time"] - mid).total_seconds(), 3), round(t1 - t0, 3))


def verify_openssl(data: bytes, tsr: bytes, ca_file: str | None = None) -> tuple[bool | None, str]:
    """Full check of the TSA's signature with `openssl ts -verify`. (None, reason) when openssl is not available."""
    exe = shutil.which("openssl")
    if not exe:
        return None, "openssl is not installed"
    ca = ca_file or next((p for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                                      "/etc/ssl/cert.pem") if os.path.exists(p)), None)
    if not ca:
        return None, "no CA certificate bundle found"
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "data").write_bytes(data)
        (Path(d) / "tsr").write_bytes(tsr)
        p = subprocess.run([exe, "ts", "-verify", "-data", "data", "-in", "tsr", "-CAfile", ca], cwd=d,
                           capture_output=True, text=True, timeout=30)
    out = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0 and any("Verification: OK" in l for l in out), (out[-1] if out else "")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
