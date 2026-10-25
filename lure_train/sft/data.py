"""Generic SFT manifest: one jsonl row per clip. Works for any corpus (YODAS or anything else) once it is in this shape.

  {"audio": "data/x/0001.flac",          # volume-relative or /vol path, any rate (encoded at 24 kHz)
   "text": "Okay (laugh) you got me.",   # exact transcript, inline round-bracket tags allowed
   "ref_audio": "...", "ref_text": "...",# optional ICL reference (another clip of the same voice + its exact transcript)
   "speaker": "chan_42",                 # optional; used to pick a reference when ref_audio is missing
   "direction": "warm, amused",          # optional instruction text
   "split": "train" | "heldout",         # optional, default train
   "src": "yodas", "id": "..."}           # optional source label (for mixing weights) and id (default sha1 of audio)

A row without ref_audio clones from another row of the same speaker (chosen by hash, its tag-stripped text as ref_text). Rows with
neither are dropped: cloning a clip from itself teaches the model to copy the reference.
"""
import collections
import hashlib
import os

from ..common import jl, vpath
from ..model import TAG


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()


def plain_of(text):
    return " ".join(TAG.sub(" ", text or "").split())


def load_manifest(path):
    rows = jl(vpath(path))
    for r in rows:
        r.setdefault("id", sha(r["audio"])[:16])
        r.setdefault("split", "train")
        r.setdefault("src", "data")
    by_spk = collections.defaultdict(list)
    for r in rows:
        if r.get("speaker"):
            by_spk[r["speaker"]].append(r)
    out, dropped = [], 0
    for r in rows:
        if not r.get("ref_audio"):
            others = [o for o in by_spk.get(r.get("speaker"), []) if o["id"] != r["id"]]
            if not others:
                dropped += 1
                continue
            o = sorted(others, key=lambda o: sha(r["id"] + o["id"]))[0]
            r["ref_audio"], r["ref_text"] = o["audio"], plain_of(o["text"])
        r["voice"] = {"voice_id": "v_" + sha(r["ref_audio"])[:16], "ref_wav": r["ref_audio"], "ref_text": r["ref_text"]}
        out.append(r)
    print(f"manifest {path}: {len(out)} rows, {dropped} dropped (no reference)", flush=True)
    return out


def load_codes(codes_dir):
    import torch
    d, out = vpath(codes_dir), {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".pt"):
                out.update(torch.load(f"{d}/{f}"))
    return out


def load_data(manifest, codes_dir):
    """-> ({split: rows}, codes) for rows that have codes."""
    codes = load_codes(codes_dir)
    sp = collections.defaultdict(list)
    for r in load_manifest(manifest):
        if r["id"] in codes:
            sp[r["split"]].append(r)
    return sp, codes


def req_of(r):
    return {"text": r["text"], "direction": r.get("direction"), "voice": r["voice"]}


class Sampler:
    """Per-epoch permutation over a pool, resumable."""

    def __init__(self, ids, seed):
        self.ids, self.seed, self.epoch, self.pos = list(ids), seed, 0, 0
        self._new()

    def _new(self):
        import numpy as np
        self.perm = np.random.default_rng(self.seed * 7919 + self.epoch).permutation(len(self.ids)).tolist()

    def take(self, n):
        out = []
        while len(out) < n:
            if self.pos >= len(self.perm):
                self.epoch, self.pos = self.epoch + 1, 0
                self._new()
            out.append(self.ids[self.perm[self.pos]])
            self.pos += 1
        return out

    def state(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def load(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._new()


class Mixer:
    """Batches drawn from several sources with fixed weights ({"yodas": 0.75, "other": 0.25}); one source by default."""

    def __init__(self, rows, weights=None, seed=0):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["src"]].append(r["id"])
        w = weights or {k: len(v) for k, v in by.items()}
        self.w = {k: float(v) for k, v in w.items() if by.get(k)}
        self.s = {k: Sampler(by[k], seed + i * 17) for i, k in enumerate(sorted(self.w))}

    def take(self, n):
        tot = sum(self.w.values())
        ks = sorted(self.w)
        cnt = {k: int(n * self.w[k] / tot) for k in ks}
        for k in ks[: n - sum(cnt.values())]:
            cnt[k] += 1
        return [i for k in ks for i in self.s[k].take(cnt[k])]

    def state(self):
        return {k: s.state() for k, s in self.s.items()}

    def load(self, st):
        for k, v in st.items():
            if k in self.s:
                self.s[k].load(v)
