"""Tag / laugh detector (TagNet) and its reward policy. Inference only.

Model: frozen WavLM-large (bf16, every layer) -> learned layer mix -> 25 fps -> Linear(256) + 2-layer BiGRU, with
  family   sigmoid per FAMILIES entry (multi-label)
  subs     one softmax per family over SUBS[family] (last class = other)
  overlap  laughing-while-talking and voice-breaking heads (fire on speech frames)
  speech   speech activity
Optional specialists (spec_dir: pca.pt + <group>.pt) reuse the same encoder pass.

Policy (TagReward, thresholds from a calibrated policy JSON): each requested tag is looked for in a window around the gap between
the forced-aligned words it sits between (Qwen3-ForcedAligner-0.6B on the tag-stripped script; no alignment -> whole clip).
Look-alikes earn partial credit, unrequested confident events cost lambda_unreq each (capped), and a tag read aloud (its words in
the ASR beyond the script's own) is a hard gate. Spoken interjections (hmm, ugh, ow, pfft, tsk) are checked in the ASR; pause by
an energy VAD.
"""
import collections
import json
import re

import numpy as np

FPS, SR = 25, 16000
TAGS = ["laugh", "chuckle", "giggle", "nervous laugh", "laughs while talking", "wheezing laugh", "snort", "sigh", "relieved sigh",
        "frustrated sigh", "inhale", "sharp inhale", "exhale", "gasp", "happy gasp", "sniff", "sniffle", "sob", "whimper",
        "voice breaking", "cough", "clears throat", "swallow", "gulp", "lip smack", "tsk", "groan", "grunt", "scoff", "yawn", "hum",
        "hmm", "ugh", "pfft", "ow", "pause"]
FAMILIES = ["LAUGH", "BREATH", "CRY", "THROAT", "MOUTH", "VOICED", "OTHER"]
SUBS = {"LAUGH": ["laugh", "chuckle", "giggle", "other"], "BREATH": ["sigh", "exhale", "inhale", "gasp", "yawn", "other"],
        "CRY": ["sob", "whimper", "sniff", "other"], "THROAT": ["cough", "clears throat", "other"],
        "MOUTH": ["swallow", "lip smack", "tsk", "snort", "scoff", "other"], "VOICED": ["hum", "groan", "grunt", "other"]}
OVERLAP_HEADS = ["speech_laugh", "voice_break"]
# tag -> (family, canonical sub or None, mode): acoustic | overlap:<head> | spoken (ASR token) | code (silence)
TAG_SPEC = {
    "laugh": ("LAUGH", "laugh", "acoustic"), "chuckle": ("LAUGH", "chuckle", "acoustic"), "giggle": ("LAUGH", "giggle", "acoustic"),
    "nervous laugh": ("LAUGH", None, "acoustic"), "wheezing laugh": ("LAUGH", None, "acoustic"),
    "laughs while talking": ("LAUGH", None, "overlap:speech_laugh"),
    "sigh": ("BREATH", "sigh", "acoustic"), "frustrated sigh": ("BREATH", "sigh", "acoustic"), "relieved sigh": ("BREATH", "sigh", "acoustic"),
    "exhale": ("BREATH", "exhale", "acoustic"), "inhale": ("BREATH", "inhale", "acoustic"), "sharp inhale": ("BREATH", "inhale", "acoustic"),
    "gasp": ("BREATH", "gasp", "acoustic"), "happy gasp": ("BREATH", "gasp", "acoustic"), "yawn": ("BREATH", "yawn", "acoustic"),
    "sob": ("CRY", "sob", "acoustic"), "whimper": ("CRY", "whimper", "acoustic"), "sniff": ("CRY", "sniff", "acoustic"),
    "sniffle": ("CRY", "sniff", "acoustic"), "voice breaking": ("CRY", None, "overlap:voice_break"),
    "cough": ("THROAT", "cough", "acoustic"), "clears throat": ("THROAT", "clears throat", "acoustic"),
    "swallow": ("MOUTH", "swallow", "acoustic"), "gulp": ("MOUTH", "swallow", "acoustic"), "lip smack": ("MOUTH", "lip smack", "acoustic"),
    "tsk": ("MOUTH", "tsk", "acoustic"), "snort": ("MOUTH", "snort", "acoustic"), "scoff": ("MOUTH", "scoff", "acoustic"),
    "pfft": ("MOUTH", "scoff", "acoustic"), "hum": ("VOICED", "hum", "acoustic"), "groan": ("VOICED", "groan", "acoustic"),
    "grunt": ("VOICED", "grunt", "acoustic"), "hmm": ("VOICED", "hum", "spoken"), "ugh": ("VOICED", "groan", "spoken"),
    "ow": ("VOICED", "grunt", "spoken"), "pause": (None, None, "code"),
}
SECONDARY = {"snort": {"LAUGH": 0.7}, "scoff": {"BREATH": 0.5}, "pfft": {"BREATH": 0.5}, "sniff": {"BREATH": 0.4}, "sniffle": {"BREATH": 0.4},
             "sob": {"BREATH": 0.3}, "groan": {"BREATH": 0.3}, "ugh": {"BREATH": 0.3}, "yawn": {"VOICED": 0.5}}
SAME_FAMILY, OTHER_EXIT = 0.3, 0.4
_PAIRS = {("laugh", "chuckle"): 0.8, ("laugh", "giggle"): 0.7, ("chuckle", "giggle"): 0.8, ("sigh", "exhale"): 0.8, ("inhale", "gasp"): 0.8,
          ("yawn", "sigh"): 0.4, ("sob", "whimper"): 0.7, ("sob", "sniff"): 0.3, ("cough", "clears throat"): 0.6, ("swallow", "lip smack"): 0.4,
          ("lip smack", "tsk"): 0.7, ("snort", "scoff"): 0.6, ("hum", "groan"): 0.4, ("groan", "grunt"): 0.7}
CREDIT = {**_PAIRS, **{(b, a): v for (a, b), v in _PAIRS.items()}}
MAX_DUR_S = {"chuckle": 1.5, "giggle": 2.0, "laugh": 4.0, "sigh": 2.0, "exhale": 1.5, "inhale": 1.2, "gasp": 1.0, "yawn": 4.0, "sob": 4.0,
             "whimper": 2.5, "sniff": 1.0, "cough": 2.5, "clears throat": 1.5, "swallow": 1.0, "lip smack": 0.5, "tsk": 0.5, "snort": 1.0,
             "scoff": 1.0, "hum": 3.0, "groan": 2.5, "grunt": 1.0}
TAG_RE = re.compile(r"[\(\[]\s*([a-z][a-z ]*?)\s*[\)\]]")
WORD_RE = re.compile(r"[a-z0-9']+")
READ_ALOUD = {
    "laugh": ["laugh", "laughs", "laughing", "laughed", "laughter"], "chuckle": ["chuckle", "chuckles", "chuckling"],
    "giggle": ["giggle", "giggles", "giggling"], "nervous laugh": ["nervous", "nervously"], "wheezing laugh": ["wheezing", "wheeze", "wheezes"],
    "laughs while talking": ["laughs", "laughing", "talking"], "snort": ["snort", "snorts", "snorting"],
    "sigh": ["sigh", "sighs", "sighing"], "relieved sigh": ["relieved", "sigh"], "frustrated sigh": ["frustrated", "sigh"],
    "inhale": ["inhale", "inhales"], "sharp inhale": ["sharp", "inhale"], "exhale": ["exhale", "exhales"], "gasp": ["gasp", "gasps"],
    "happy gasp": ["happy", "gasp"], "sniff": ["sniff", "sniffs"], "sniffle": ["sniffle", "sniffles"], "sob": ["sob", "sobs", "sobbing"],
    "whimper": ["whimper", "whimpers"], "voice breaking": ["voice", "breaking"], "cough": ["cough", "coughs", "coughing"],
    "clears throat": ["clears", "clear", "throat"], "swallow": ["swallow", "swallows"], "gulp": ["gulp", "gulps"],
    "lip smack": ["lip", "smack", "smacks"], "tsk": [], "groan": ["groan", "groans"], "grunt": ["grunt", "grunts"],
    "scoff": ["scoff", "scoffs"], "yawn": ["yawn", "yawns", "yawning"], "hum": ["hum", "hums", "humming"], "hmm": [], "ugh": [], "pfft": [],
    "ow": [], "pause": ["pause"]}
SPOKEN = {"hmm": re.compile(r"^(h+m+|m+h?m+|hm+)$"), "ugh": re.compile(r"^(u+g+h+|ugh+|urgh+)$"), "ow": re.compile(r"^(o+w+|ouch|owie)$"),
          "pfft": re.compile(r"^(p+f+t*|pff+|psh+)$"), "tsk": re.compile(r"^(tsk+|tut)$")}
# tokens that are a correct rendering of an event, never a read-aloud (a real laugh transcribed as "haha")
EVENT_TOKENS = re.compile(r"^(?:(?:ha)+h?|(?:he)+h?|ah+|h+m+|m+h?m*|u+h+|u+m+|uhm+|e+r+m*|p+f+t*|u+g+h+|o+w+|tsk|psh+|hee+|heh+|oh+)$")


def credit(want, got, fam):
    if want is None or got == want:
        return 1.0
    if got == "other":
        return OTHER_EXIT
    return CREDIT.get((want, got), SAME_FAMILY if got in SUBS.get(fam, []) else 0.0)


def norm_tokens(s):
    return WORD_RE.findall((s or "").lower())


def parse_tagged(text):
    """'Okay (laugh) you got me.' -> ('Okay you got me.', [('laugh', 1)]). Unknown tags come back as None."""
    tags, plain, pos, last = [], [], 0, 0
    for m in TAG_RE.finditer(text):
        before = text[last:m.start()]
        plain.append(before)
        pos += len(norm_tokens(before))
        t = m.group(1).strip()
        tags.append((t if t in TAG_SPEC else None, pos))
        last = m.end()
    plain.append(text[last:])
    p = re.sub(r"\s+", " ", "".join(plain)).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", p), tags


# ------------------------------------------------------------------------------------------------ model
def loudness_normalize(x, target=-23.0):
    x = np.asarray(x, dtype=np.float32)
    try:
        import pyloudnorm as pyln
        lufs = pyln.Meter(SR).integrated_loudness(x.astype(np.float64)) if len(x) >= int(0.4 * SR) else -np.inf
    except Exception:  # noqa: BLE001
        lufs = -np.inf
    if not np.isfinite(lufs):
        lufs = 20 * np.log10(float(np.sqrt(np.mean(x ** 2)) + 1e-9)) - 0.7
    y = x * (10 ** ((target - lufs) / 20))
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


def batch(xs, device="cuda"):
    """16 kHz arrays -> padded wav (loudness-normalised, then zero-mean unit-variance), mask, frame counts at 25 fps."""
    import torch
    ys = []
    for x in xs:
        y = loudness_normalize(x)
        ys.append(((y - y.mean()) / (y.std() + 1e-7)).astype(np.float32))
    L = max(len(y) for y in ys)
    wav, mask = torch.zeros(len(ys), L), torch.zeros(len(ys), L, dtype=torch.long)
    for i, y in enumerate(ys):
        wav[i, :len(y)] = torch.from_numpy(y)
        mask[i, :len(y)] = 1
    return wav.to(device), mask.to(device), [len(y) // (SR // FPS) for y in ys]


def build(encoder="microsoft/wavlm-large", hidden=256):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class TagNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(encoder).eval()
            for p in self.enc.parameters():
                p.requires_grad_(False)
            d = self.enc.config.hidden_size
            self.layer_w = nn.Parameter(torch.zeros(self.enc.config.num_hidden_layers + 1))
            self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1))
            self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.family = nn.Linear(2 * hidden, len(FAMILIES))
            self.subs = nn.ModuleDict({f: nn.Linear(2 * hidden, len(SUBS[f])) for f in SUBS})
            self.overlap = nn.Linear(2 * hidden, len(OVERLAP_HEADS))
            self.speech = nn.Linear(2 * hidden, 1)

        def encode(self, wav, mask=None):
            # encoder in its own bf16 dtype, no autocast (autocast around bf16 WavLM measured 5x slower); mask dropped when unneeded
            if mask is not None and bool(mask.all()):
                mask = None
            with torch.no_grad():
                return self.enc(wav.to(next(self.enc.parameters()).dtype), attention_mask=mask, output_hidden_states=True).hidden_states

        def mix(self, hs):
            w, h = torch.softmax(self.layer_w, 0), None
            for i, x in enumerate(hs):
                h = w[i] * x.float() if h is None else h + w[i] * x.float()
            T = h.shape[1] // 2 * 2
            return h[:, :T].reshape(h.shape[0], T // 2, 2, -1).mean(2)

        def heads(self, hs, h=None):
            h, _ = self.gru(self.proj(self.mix(hs) if h is None else h))
            return {"family": self.family(h), "subs": {f: m(h) for f, m in self.subs.items()}, "overlap": self.overlap(h),
                    "speech": self.speech(h).squeeze(-1)}

    return TagNet()


def build_spec(C, d_in=384, h=192):
    import torch
    import torch.nn as nn

    class Spec(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.15))
            self.conv = nn.Conv1d(h, h, 5, padding=2)
            self.gru = nn.GRU(h, h, num_layers=1, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * h, C)
            self.att = nn.Linear(2 * h, 1)
            self.clip = nn.Linear(2 * h, 1)

        def forward(self, x, mask):
            z = self.inp(x)
            z = z + torch.relu(self.conv(z.transpose(1, 2))).transpose(1, 2)
            g, _ = self.gru(z)
            a = self.att(g).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
            return self.out(g), self.clip((a.unsqueeze(-1) * g).sum(1)).squeeze(-1)

    return Spec()


def to_numpy(out, i, n_frames=None):
    import torch
    sl = slice(0, n_frames)
    return {"family": torch.sigmoid(out["family"][i, sl]).float().cpu().numpy(),
            "subs": {f: torch.softmax(v[i, sl], -1).float().cpu().numpy() for f, v in out["subs"].items()},
            "overlap": torch.sigmoid(out["overlap"][i, sl]).float().cpu().numpy(),
            "speech": torch.sigmoid(out["speech"][i, sl]).float().cpu().numpy(), "fps": FPS}


# ------------------------------------------------------------------------------------------------ policy
def _smooth(p, k=3):
    if len(p) < k:
        return p
    pad = np.pad(p, [(k // 2, k // 2)] + [(0, 0)] * (p.ndim - 1), mode="edge")
    return np.median(np.stack([pad[i:i + len(p)] for i in range(k)], 0), 0)


def _ramp(p, lo, hi):
    return float(np.clip((p - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def _runs(mask):
    out, a = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i))
            a = None
    return out


def _silence_mask(x16, fps):
    """Energy VAD at the head frame rate: silent below max(peak - 35 dB, p10 + 8 dB)."""
    n = SR // fps
    m = len(x16) // n
    if m < 3:
        return np.zeros(max(m, 0), bool)
    db = 20 * np.log10(np.sqrt(np.mean(np.asarray(x16[:m * n], dtype=np.float64).reshape(m, n) ** 2, 1)) + 1e-9)
    return db <= max(db.max() - 35, np.percentile(db, 10) + 8)


def tag_window(words, k, dur, tol, span_words=0):
    left = words[k - 1]["e"] if 0 < k <= len(words) else 0.0
    right = words[k]["s"] if k < len(words) else dur
    if span_words and k < len(words):
        right = words[min(len(words), k + span_words) - 1]["e"]
    return max(0.0, left - tol), min(dur, max(right, left) + tol), left, right


def _extent(on, peak):
    lo, hi = peak, peak + 1
    while lo > 0 and on[lo - 1]:
        lo -= 1
    while hi < len(on) and on[hi]:
        hi += 1
    return lo, hi


class TagReward:
    """Policy JSON keys: tol_s, family_base, overlong_factor, pause_min_s, pause_tol_s, lambda_unreq, unreq_cap, ignore_soft_breaths,
    soft_breath_max_s, family{f: lo, hi, unreq, min_dur_s}, overlap{h: lo, hi}, tag_status{tag: reliable|family_only|unsupported},
    optional tag_gate, family_ok, spec_thr. Refuses an uncalibrated policy."""

    def __init__(self, policy):
        if not policy.get("calibrated"):
            raise ValueError("tag policy is not calibrated")
        self.P = policy
        self.fi = {f: i for i, f in enumerate(FAMILIES)}

    def _spec(self, tag, spec, a, b, fps):
        want = TAG_SPEC[tag][1]
        th = self.P.get("spec_thr", {}).get(want)
        if th is None or spec is None or want not in spec:
            return None
        p = np.asarray(spec[want])
        seg = p[a:b]
        if not len(seg):
            return 0.0, {"spec": want, "family_hit": 0.0}
        lo_i, hi_i = _extent(p >= th["lo"], a + int(np.argmax(seg)))
        s = _ramp(float(seg.max()), th["lo"], th["hi"])
        info = {"spec": want, "family_hit": round(s, 3)}
        if s > 0 and want in MAX_DUR_S and (hi_i - lo_i) / fps > MAX_DUR_S[want]:
            s *= self.P["overlong_factor"]
            info["overlong"] = True
        return s, info

    def _acoustic(self, tag, fam_p, subs, a, b, fps, spec=None):
        P = self.P
        r = self._spec(tag, spec, a, b, fps)
        if r is not None:
            return r
        fam, want, _ = TAG_SPEC[tag]
        status, gate = P["tag_status"].get(tag, "unsupported"), P.get("tag_gate", {}).get(tag)
        best = (0.0, fam, 0)
        for f, wgt in ([(fam, 1.0)] if gate else [(fam, 1.0)] + list(SECONDARY.get(tag, {}).items())):
            seg = fam_p[a:b, self.fi[f]]
            if len(seg):
                pk = int(np.argmax(seg))
                fp = gate or P["family"][f]
                s = _ramp(float(seg[pk]), fp["lo"], fp["hi"]) * wgt
                if s > best[0]:
                    best = (s, f, a + pk)
        s, f, peak = best
        info = {"family_hit": round(s, 3), "family_seen": f}
        if s <= 0:
            return 0.0, info
        lo_i, hi_i = _extent(fam_p[:, self.fi[f]] >= (gate or P["family"][f])["lo"], peak)
        score = s
        if status == "reliable" and want is not None and f in SUBS:
            w = fam_p[lo_i:hi_i, self.fi[f]][:, None]
            dist = (subs[f][lo_i:hi_i] * w).sum(0) / max(float(w.sum()), 1e-6)
            exp_credit = float(sum(p * credit(want if f == fam else None, sn, f) for p, sn in zip(dist, SUBS[f])))
            base = 0.0 if gate else P["family_base"]
            score = s * (base + (1 - base) * exp_credit)
        if want in MAX_DUR_S and (hi_i - lo_i) / fps > MAX_DUR_S[want]:
            score *= P["overlong_factor"]
            info["overlong"] = True
        return score, info

    def score(self, probs, words, tags, plain, asr, x16=None, asr_words=None):
        P, fps = self.P, probs["fps"]
        fam_p, ovl = _smooth(np.asarray(probs["family"])), _smooth(np.asarray(probs["overlap"]))
        subs = {f: np.asarray(v) for f, v in probs["subs"].items()}
        T = len(fam_p)
        dur = T / fps
        fr = lambda t: int(np.clip(round(t * fps), 0, T))  # noqa: E731
        surplus = collections.Counter(norm_tokens(asr)) - collections.Counter(norm_tokens(plain))
        read = []
        for tag, _ in tags:
            hit = [w for w in READ_ALOUD.get(tag, []) if surplus.get(w, 0) > 0 and not EVENT_TOKENS.match(w)] if tag else []
            for w in hit:
                surplus[w] -= 1
            if hit:
                read.append({"tag": tag, "words": hit})
        read_tags = {r["tag"] for r in read}
        per_tag, windows = [], []
        for tag, k in tags:
            if tag is None:
                continue
            fam, _, mode = TAG_SPEC[tag]
            wa, wb, gl, gr = tag_window(words, k, dur, P["tol_s"], 3 if mode.startswith("overlap") else 0)
            windows.append((wa, wb))
            a, b = fr(wa), max(fr(wb), fr(wa) + 1)
            status = P["tag_status"].get(tag, "unsupported")
            if status == "unsupported" and mode != "code":
                continue
            if tag in read_tags:
                sc = 0.0
            elif mode == "code":
                if x16 is None:
                    continue
                sil = _silence_mask(x16, fps)
                pa, pb = fr(max(0.0, gl - P["pause_tol_s"])), fr(min(dur, gr + P["pause_tol_s"]))
                sc = min(1.0, max([(r[1] - r[0]) / fps for r in _runs(sil[pa:pb])] or [0.0]) / P["pause_min_s"])
            elif mode.startswith("overlap"):
                op = P["overlap"][mode.split(":")[1]]
                sc = _ramp(float(ovl[a:b, OVERLAP_HEADS.index(mode.split(":")[1])].max()), op["lo"], op["hi"])
                if status == "family_only":
                    sc = max(sc, _ramp(float(fam_p[a:b, self.fi[fam]].max()), P["family"][fam]["lo"], P["family"][fam]["hi"]))
            elif mode == "spoken":
                located = bool(asr_words) and any(SPOKEN[tag].match(w) for x in asr_words if wa <= x["s"] <= wb for w in norm_tokens(x["w"]))
                sc = 1.0 if located else 0.0
                if not located:
                    n_asr = sum(1 for w in norm_tokens(asr) if SPOKEN[tag].match(w))
                    n_plain = sum(1 for w in norm_tokens(plain) if SPOKEN[tag].match(w))
                    fp = P["family"]["VOICED"]
                    sc = 0.7 if n_asr > n_plain else 0.5 * _ramp(float(fam_p[a:b, self.fi["VOICED"]].max()), fp["lo"], fp["hi"])
            else:
                sc, _ = self._acoustic(tag, fam_p, subs, a, b, fps, probs.get("spec"))
                if tag in SPOKEN and asr_words and any(SPOKEN[tag].match(w) for x in asr_words if wa <= x["s"] <= wb for w in norm_tokens(x["w"])):
                    sc = 1.0
            per_tag.append({"tag": tag, "score": round(float(sc), 3)})
        unreq = []
        for f in FAMILIES:
            if not P.get("family_ok", {}).get(f, True):
                continue
            fp = P["family"][f]
            for r0, r1 in _runs(fam_p[:, self.fi[f]] >= fp["unreq"]):
                d, t0, t1 = (r1 - r0) / fps, r0 / fps, r1 / fps
                if d < fp["min_dur_s"] or any(t0 < wb and t1 > wa for wa, wb in windows):
                    continue
                if f == "BREATH" and P["ignore_soft_breaths"] and d <= P["soft_breath_max_s"] and \
                        SUBS["BREATH"][int(np.argmax(subs["BREATH"][r0:r1].mean(0)))] in ("inhale", "exhale", "other"):
                    continue  # natural breath between phrases
                unreq.append({"family": f, "s": round(t0, 2), "e": round(t1, 2)})
        tag_score = float(np.mean([t["score"] for t in per_tag])) if per_tag else None
        soft = (tag_score or 0.0) - P["lambda_unreq"] * min(len(unreq), P["unreq_cap"])
        return {"soft": round(soft, 4), "gate_ok": not read, "tag_score": tag_score, "per_tag": per_tag, "unrequested": unreq, "read_aloud": read}


class Aligner:
    """Qwen3-ForcedAligner-0.6B word times for the tag-stripped script."""

    def __init__(self, device="cuda:0"):
        import torch
        from qwen_asr import Qwen3ForcedAligner
        self.m = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map=device)

    def __call__(self, xs, texts):
        try:
            res = self.m.align(audio=[(x, SR) for x in xs], text=texts, language=["English"] * len(xs))
        except Exception as e:  # noqa: BLE001  no alignment -> windows fall back to the whole clip
            print("align failed", type(e).__name__, str(e)[:200], flush=True)
            return [[] for _ in xs]
        return [[{"w": w.text, "s": float(w.start_time), "e": float(w.end_time)} for w in ws] for ws in res]


class TagJudge:
    def __init__(self, ckpt, policy_path, device="cuda", spec_dir=None):
        import os
        import torch
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        enc = st["encoder"]
        self.net = build("microsoft/wavlm-large" if enc == "wavlm-large" else enc).to(device).eval()
        self.net.load_state_dict(st["heads"], strict=False)
        self.net.enc.to(torch.bfloat16)
        self.spec = []
        if spec_dir:
            pca = torch.load(os.path.join(spec_dir, "pca.pt"), map_location="cpu")
            self.pca = (pca["mean"].to(device), pca["comp"].to(device))
            for fn in sorted(os.listdir(spec_dir)):
                if fn.endswith(".pt") and fn != "pca.pt":
                    sd = torch.load(os.path.join(spec_dir, fn), map_location="cpu", weights_only=False)
                    m = build_spec(len(sd["classes"]), sd.get("pca_d", 384)).to(device).eval()
                    m.load_state_dict(sd["state"])
                    self.spec.append((sd["classes"], m))
        self.device = device
        self.reward = TagReward(json.load(open(policy_path)))
        self.align = Aligner()

    def probs(self, xs):
        import torch
        wav, mask, nf = batch(xs, self.device)
        with torch.no_grad():
            hs = self.net.encode(wav, mask)
            h = self.net.mix(hs)
            out = self.net.heads(hs, h)
            res = [to_numpy(out, i, nf[i]) for i in range(len(xs))]
            if self.spec:
                z = (h - self.pca[0]) @ self.pca[1]
                fm = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
                for i, n in enumerate(nf):
                    fm[i, :n] = True
                for r in res:
                    r["spec"] = {}
                for classes, m in self.spec:
                    pr = torch.sigmoid(m(z, fm)[0]).float().cpu().numpy()
                    for i, n in enumerate(nf):
                        for k, c in enumerate(classes):
                            res[i]["spec"][c] = pr[i, :n, k]
        return res

    def score(self, clips):
        """clips: [{x16, text (with inline tags), asr}] -> reward dicts; clips without tags get None."""
        parsed = [parse_tagged(c["text"]) for c in clips]
        idx = [i for i, (_, t) in enumerate(parsed) if any(x[0] for x in t)]
        out = [None] * len(clips)
        if not idx:
            return out
        xs = [clips[i]["x16"] for i in idx]
        words = self.align(xs, [parsed[i][0] for i in idx])
        for i, p, w in zip(idx, self.probs(xs), words):
            out[i] = self.reward.score(p, w, parsed[i][1], parsed[i][0], clips[i]["asr"], clips[i]["x16"])
        return out
