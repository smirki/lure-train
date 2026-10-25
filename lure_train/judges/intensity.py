"""Intensity judge (int_v2): how strongly is the emotion played, compared with a plain read by the same voice? Inference only.

Frozen Qwen3-ASR audio encoder (AuT, every layer) -> weighted layer sum -> adapter -> attentive stats pooling -> scalar latent s.
Ordinal output (CORAL): P(level > k) = sigmoid(scale * s - b_k) over neutral / medium / full; expected level E in 0..2.
Policy is code, not model: the reward matches a requested dial level (never "more is better"), and gates on the RAW take
(clipping, too loud, near silent, F0 drift vs the reference, speaking-rate band) floor it.

Checkpoint: head_cal.pt {"config": {backbone, n_layers, dim, cond}, "state"}; thresholds JSON with dial_targets, over_top_latent,
gate_clip_frac, gate_max_lufs, optional gate_rate_ratio + nominal_wps, optional f0_caps.
Every judge input uses preprocess(): 16 kHz, light -45 dB trim, centre crop to 12 s, -23 LUFS with a 0.98 peak cap.
Runs in the /opt/judges venv (qwen-asr package), never in the TTS process.
"""
import json

import numpy as np

SR = 16000
TARGET_LUFS, PEAK_CAP, MAX_S = -23.0, 0.98, 12.0
LEVELS = ["neutral", "medium", "full"]
# Optional steer conditioning of the head: 38 learned steer embeddings + one "unknown" (index 38, what the reward uses).
N_STEERS = 38
UNKNOWN_STEER = N_STEERS
QWEN_ASR = "Qwen/Qwen3-ASR-1.7B"
GATE_KEYS = ("gate_clipping", "gate_too_loud", "gate_silent", "gate_f0", "gate_rate")
HARD_GATES = ("gate_clipping", "gate_too_loud", "gate_silent")


# ------------------------------------------------------------------------------------------------ audio
def to16k(x, sr):
    if sr == SR:
        return np.asarray(x, dtype=np.float32)
    import soxr
    return soxr.resample(x, sr, SR, quality="HQ").astype(np.float32)


def loudness(x, sr=SR):
    import pyloudnorm as pyln
    if len(x) < int(0.4 * sr) or not np.any(x):
        return -70.0
    v = pyln.Meter(sr).integrated_loudness(np.asarray(x, dtype=np.float64))
    return float(v) if np.isfinite(v) else -70.0


def clip_stats(x):
    a = np.abs(x)
    return {"peak": float(a.max()) if len(a) else 0.0, "clip_frac": float(np.mean(a >= 0.999)) if len(a) else 0.0}


def normalize(x, sr=SR, target=TARGET_LUFS):
    lv = loudness(x, sr)
    if lv <= -69.0:
        return np.asarray(x, dtype=np.float32)
    y = x * (10.0 ** ((target - lv) / 20.0))
    p = float(np.abs(y).max())
    return (y * (PEAK_CAP / p) if p > PEAK_CAP else y).astype(np.float32)


def crop(x, max_s=MAX_S):
    n = int(max_s * SR)
    return x if len(x) <= n else x[(len(x) - n) // 2:(len(x) - n) // 2 + n]


def trim(x, db=-45.0, pad=0.1):
    frame = 400
    n = len(x) // frame
    if n < 3:
        return x
    e = 20 * np.log10(np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-9)
    on = np.where(e > e.max() + db)[0]
    if not len(on):
        return x
    return x[max(0, on[0] * frame - int(pad * SR)):min(len(x), (on[-1] + 1) * frame + int(pad * SR))]


def preprocess(x, sr=SR):
    return normalize(crop(trim(to16k(x, sr))))


def f0_median_st(x, sr=SR, octave_tol=0.8):
    """Median F0 of voiced frames in semitones re 55 Hz (pYIN), octave errors dropped. None under 10 voiced frames."""
    import librosa
    f0, voiced, _ = librosa.pyin(np.asarray(x, dtype=np.float32), fmin=65, fmax=1000, sr=sr, frame_length=1024, hop_length=320)
    v = f0[voiced & np.isfinite(f0)] if f0 is not None else np.array([])
    if len(v) < 10:
        return None
    v = v[np.abs(np.log2(v / np.median(v))) <= octave_tol]
    return float(12 * np.log2(np.median(v) / 55.0)) if len(v) >= 10 else None


def speech_seconds(x, sr=SR, db=-35.0):
    fr = int(0.025 * sr)
    n = len(x) // fr
    if n < 2:
        return len(x) / sr
    e = 10 * np.log10((np.asarray(x[:n * fr], dtype=np.float64).reshape(n, fr) ** 2).mean(1) + 1e-12)
    return float((e > e.max() + db).sum() * fr / sr)


def n_words(text):
    import re
    return len(re.findall(r"[^\W_]+(?:'[^\W_]+)?", re.sub(r"[\[(][^\])]*[\])]", " ", text or "")))


# ------------------------------------------------------------------------------------------------ encoder + head
class QwenAudioEncoder:
    """Frozen Qwen3-ASR audio tower; __call__(wavs16k) -> (hs [L, B, T, D], mask [B, T]). The 1.7B LLM is dropped after loading."""

    def __init__(self, device="cuda"):
        import torch
        from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
        self.device, self.dtype = device, torch.bfloat16
        full = Qwen3ASRForConditionalGeneration.from_pretrained(QWEN_ASR, torch_dtype=self.dtype)
        self.m = full.thinker.audio_tower.to(device).eval()
        del full
        self.fe = Qwen3ASRProcessor.from_pretrained(QWEN_ASR).feature_extractor
        self.n_layers, self.dim = len(self.m.layers), self.m.config.d_model
        for p in self.m.parameters():
            p.requires_grad_(False)
        self._cap = []
        for layer in self.m.layers:
            layer.register_forward_hook(lambda _m, _a, out: self._cap.append(out[0] if isinstance(out, tuple) else out))

    def __call__(self, wavs):
        import torch
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths
        with torch.no_grad():
            f = self.fe(wavs, sampling_rate=SR, padding="longest", return_attention_mask=True, return_tensors="pt")
            flen = f.attention_mask.sum(-1)
            packed = torch.cat([f.input_features[i, :, :int(flen[i])] for i in range(len(wavs))], dim=1)
            self._cap.clear()
            self.m(packed.to(self.device, self.dtype), feature_lens=flen.to(self.device))
            lens = _get_feat_extract_output_lengths(flen).long().to(self.device)
            T = int(lens.max())
            hs = torch.zeros(len(self._cap), len(wavs), T, self.dim, device=self.device, dtype=self.dtype)
            for li, h in enumerate(self._cap):
                for b, seg in enumerate(torch.split(h, lens.tolist(), dim=0)):
                    hs[li, b, :seg.shape[0]] = seg
            self._cap.clear()
            mask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        return hs, mask


def build_head(n_layers, dim, hid=256, n_levels=3, cond=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class IntensityHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_w = nn.Parameter(torch.zeros(n_layers))
            self.norm = nn.LayerNorm(dim)
            self.adapt = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.GELU())
            self.att = nn.Linear(hid, 1)
            self.cond = cond
            self.steer = nn.Embedding(N_STEERS + 1, 2 * hid) if cond else None
            self.out = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))
            self.b0 = nn.Parameter(torch.tensor(-1.0))
            self.gaps = nn.Parameter(torch.zeros(n_levels - 2) + 0.5)
            self.register_buffer("scale", torch.tensor(1.0))

        def thresholds(self):
            return torch.cat([self.b0[None], self.b0 + torch.cumsum(F.softplus(self.gaps) + 1e-3, 0)])

        def forward(self, hs, mask, steer=None):
            x = torch.einsum("l,lbtd->btd", torch.softmax(self.layer_w, 0).to(hs.dtype), hs).float()
            h = self.adapt(self.norm(x))
            a = torch.softmax(self.att(h).squeeze(-1).masked_fill(~mask, -1e4), -1).unsqueeze(-1)
            mu = (a * h).sum(1)
            z = torch.cat([mu, (a * (h - mu[:, None]) ** 2).sum(1).clamp_min(1e-6).sqrt()], -1)
            if self.cond and steer is not None:
                z = z + self.steer(steer)
            return self.out(z).squeeze(-1)

        def ordinal_logits(self, s):
            return self.scale * s[:, None] - self.thresholds()[None, :]

        def level_probs(self, s):
            g = torch.sigmoid(self.ordinal_logits(s))
            cum = torch.cat([torch.ones_like(g[:, :1]), g, torch.zeros_like(g[:, :1])], 1)
            return (cum[:, :-1] - cum[:, 1:]).clamp_min(0)

        def expected(self, s):
            return torch.sigmoid(self.ordinal_logits(s)).sum(-1)

    return IntensityHead()


# ------------------------------------------------------------------------------------------------ gates + reward (policy)
def gate_flags(raw16, th, ref16=None, text=None, ref_text=None, steer=None, target=None):
    """Code-owned gates on the RAW 16 kHz take (before normalisation)."""
    cs, lv = clip_stats(raw16), loudness(raw16)
    out = {"raw_lufs": round(lv, 2), "clip_frac": round(cs["clip_frac"], 5), "gate_clipping": cs["clip_frac"] > th["gate_clip_frac"],
           "gate_too_loud": lv > th["gate_max_lufs"], "gate_silent": lv < -50}
    if ref16 is not None and ("f0_caps" in th or "gate_f0_drift_st" in th):
        ft, fr = f0_median_st(raw16), f0_median_st(ref16)
        out["f0_drift_st"] = None if ft is None or fr is None else round(ft - fr, 2)
        if "f0_caps" in th:  # per steer: rise allowed = p90 of real full-vs-neutral shifts, scaled down for lower dial levels
            fc = th["f0_caps"]
            cp = fc["by_steer"].get((steer or "").lower(), fc["pooled"])
            up, down = cp["up_cap_st"], cp["down_cap_st"]
            if up is not None and target is not None:
                up = max(fc["min_up_cap_st"], up * fc["dial_scale"][int(target)])
        else:
            up = down = th["gate_f0_drift_st"]
        out["gate_f0"] = bool(up is not None and out["f0_drift_st"] is not None and (out["f0_drift_st"] > up or out["f0_drift_st"] < -down))
    if text and "gate_rate_ratio" in th:
        wps = n_words(text) / max(speech_seconds(raw16), 0.3)
        ref_wps = n_words(ref_text) / max(speech_seconds(ref16), 0.3) if (ref_text and ref16 is not None) else th["nominal_wps"]
        out["rate_ratio"] = round(wps / max(ref_wps, 1e-3), 3)
        lo, hi = th["gate_rate_ratio"]
        out["gate_rate"] = bool(not lo <= out["rate_ratio"] <= hi)
    return out


def intensity_reward(r, target, th):
    """Match the requested dial level (0..3) -> -|E - dial target| in [-2, 0]. Over the top costs 1 unless the strongest level
    was asked for; any gate floors the reward."""
    if any(r.get(k) for k in GATE_KEYS):
        return -2.0
    rew = -abs(r["expected"] - list(th["dial_targets"].values())[int(target)])
    if r["over_top"] and int(target) < 3:
        rew -= 1.0
    return float(max(rew, -2.0))


class IntensityJudge:
    def __init__(self, ckpt, thresholds, encoder=None, device="cuda"):
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cf = ck["config"]
        assert cf["backbone"] == "qwen3asr", cf["backbone"]
        self.enc = encoder or QwenAudioEncoder(device)
        self.head = build_head(cf["n_layers"], cf["dim"], cond=cf.get("cond", False)).to(device).eval()
        self.head.load_state_dict(ck["state"])
        self.th = json.load(open(thresholds))
        self.device = device

    def expected(self, wavs, bs=64):
        """Preprocessed 16 kHz wavs -> (latent s, expected level E) numpy arrays."""
        import torch
        S, E = [], []
        with torch.no_grad():
            for i in range(0, len(wavs), bs):
                hs, mask = self.enc(wavs[i:i + bs])
                s = self.head(hs, mask, torch.full((hs.shape[1],), UNKNOWN_STEER, device=self.device))
                S.append(s.float().cpu()); E.append(self.head.expected(s).float().cpu())
        return torch.cat(S).numpy(), torch.cat(E).numpy()

    def score(self, items):
        """items: [{x16 (raw 16 kHz take), target? (dial 0-3), text?, ref16?, ref_text?, steer?}] -> [{expected, over_top, gates, reward?}]"""
        raws = [trim(np.asarray(it["x16"], dtype=np.float32)) for it in items]
        s, e = self.expected([normalize(crop(x)) for x in raws])
        out = []
        for it, x, si, ei in zip(items, raws, s, e):
            ref = trim(it["ref16"]) if it.get("ref16") is not None else None
            r = {"expected": float(ei), "latent": float(si), "over_top": bool(si > self.th["over_top_latent"]),
                 **gate_flags(x, self.th, ref, it.get("text"), it.get("ref_text"), it.get("steer"), it.get("target"))}
            if it.get("target") is not None:
                r["reward"] = intensity_reward(r, it["target"], self.th)
            out.append(r)
        return out
