"""Emotion judge (e1 head on emotion2vec+ large features). Inference and reward policy only.

Answers: probabilities over CLASSES (temperature-calibrated) and a Platt-calibrated "emotional at all" guard. The reward uses the
intensity judge's expected level as the emotional-at-all signal when it is available (POLICY["EMOTIONAL_SOURCE"]).
Head checkpoint: {"classes", "version", "backbone": "e2v_ms", "L", "D", "hid", "state", "temperature", "guard_platt": [a, b]}.
Encoder: iic/emotion2vec_plus_large from ModelScope through funasr, every transformer layer, batched with a padding mask.
"""
import math
import re

import numpy as np

VERSION = "emo-v0"  # class schema version stored in the head checkpoint
CLASSES = ["happy", "amused", "warm", "calm", "sad", "angry", "fearful", "disgusted", "surprised", "dismissive", "unsure", "assertive", "other"]
IDX = {c: i for i, c in enumerate(CLASSES)}
# prompt emotion field -> target ("neutral" is a guard target, not a class)
PROMPT_TO_TARGET = {"neutral": None, "calm": "calm", "happy": "happy", "sad": "sad", "angry": "angry", "fearful": "fearful",
                    "disgusted": "disgusted", "surprised": "surprised", "amused": "amused"}
# free-text direction words -> class (first match wins; unmatched directions get no emotion reward)
DIRECTION_WORDS = {
    "happy": ["happy", "joyful", "cheerful", "excited", "elated", "delighted", "proud", "thrilled"],
    "amused": ["amused", "laughing", "playful", "teasing", "giggly", "mischievous"],
    "warm": ["warm", "tender", "affectionate", "soothing", "gentle", "loving", "reassuring", "encouraging", "grateful"],
    "calm": ["calm", "relaxed", "relieved", "serene", "peaceful", "unhurried"],
    "sad": ["sad", "sorrowful", "grieving", "tearful", "crying", "heartbroken", "disappointed", "downcast", "melancholy"],
    "angry": ["angry", "furious", "irritated", "frustrated", "annoyed", "enraged", "livid"],
    "fearful": ["afraid", "scared", "fearful", "terrified", "anxious", "panicked", "nervous"],
    "disgusted": ["disgusted", "revolted", "grossed"],
    "surprised": ["surprised", "astonished", "shocked", "amazed", "stunned"],
    "dismissive": ["sarcastic", "bored", "dismissive", "unimpressed", "contemptuous", "deadpan", "sardonic"],
    "unsure": ["hesitant", "unsure", "confused", "embarrassed", "apologetic", "sheepish", "guilty", "uncertain"],
    "assertive": ["assertive", "confident", "firm", "commanding", "authoritative", "decisive"],
}
POLICY = {
    # credit = p_t + NEIGHBOUR_CREDIT * sum of neighbour probs; runner = max(best non-neighbour, (1 - credit) * best neighbour)
    # m = clip(log credit - log runner, M_LO, M_HI); reward = W_MARGIN * m / 3. M_HI is low on purpose (no pay for over-acting).
    "M_LO": -3.0, "M_HI": 1.0, "W_MARGIN": 1.0, "NEIGHBOUR_CREDIT": 0.5,
    "NEIGHBOURS": [["happy", "amused"], ["angry", "assertive"], ["happy", "warm"], ["warm", "calm"], ["sad", "unsure"], ["happy", "surprised"]],
    # flat read for an emotional target: -W_FLAT * max(0, FLAT_E - E) / FLAT_E (E = intensity expected level), not for calm
    "EMOTIONAL_SOURCE": "intensity", "W_FLAT": 0.5, "FLAT_E": 0.7, "NO_FLAT_FOR": ["calm"],
    # neutral target: -W_NEUTRAL * max(0, E - NEUTRAL_E) / (2 - NEUTRAL_E)
    "W_NEUTRAL": 1.0, "NEUTRAL_E": 0.3,
    "FLAT_AT": 0.5, "NEUTRAL_OK": 0.35,  # fallback on this judge's guard when no intensity is available
    "OTHER_BLOCKS": True, "HIT": 0.6, "MISS": 0.2,
}


def direction_target(text):
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    for c, ws in DIRECTION_WORDS.items():
        if words & set(ws):
            return c
    return None


def target_for(item):
    """Prompt item -> target class, "neutral", or None. Explicit emotion field first, then a direction word."""
    e = (item.get("emotion") or "").strip().lower()
    if e in PROMPT_TO_TARGET:
        return PROMPT_TO_TARGET[e] or "neutral"
    if e in IDX and e != "other":
        return e
    return direction_target(item.get("direction"))


def neighbours(target, policy=POLICY):
    return sorted({b if a == target else a for a, b in policy["NEIGHBOURS"] if target in (a, b)})


def emotion_reward(s, target, policy=POLICY):
    """s: one answer dict ({probs, emotional, top, intensity?}). -> (reward, parts)."""
    if target is None:
        return 0.0, {"skip": True}
    use_int = policy["EMOTIONAL_SOURCE"] == "intensity" and s.get("intensity") is not None
    e = s["intensity"] if use_int else s["emotional"]
    if target == "neutral":
        ok, top = (policy["NEUTRAL_E"], 2.0) if use_int else (policy["NEUTRAL_OK"], 1.0)
        return -policy["W_NEUTRAL"] * max(0.0, e - ok) / (top - ok), {"emotional": e, "band": "hit" if e < ok else "miss"}
    p, nb = s["probs"], neighbours(target, policy)
    c = policy["NEIGHBOUR_CREDIT"] if nb else 0.0
    pt = max(p[target], 1e-6)
    credit = pt + c * sum(p[n] for n in nb)
    runner = max(max(v for k, v in p.items() if k != target and k not in nb), (1 - c) * max(p[n] for n in nb) if nb else 0.0)
    m = math.log(credit) - math.log(max(runner, 1e-6))
    if policy["OTHER_BLOCKS"] and s["top"] == "other":
        m = min(m, 0.0)
    m = min(max(m, policy["M_LO"]), policy["M_HI"])
    r = policy["W_MARGIN"] * m / max(abs(policy["M_LO"]), policy["M_HI"])
    flat = 0.0
    if target not in policy["NO_FLAT_FOR"]:
        flat = (policy["W_FLAT"] * max(0.0, policy["FLAT_E"] - e) / policy["FLAT_E"] if use_int else policy["W_FLAT"] * max(0.0, policy["FLAT_AT"] - e))
        r -= flat
    band = "hit" if (pt >= policy["HIT"] and s["top"] == target) else "miss" if pt < policy["MISS"] else "unsure"
    return r, {"p_target": pt, "margin": m, "emotional": e, "flat_pen": flat, "band": band, "top": s["top"]}


def pooled_stats(hs, mask):
    """hs [L, B, T, D], mask [B, T] -> float32 tensor (B, L, 2, D) of masked mean and std over time."""
    import torch
    h, m = hs.float(), mask[None, :, :, None].float()
    n = m.sum(2).clamp(min=1.0)
    mean = (h * m).sum(2) / n
    var = (((h - mean[:, :, None]) ** 2) * m).sum(2) / n
    return torch.stack([mean, var.clamp(min=0).sqrt()], 2).permute(1, 0, 2, 3).contiguous()


class E2VEncoder:
    """emotion2vec+ large (ModelScope iic/emotion2vec_plus_large), every block exposed through forward hooks, batched."""

    def __init__(self, device="cuda", model_id="iic/emotion2vec_plus_large", hub="ms"):
        import torch
        from funasr import AutoModel
        self.device = device
        self.net = AutoModel(model=model_id, hub=hub, disable_update=True, device=device).model.to(device).eval()
        blocks = None
        for _, mod in self.net.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and (blocks is None or len(mod) > len(blocks)):
                blocks = mod
        assert blocks is not None, "transformer blocks not found in emotion2vec"
        self._cap = []
        for b in blocks:
            b.register_forward_hook(lambda _m, _i, o: self._cap.append(o[0] if isinstance(o, (tuple, list)) else o))
        self.n_extra = 0
        try:
            for enc in self.net.modality_encoders.values():
                self.n_extra = int(enc.modality_cfg.num_extra_tokens)
        except Exception:  # noqa: BLE001
            pass
        self.normalize = bool(getattr(getattr(self.net, "cfg", None), "normalize", True))

    def __call__(self, wavs):
        import torch
        lens = [len(w) for w in wavs]
        N = max(lens)
        src = torch.zeros(len(wavs), N, device=self.device)
        pm = torch.ones(len(wavs), N, dtype=torch.bool, device=self.device)
        for b, w in enumerate(wavs):
            x = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(self.device)
            if self.normalize:
                x = torch.nn.functional.layer_norm(x, x.shape)
            src[b, :len(x)] = x
            pm[b, :len(x)] = False
        self._cap.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.net.extract_features(src, padding_mask=pm if min(lens) < N else None)
        x = feats["x"].float()
        T = x.shape[1]
        hs = torch.stack([h.float()[:, self.n_extra:self.n_extra + T] for h in self._cap] + [x], 0)
        self._cap.clear()
        fpm = feats.get("padding_mask")
        mask = torch.ones(len(wavs), T, dtype=torch.bool, device=self.device) if fpm is None else ~fpm[:, :T].bool()
        return hs, mask


def build_heads(L, D, n_cls, hid=256):
    import torch
    import torch.nn as nn

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(L))
            self.net = nn.Sequential(nn.LayerNorm(2 * D), nn.Linear(2 * D, hid), nn.GELU(), nn.Dropout(0.0), nn.Linear(hid, hid), nn.GELU(),
                                     nn.Dropout(0.0))

        def forward(self, f):
            return self.net((torch.softmax(self.w, 0)[None, :, None, None] * f).sum(1).flatten(1))

    class EmotionHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.choice_t, self.guard_t = Tower(), Tower()
            self.choice, self.guard = nn.Linear(hid, n_cls), nn.Linear(hid, 1)

        def forward(self, f):
            return self.choice(self.choice_t(f)), self.guard(self.guard_t(f)).squeeze(-1)

    return EmotionHeads()


class EmotionJudge:
    def __init__(self, ckpt, device="cuda"):
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert ck["classes"] == CLASSES and ck["version"] == VERSION, "head trained on another class schema"
        assert ck["backbone"] == "e2v_ms", f"this port loads e2v_ms heads only, got {ck['backbone']}"
        self.T, (self.pa, self.pb) = ck["temperature"], ck["guard_platt"]
        self.m = build_heads(ck["L"], ck["D"], len(CLASSES), ck["hid"])
        self.m.load_state_dict(ck["state"])
        self.m.to(device).eval()
        self.enc = E2VEncoder(device)

    def score(self, wavs, intensity=None):
        """Preprocessed 16 kHz wavs (judges.intensity.preprocess) -> answer dicts. intensity: optional expected levels."""
        import torch
        with torch.no_grad():
            lc, lg = self.m(pooled_stats(*self.enc(wavs)))
            P = torch.softmax(lc / self.T, -1).cpu().numpy()
            G = torch.sigmoid(self.pa * lg + self.pb).cpu().numpy()
        out = []
        for i, (p, g) in enumerate(zip(P, G)):
            k = int(p.argmax())
            out.append({"probs": dict(zip(CLASSES, p.tolist())), "emotional": float(g), "top": CLASSES[k], "top_p": float(p[k]),
                        "intensity": None if intensity is None else float(intensity[i])})
        return out
