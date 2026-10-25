"""Qwen3-TTS-12Hz-0.6B wrapper used by every stage: sampling, exact inference prompts, teacher-forced logits, decoding.

Layout (non-streaming): [prompt P, frame embeddings + tts_pad]. P is captured from qwen-tts's own generate path, so training sees
exactly the inference prompt: ICL clone from a reference wav + its transcript, optional direction through Qwen's instruct prefix
("<|im_start|>user\\n{direction}<|im_end|>\\n", which qwen-tts disables for the 0.6B models; we call generate with instruct_ids).
Codebook 0 has 2048 codes + EOS (support index 2048); codebooks 1-15 come from the code predictor (CP).
Imported inside containers only (torch / qwen_tts are imported lazily).
"""
import re

from .common import CODES, FRAME_HZ, NCB, SPF, TTS, vpath

TAG = re.compile(r"\(([^()]{1,40})\)")  # inline tags use round brackets: "Okay (laugh) you got me."


def tags_of(text):
    return [t.strip().lower() for t in TAG.findall(text or "")]


def max_frames(est_s, cap_s=60.0):
    """Generation cap for a batch from the longest estimated duration."""
    return int(FRAME_HZ * min(cap_s, max(est_s) * 1.8 + 4))


class _Captured(Exception):
    pass


class Student:
    """Qwen3-TTS-12Hz-0.6B-Base plus a talker checkpoint. dtype "float32" for a trained copy, "bfloat16" for frozen copies."""

    def __init__(self, dtype="float32"):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.torch = torch
        self.tts = Qwen3TTSModel.from_pretrained(TTS, device_map="cuda:0", dtype=getattr(torch, dtype), attn_implementation="sdpa")
        self.m = self.tts.model
        tc = self.m.config.talker_config
        self.EOS = tc.codec_eos_token_id
        assert tc.num_code_groups == NCB, tc.num_code_groups
        self.sup0 = torch.cat([torch.arange(CODES), torch.tensor([self.EOS])]).cuda()
        gd = dict(self.tts.generate_defaults)
        self.temp, self.stemp = float(gd.get("temperature", 0.9)), float(gd.get("subtalker_temperature", 0.9))
        self.voices, self.cap = {}, {}
        self.lp = None  # optional extra codebook-0 logits processor (CapEOS / NoEOS)
        self.capture_only = False
        orig = self.m.talker.generate

        def tg(**kw):
            self.cap.update({k: kw[k] for k in ("inputs_embeds", "attention_mask", "tts_pad_embed")})
            if self.capture_only:
                raise _Captured()
            if self.lp is not None:
                from transformers import LogitsProcessorList
                kw["logits_processor"] = LogitsProcessorList([self.lp])
            return orig(**kw)
        self.m.talker.generate = tg

    # ---------------------------------------------------------------- weights
    def load_talker(self, path):
        sd = self.torch.load(vpath(path), map_location="cuda:0")
        cur = self.m.talker.state_dict()
        self.m.talker.load_state_dict({k: v.to(cur[k].dtype) for k, v in sd.items()})

    def freeze(self):
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        return self

    def train_talker(self, cp=True):
        """Full fine-tune of the talker; text_embedding / text_projection stay frozen (as in every Lure stage). cp=False also
        freezes the code predictor (when a fast head replaces it)."""
        params = []
        for n, p in self.m.named_parameters():
            ok = n.startswith("talker.") and "text_embedding" not in n and "text_projection" not in n and (cp or "code_predictor" not in n)
            p.requires_grad_(ok)
            if ok:
                params.append(p)
        return params

    def save_talker(self, path):
        self.torch.save({k: v.to(self.torch.bfloat16) for k, v in self.m.talker.state_dict().items()}, path)

    # ---------------------------------------------------------------- prompts
    def voice(self, v):
        """v: {voice_id, ref_wav (volume path), ref_text} -> cached qwen-tts voice prompt item."""
        if v["voice_id"] not in self.voices:
            import librosa
            x, _ = librosa.load(vpath(v["ref_wav"]), sr=24000, mono=True)
            it = self.tts.create_voice_clone_prompt(ref_audio=[(x.astype("float32"), 24000)], ref_text=[v["ref_text"]], x_vector_only_mode=False)[0]
            self.voices[v["voice_id"]] = it
        return self.voices[v["voice_id"]]

    def _call(self, reqs, max_new):
        tts = self.tts
        items = [self.voice(r["voice"]) for r in reqs]
        vcp = tts._prompt_items_to_voice_clone_prompt(items)
        ids = tts._tokenize_texts([tts._build_assistant_text(r["text"]) for r in reqs])
        ref_ids = [tts._tokenize_texts([tts._build_ref_text(it.ref_text)])[0] for it in items]
        ins = [tts._tokenize_texts([tts._build_instruct_text(r["direction"])])[0] if r.get("direction") else None for r in reqs]
        gk = tts._merge_generate_kwargs(max_new_tokens=max_new)
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            codes, _ = self.m.generate(input_ids=ids, instruct_ids=ins, ref_ids=ref_ids, voice_clone_prompt=vcp, languages=["English"] * len(reqs),
                                       non_streaming_mode=True, **gk)
        return codes, items

    def _captured(self, n):
        emb, mask, pad = self.cap["inputs_embeds"], self.cap["attention_mask"], self.cap["tts_pad_embed"]
        return [{"P": emb[i][mask[i].bool()].detach().float(), "pad": pad.reshape(1, -1).detach().float()} for i in range(n)]

    def prompts(self, reqs):
        """Exact inference prompts without generating a frame: [{P, pad}]."""
        was = self.m.training
        self.m.eval()
        self.capture_only = True
        try:
            self._call(reqs, 8)
            raise RuntimeError("talker.generate was not reached")
        except _Captured:
            pass
        finally:
            self.capture_only = False
            self.m.train(was)
        return self._captured(len(reqs))

    def generate(self, reqs, max_new, seed=None):
        """reqs: [{text, direction|None, voice}] -> [{P, pad, codes (T, 16) long, ended, ref_code, T}]."""
        if seed is not None:
            self.torch.manual_seed(seed)
        was = self.m.training
        self.m.eval()
        codes, items = self._call(reqs, max_new)
        self.m.train(was)
        out = []
        for i, (c, s) in enumerate(zip(codes, self._captured(len(reqs)))):
            out.append({**s, "codes": c.long(), "ended": c.shape[0] < max_new, "ref_code": items[i].ref_code.long(), "T": int(c.shape[0])})
        return out

    def decode(self, outs, bs=8):
        """Batched codec decode with the reference prefix (as qwen-tts does), reference part cut off -> list of 24 kHz float32."""
        import numpy as np
        torch = self.torch
        order = sorted(range(len(outs)), key=lambda i: outs[i]["T"])
        wavs = [None] * len(outs)
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                allc = [torch.cat([outs[i]["ref_code"].to(outs[i]["codes"].device), outs[i]["codes"]], 0) for i in idx]
                ws, sr = self.m.speech_tokenizer.decode([{"audio_codes": c} for c in allc])
                assert sr == 24000, sr
                for i, w in zip(idx, ws):
                    rc = outs[i]["ref_code"].shape[0]
                    wavs[i] = np.asarray(w, dtype=np.float32)[rc * SPF:(rc + outs[i]["T"]) * SPF]
        return wavs

    # ---------------------------------------------------------------- teacher forcing
    def hidden(self, s, tal=None, grad=False):
        """Talker hidden states that predict each frame: (n0, H), n0 = T + 1 when the take ended (last row predicts EOS)."""
        import contextlib
        torch = self.torch
        tal = tal if tal is not None else self.m.talker
        codes, P = s["codes"], s["P"].to(tal.dtype)
        T, Lp = codes.shape[0], P.shape[0]
        with (contextlib.nullcontext() if grad else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16):
            ce = tal.get_input_embeddings()(codes[:, 0])
            for k in range(NCB - 1):
                ce = ce + tal.code_predictor.get_input_embeddings()[k](codes[:, k + 1])
            x = torch.cat([P, ce + s["pad"].to(tal.dtype)]).unsqueeze(0)
            L = x.shape[1]
            pos = torch.arange(L, device=x.device).view(1, 1, L).expand(3, 1, L)
            h = tal.model(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones(1, L, device=x.device, dtype=torch.long)).last_hidden_state[0]
        n0 = T + 1 if s["ended"] else T
        return h[Lp - 1:Lp - 1 + n0]

    def logits(self, s, tal=None, grad=False):
        """-> cb0 logits (n0, 2049) over [codes, EOS] and cb1-15 logits (T, 15, 2048) from the talker's own 15-pass CP."""
        import contextlib
        torch = self.torch
        tal = tal if tal is not None else self.m.talker
        hp = self.hidden(s, tal, grad)
        with (contextlib.nullcontext() if grad else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16):
            l0 = tal.codec_head(hp).float()[:, self.sup0]
            ld, _ = tal.forward_sub_talker_finetune(s["codes"], hp[:s["codes"].shape[0]])
        return l0, ld.float()


def tf_logp(st, tal, ss):
    """Batched (right-padded) teacher forcing: per sample, temperature-scaled log-probs of the sampled cb0 codes (+EOS if ended)
    and cb1-15 as one 1-D tensor. Matches Student.logits to bf16 noise; this is the GRPO policy / reference log-prob."""
    torch = st.torch
    xs, meta = [], []
    for s in ss:
        codes, P = s["codes"], s["P"].to(tal.dtype)
        ce = tal.get_input_embeddings()(codes[:, 0])
        for k in range(NCB - 1):
            ce = ce + tal.code_predictor.get_input_embeddings()[k](codes[:, k + 1])
        xs.append(torch.cat([P, ce + s["pad"].to(tal.dtype)]))
        meta.append((P.shape[0], codes.shape[0]))
    L = max(x.shape[0] for x in xs)
    X = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, L - x.shape[0])) for x in xs])
    am = torch.stack([torch.arange(L, device=X.device) < x.shape[0] for x in xs]).long()
    pos = torch.arange(L, device=X.device).view(1, 1, L).expand(3, len(xs), L)
    h = tal.model(inputs_embeds=X, position_ids=pos, attention_mask=am).last_hidden_state
    lps, hps, cs = [], [], []
    for b, (s, (Lp, T)) in enumerate(zip(ss, meta)):
        n0 = T + 1 if s["ended"] else T
        hp = h[b, Lp - 1:Lp - 1 + n0]
        l0 = tal.codec_head(hp).float()[:, st.sup0] / st.temp
        tgt = torch.cat([s["codes"][:, 0], torch.tensor([CODES], device=hp.device)])[:n0]
        lps.append(torch.log_softmax(l0, -1).gather(-1, tgt[:, None])[:, 0])
        hps.append(hp[:T])
        cs.append(s["codes"])
    ld, _ = tal.forward_sub_talker_finetune(torch.cat(cs), torch.cat(hps))
    lsub = torch.log_softmax(ld.float() / st.stemp, -1).gather(-1, torch.cat(cs)[:, 1:, None])[..., 0]
    out, a = [], 0
    for lp0, c in zip(lps, cs):
        T = c.shape[0]
        out.append(torch.cat([lp0, lsub[a:a + T].reshape(-1)]))
        a += T
    return out


class CapEOS:
    """Per-row frame cap on codebook 0: row i is forced to EOS once it has generated caps[i] frames."""

    def __init__(self, caps, eos):
        import torch
        self.caps, self.eos = torch.as_tensor(caps, dtype=torch.long), eos

    def __call__(self, input_ids, scores):
        hit = self.caps.to(scores.device)[: scores.shape[0]] <= input_ids.shape[1]
        if hit.any():
            row = scores[0].new_full(scores[0].shape, float("-inf"))
            row[self.eos] = 0.0
            scores[hit] = row
        return scores


def kl_terms(lp_s, lp_t):
    """Summed reverse KL(S||T) and forward KL(T||S) over the last dim of log-prob tensors."""
    return (lp_s.exp() * (lp_s - lp_t)).sum(), (lp_t.exp() * (lp_t - lp_s)).sum()
