"""Scorer server (runs as `/opt/sv/bin/python -m lure_train.rewards.worker` next to the trainer, same GPU, transformers 5.x venv).

ASR = Qwen3-ASR-1.7B + Granite Speech 5 CTC; speaker = SpeechBrain ECAPA + WavLM-base-plus-sv; quality = DNSMOS;
events = AST AudioSet probabilities (3 s windows, 1.5 s hop, max per clip).

Request {"clips": [{id, x16 (float32 16 kHz), ref (container path), refs [reference texts], hard bool, ast bool}]}
 -> [{id, asr_q, asr_g, wer_q, wer_g, ecapa, wavlm, dnsmos_*, repeat, tail_ok, ast?}]  (WER = min over refs, each capped at 2)
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lure_train.common import AST, GRANITE, QASR, WAVLM_SV
from lure_train.ipc import log, serve

AST_LABELS = ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Sigh", "Breathing", "Gasp", "Pant", "Wheeze",
              "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff", "Groan", "Grunt", "Crying, sobbing", "Whimper", "Humming", "Speech",
              "Silence", "Snoring", "Burping, eructation", "Hiccup", "Screaming", "Whispering"]
FILLER = re.compile(r"^(?:(?:ha)+h?|(?:he)+h?|ah+|h+m+|m+h?m*|u+h+|u+m+|uhm+|e+r+m*|p+f+t*|u+g+h+|ow+|tsk|psh+|hee+|heh+)$")
try:
    from num2words import num2words as N2W
except Exception:  # noqa: BLE001
    N2W = None


def norm_words(s):
    """Tags and bracketed text removed, digits as words, lowercase, fillers dropped."""
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>", " ", s or "")
    if N2W:
        s = re.sub(r"\d+", lambda m: " " + N2W(int(m.group(0))) + " " if len(m.group(0)) < 10 else m.group(0), s)
    s = re.sub(r"[^a-z' ]", " ", s.lower().replace("-", " ")).replace("'", "")
    return [w for w in s.split() if not FILLER.match(w)]


def collapse(ws, drop_and=False):
    """Runs of single letters joined ('f b i' == 'fbi'); optionally 'and' dropped (number readings)."""
    out, buf = [], ""
    for w in ws + [""]:
        if len(w) == 1:
            buf += w
            continue
        if buf:
            out.append(buf)
            buf = ""
        if w and not (drop_and and w == "and"):
            out.append(w)
    return out


def wer_one(ref, hyp, hard=False):
    import jiwer
    r, h = norm_words(ref), norm_words(hyp)
    if hard:
        r, h = collapse(r, True), collapse(h, True)
    r, h = " ".join(r), " ".join(h)
    if not r:
        return 0.0 if not h else 1.0
    return round(min(float(jiwer.wer(r, h)), 2.0), 4) if h else 1.0


def max_rep(ws):
    """Longest run of an immediately repeated 1-3-gram. Single words need 4 copies ("no no no" is normal speech)."""
    best = 1
    for n in (1, 2, 3):
        for i in range(len(ws) - n + 1):
            k = 1
            while i + (k + 1) * n <= len(ws) and ws[i + k * n:i + (k + 1) * n] == ws[i:i + n]:
                k += 1
            best = max(best, k if n > 1 else (k - 1 if k >= 2 else 1))
    return best


class Models:
    def __init__(self):
        import onnxruntime as ort
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import (ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor, AutoModelForCTC,
                                  AutoModelForMultimodalLM, AutoProcessor, WavLMForXVector)
        t0 = time.time()
        self.torch = torch
        self.qp = AutoProcessor.from_pretrained(QASR)
        self.qm = AutoModelForMultimodalLM.from_pretrained(QASR, device_map="cuda", dtype=torch.bfloat16).eval()
        self.gp = AutoProcessor.from_pretrained(GRANITE)
        self.gm = AutoModelForCTC.from_pretrained(GRANITE).to("cuda").eval()
        self.ec = EncoderClassifier.from_hparams(source="/opt/ecapa", savedir="/opt/ecapa", run_opts={"device": "cuda"})
        self.wfe = AutoFeatureExtractor.from_pretrained(WAVLM_SV)
        self.wm = WavLMForXVector.from_pretrained(WAVLM_SV).to("cuda").eval()
        self.afe = ASTFeatureExtractor.from_pretrained(AST)
        self.am = ASTForAudioClassification.from_pretrained(AST).to("cuda").eval().half()
        lab = self.am.config.label2id
        self.alabels = [lbl for lbl in AST_LABELS if lbl in lab]
        self.aidx = [lab[lbl] for lbl in self.alabels]
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        self.dns = [(ort.InferenceSession("/opt/dnsmos/sig_bak_ovr.onnx", opt, providers=["CPUExecutionProvider"]),
                     ort.InferenceSession("/opt/dnsmos/model_v8.onnx", opt, providers=["CPUExecutionProvider"])) for _ in range(6)]
        self.pool = ThreadPoolExecutor(6)
        self.refc, self.qbatch_ok = {}, None
        log(f"[scorer] models up in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- ASR
    def _q(self, xs):
        mx = max(len(x) for x in xs) / 16000
        inp = self.qp.apply_transcription_request(audio=list(xs), language=["English"] * len(xs)).to(self.qm.device, self.qm.dtype)
        with self.torch.inference_mode():
            o = self.qm.generate(**inp, max_new_tokens=max(320, int(mx * 6) + 48), do_sample=False)
        out = self.qp.decode(o[:, inp["input_ids"].shape[1]:], return_format="transcription_only")
        return [t.strip() for t in (out if isinstance(out, list) else [out])]

    def qwen(self, xs):
        if self.qbatch_ok is None and len(xs) >= 4:  # padded batches must transcribe like single clips, else go one by one
            b, s = self._q(xs[:4]), [self._q([x])[0] for x in xs[:4]]
            self.qbatch_ok = all(wer_one(a, c) <= 0.05 for a, c in zip(s, b))
            log(f"[scorer] qwen3-asr batch check ok={self.qbatch_ok}")
        return self._batched(xs, self._q) if self.qbatch_ok else [self._q([x])[0] for x in xs]

    def _granite(self, xs):
        inp = self.gp(list(xs), sampling_rate=16000)
        inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
        with self.torch.no_grad():
            return [t.strip() for t in self.gp.batch_decode(self.gm.generate(**inp), skip_special_tokens=True)]

    def _batched(self, xs, fn):
        order = sorted(range(len(xs)), key=lambda i: len(xs[i]))
        res, i = {}, 0
        while i < len(order):
            bs = 16 if len(xs[order[min(i + 15, len(order) - 1)]]) < 25 * 16000 else 4
            idx = order[i:i + bs]
            for j, t in zip(idx, fn([xs[k] for k in idx])):
                res[j] = t
            i += bs
        return [res[i] for i in range(len(xs))]

    # ---------------------------------------------------------------- speaker
    def ecapa(self, x):
        x = x if len(x) >= 1600 else np.pad(x, (0, 1600 - len(x)))
        with self.torch.no_grad():
            e = self.ec.encode_batch(self.torch.from_numpy(np.ascontiguousarray(x))[None].to("cuda"))[0, 0]
        return (e / e.norm()).cpu()

    def wavlm(self, xs):
        out = []
        for i in range(0, len(xs), 16):
            b = [(x if len(x) >= 1600 else np.pad(x, (0, 1600 - len(x))))[:30 * 16000] for x in xs[i:i + 16]]
            with self.torch.no_grad():
                inp = self.wfe(b, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
                out += list(self.torch.nn.functional.normalize(self.wm(**inp).embeddings, dim=-1).cpu())
        return out

    def ref(self, path):
        if path not in self.refc:
            import librosa
            x = librosa.load(path, sr=16000, mono=True)[0].astype(np.float32)
            self.refc[path] = (self.ecapa(x), self.wavlm([x])[0])
        return self.refc[path]

    # ---------------------------------------------------------------- DNSMOS P.835 + P.808 (9.01 s windows, no overlap)
    def dnsmos(self, k, x):
        import librosa
        sess, p808s = self.dns[k % len(self.dns)]
        n = int(9.01 * 16000)
        x = np.asarray(x, dtype=np.float32)
        if len(x) < 1600:
            return {}
        while len(x) < n:
            x = np.concatenate([x, x])
        out = []
        for s in range(0, max(1, len(x) - n + 1), n):
            seg = x[s:s + n]
            mel = librosa.feature.melspectrogram(y=seg[:-160], sr=16000, n_fft=321, hop_length=160, n_mels=120)
            mel = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T.astype(np.float32)
            p808 = float(p808s.run(None, {"input_1": mel[None]})[0][0][0])
            sig, bak, ovr = sess.run(None, {"input_1": seg[None]})[0][0]
            out.append((np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig), np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak),
                        np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr), p808))
        m = np.mean(out, axis=0)
        return {"dnsmos_sig": round(float(m[0]), 3), "dnsmos_bak": round(float(m[1]), 3), "dnsmos_ovrl": round(float(m[2]), 3),
                "dnsmos_p808": round(float(m[3]), 3)}

    # ---------------------------------------------------------------- AST events
    def ast(self, xs, win=3.0, hop=1.5, bs=64):
        segs, owner = [], []
        W, H = int(win * 16000), int(hop * 16000)
        for i, x in enumerate(xs):
            for s in list(range(0, max(1, len(x) - W + H), H)) or [0]:
                segs.append(x[s:s + W])
                owner.append(i)
        probs = []
        for i in range(0, len(segs), bs):
            f = self.afe(segs[i:i + bs], sampling_rate=16000, return_tensors="pt")["input_values"].to("cuda").half()
            with self.torch.no_grad():
                probs.append(self.torch.sigmoid(self.am(input_values=f).logits.float())[:, self.aidx].cpu().numpy())
        out = [np.zeros(len(self.aidx)) for _ in xs]
        for o, p in zip(owner, np.concatenate(probs) if probs else []):
            out[o] = np.maximum(out[o], p)
        return [{lbl: round(float(v), 4) for lbl, v in zip(self.alabels, o)} for o in out]

    # ---------------------------------------------------------------- one request
    def score(self, req):
        t0 = time.time()
        clips = req["clips"]
        xs = [np.asarray(c["x16"], dtype=np.float32) for c in clips]
        dfut = [self.pool.submit(self.dnsmos, k, x) for k, x in enumerate(xs)]
        qa, ga, wl = self.qwen(xs), self._batched(xs, self._granite), self.wavlm(xs)
        need_ast = [i for i, c in enumerate(clips) if c.get("ast")]
        asts = dict(zip(need_ast, self.ast([xs[i] for i in need_ast]))) if need_ast else {}
        out = []
        for i, (c, x) in enumerate(zip(clips, xs)):
            re_, rw = self.ref(c["ref"])
            wq = [wer_one(r, qa[i], c.get("hard", False)) for r in c["refs"]]
            wg = [wer_one(r, ga[i], c.get("hard", False)) for r in c["refs"]]
            rw_ = norm_words(c["refs"][int(np.argmin([a + b for a, b in zip(wq, wg)]))])
            hq, hg = norm_words(qa[i]), norm_words(ga[i])
            ref_rep, rep_q, rep_g = max_rep(rw_), max_rep(hq), max_rep(hg)
            # a loop must show in Granite, and also in Qwen3-ASR (or be a 4x loop in Granite alone)
            repeat = (rep_g >= 3 and rep_g > ref_rep) and (rep_q >= 3 and rep_q > ref_rep or rep_g >= 4)
            tail = rw_[-2:]
            m = {"id": c["id"], "asr_q": qa[i], "asr_g": ga[i], "wer_q": min(wq), "wer_g": min(wg), "ecapa": round(float(self.ecapa(x) @ re_), 4),
                 "wavlm": round(float(wl[i] @ rw), 4), "repeat": bool(repeat), "rep_q": rep_q, "rep_g": rep_g,
                 "tail_ok": bool(tail) and any(all(w in h[-7:] for w in tail) for h in (hq, hg)), **dfut[i].result()}
            if i in asts:
                m["ast"] = asts[i]
            out.append(m)
        log(f"[scorer] {len(clips)} clips ({sum(len(x) for x in xs) / 16000:.0f} s audio) in {time.time() - t0:.1f}s")
        return out


if __name__ == "__main__":
    serve(Models().score)
