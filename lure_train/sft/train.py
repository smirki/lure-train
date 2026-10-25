"""Supervised fine-tuning of the talker on a generic manifest (lure_train/sft/data.py) with precomputed codes (sft/encode.py).

Input = the row's text (inline tags) with its direction in the instruct prefix, ICL clone from the row's reference, English,
non-streaming; the prompt is captured from qwen-tts's own generate path, so it is exactly the inference prompt.
Loss = CE(cb0 incl. EOS, mean over cb0 rows) + w_d * mean CE over cb1-15. Nothing on prompt positions. text_embedding and
text_projection frozen. Linear warmup, constant, linear decay over the last decay_frac (or --lr-const). Gradient clipping at
clip_mult x the running median grad norm. Resumes from <run>/ckpt/state.pt automatically.

  modal run --detach -m lure_train.sft.train --run sft1 --manifest /vol/data/sft/manifest.jsonl --codes /vol/data/sft/codes \
      --init /vol/ckpt/lure/talker.pt --eval-set /vol/inputs/eval.jsonl
Outputs <out>/<run>/: config.json, metrics.jsonl, heldout.jsonl, evals.jsonl, ckpt/talker_step####.pt, ckpt/state.pt.
"""
import collections
import json
import os
import time

import modal

from ..common import CODES, GPU, NCB, VOL, append, images, price_h, vpath, volume

app = modal.App("lure-sft")
vol = volume()


def lr_at(step, peak, warmup, total, decay_frac, floor):
    if step <= warmup:
        return peak * step / max(1, warmup)
    d0 = total - int(round(decay_frac * total))
    return peak if step <= d0 else peak + (floor - peak) * (step - d0) / max(1, total - d0)


def sft_terms(st, s):
    """-> ce0 sum (T+1 rows incl. EOS), cb1-15 CE sum, T."""
    import torch
    import torch.nn.functional as F
    l0, ld = st.logits(s, grad=True)
    c = s["codes"]
    ce0 = F.cross_entropy(l0.float(), torch.cat([c[:, 0], torch.tensor([CODES], device=c.device)]), reduction="sum")
    ced = F.cross_entropy(ld.float().reshape(-1, CODES), c[:, 1:].reshape(-1), reduction="sum")
    return ce0, ced, c.shape[0]


@app.function(image=images()["scorer"], gpu=GPU, volumes={VOL: vol}, timeout=12 * 3600, cpu=8, memory=96 * 1024, secrets=[])
def train(run: str, manifest: str, codes: str, init: str = "", out_root: str = "/vol/results/sft", steps: int = 0, epochs: float = 2.0,
          batch: int = 32, lr: float = 1e-5, lr_const: float = 0.0, warmup: int = 50, decay_frac: float = 0.2, lr_floor: float = 1e-6,
          w_d: float = 0.5, clip_mult: float = 3.0, clip_min: float = 0.5, max_frames: int = 500, ckpt_every: int = 50, state_every: int = 200,
          eval_every: int = 100, eval_set: str = "", mix: str = "", seed: int = 0, budget_h: float = 6.0):
    import numpy as np
    import torch
    from ..model import Student
    from .data import Mixer, load_data, req_of
    t0 = time.time()
    vol.reload()
    out = f"{vpath(out_root)}/{run}"
    os.makedirs(f"{out}/ckpt", exist_ok=True)
    sp, codes_ = load_data(manifest, codes)
    TR, HO = sp["train"], sp.get("heldout", [])[:240]
    steps = steps or int(round(epochs * len(TR) / batch))
    resume = os.path.exists(f"{out}/ckpt/state.pt")
    conf = dict(run=run, manifest=manifest, codes=codes, init=init or None, steps=steps, batch=batch, lr=lr, lr_const=lr_const, warmup=warmup,
                decay_frac=decay_frac, lr_floor=lr_floor, w_d=w_d, clip_mult=clip_mult, max_frames=max_frames, mix=mix, seed=seed,
                n_train=len(TR), n_heldout=len(HO), gpu=GPU, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not resume:
        json.dump(conf, open(f"{out}/config.json", "w"), indent=1)
    print("CONFIG", json.dumps(conf), flush=True)
    ph = price_h(GPU)
    st = Student("float32")
    if init:
        st.load_talker(init)
    params = st.train_talker()
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    byid = {r["id"]: r for k in sp for r in sp[k]}
    mixer = Mixer(TR, json.loads(mix) if mix else None, seed)
    start, gn_hist = 0, collections.deque(maxlen=50)
    if resume:
        sd = torch.load(f"{out}/ckpt/state.pt", map_location="cuda:0")
        st.m.talker.load_state_dict(sd["talker"])
        opt.load_state_dict(sd["opt"])
        start, gn_hist = sd["step"], collections.deque(sd["gn_hist"], maxlen=50)
        mixer.load(sd["samplers"])
        print(f"resumed at step {start}", flush=True)

    def sample(r):
        c = codes_[r["id"]].cuda().long()
        return {"codes": c[:max_frames], "ended": c.shape[0] <= max_frames}

    def save(step, state=False):
        st.save_talker(f"{out}/ckpt/talker_step{step:04d}.pt")
        if state:
            torch.save({"talker": st.m.talker.state_dict(), "opt": opt.state_dict(), "step": step, "gn_hist": list(gn_hist),
                        "samplers": mixer.state()}, f"{out}/ckpt/state.pt.tmp")
            os.replace(f"{out}/ckpt/state.pt.tmp", f"{out}/ckpt/state.pt")
        vol.commit()

    def heldout(step):
        if not HO:
            return
        st.m.eval()
        a0 = ad = n0 = nd = 0.0
        with torch.no_grad():
            for i in range(0, len(HO), 32):
                ch = HO[i:i + 32]
                for r, p in zip(ch, st.prompts([req_of(r) for r in ch])):
                    c0, cd, T = sft_terms(st, {**p, **sample(r)})
                    a0 += float(c0); ad += float(cd); n0 += T + 1; nd += T * (NCB - 1)
        append(f"{out}/heldout.jsonl", [{"step": step, "ce0": round(a0 / n0, 4), "ced": round(ad / nd, 4)}])
        print("HELDOUT", step, round(a0 / n0, 4), round(ad / nd, 4), flush=True)

    EV, sc = None, None

    def evaluate(step):
        nonlocal EV, sc
        if not eval_set:
            return
        from ..evaluate import load_calib, load_eval, run_eval, scorer
        EV = EV or load_eval(eval_set)
        sc = sc or scorer()
        st.m.eval()
        summ = {"step": step, **run_eval(st, sc, EV, f"{out}/eval/s{step:04d}", load_calib(os.path.join(os.path.dirname(vpath(eval_set)), "events_calib.json")))}
        append(f"{out}/evals.jsonl", [summ])
        vol.commit()
        print("EVAL", json.dumps(summ), flush=True)

    if start == 0:
        heldout(0)
        evaluate(0)
    usd = 0.0
    for step in range(start + 1, steps + 1):
        if time.time() - t0 > budget_h * 3600:
            print("time budget reached", flush=True)
            break
        ts = time.time()
        rows = [byid[i] for i in mixer.take(batch)]
        P = st.prompts([req_of(r) for r in rows])
        st.m.train()
        opt.zero_grad(set_to_none=True)
        S = [sample(r) for r in rows]
        N0, Nd = sum(s["codes"].shape[0] + 1 for s in S), sum(s["codes"].shape[0] for s in S) * (NCB - 1)
        a0 = ad = 0.0
        for p, s in zip(P, S):
            c0, cd, _ = sft_terms(st, {**p, **s})
            (c0 / N0 + w_d * cd / Nd).backward()
            a0 += float(c0); ad += float(cd)
        gn = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
        clip_at = max(clip_min, clip_mult * float(np.median(gn_hist))) if len(gn_hist) >= 5 else float("inf")
        cur_lr = lr_const if lr_const > 0 else lr_at(step, lr, warmup, steps, decay_frac, lr_floor)
        if np.isfinite(gn):
            if gn > clip_at:
                for p_ in params:
                    if p_.grad is not None:
                        p_.grad.mul_(clip_at / (gn + 1e-6))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.step()
            gn_hist.append(gn)
        dt = time.time() - ts
        usd += dt / 3600 * ph
        met = {"step": step, "loss": round(a0 / N0 + w_d * ad / Nd, 4), "ce0": round(a0 / N0, 4), "ced": round(ad / Nd, 4), "grad_norm": round(gn, 4),
               "clipped": bool(np.isfinite(gn) and gn > clip_at), "lr": cur_lr, "frames": N0 - len(S), "t_step": round(dt, 2),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "usd": round(usd, 3)}
        append(f"{out}/metrics.jsonl", [met])
        if step % 10 == 0 or step <= 3:
            print("STEP", json.dumps(met), flush=True)
        if step % ckpt_every == 0 or step == steps:
            save(step, state=step % state_every == 0 or step == steps)
            heldout(step)
            if step % eval_every == 0 or step == steps:
                evaluate(step)
    print(f"SFT DONE in {time.time() - t0:.0f}s, ${usd:.2f} on {GPU}", flush=True)


@app.local_entrypoint()
def main(run: str, manifest: str, codes: str, init: str = "", steps: int = 0, epochs: float = 2.0, batch: int = 32, lr: float = 1e-5,
         lr_const: float = 0.0, eval_set: str = "", mix: str = ""):
    train.remote(run, manifest, codes, init, "/vol/results/sft", steps, epochs, batch, lr, lr_const, eval_set=eval_set, mix=mix)
