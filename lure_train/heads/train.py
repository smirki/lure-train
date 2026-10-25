"""Warm-up of a FastCP head (V2, then V2s) against the frozen stock code predictor. The talker is frozen.

Data: any SFT manifest + codes (lure_train/sft). Per frame: teacher-forced talker hidden state and the stock CP log-probs of the
real codes; the head is trained on KL(CP || head) + ce_w * CE(real code) at every position 1..15. Cosine lr.
V2s warm-up: --name V2s --init-from <V2 head .pt> loads the trunk (add-on missing, stays at its zero-output init) and trains the
trunk at trunk_lr_mult x lr. The resulting head is the starting point of lure_train.opd.train (which also trains the talker).

  modal run --detach -m lure_train.heads.train --run head_v2 --name V2 --talker /vol/ckpt/lure/talker.pt \
      --manifest /vol/data/sft/manifest.jsonl --codes /vol/data/sft/codes
  modal run --detach -m lure_train.heads.train --run head_v2s --name V2s --init-from /vol/results/heads/head_v2/V2.pt --trunk-lr-mult 0.3 ...
Outputs <out>/<run>/: config.json, metrics.jsonl, heldout.jsonl, <name>.pt (bf16 state dict).
"""
import collections
import json
import math
import os
import time

import modal

from ..common import GPU, NCB, VOL, append, images, spend_guard, vpath, volume

app = modal.App("lure-heads")
vol = volume()
SPEND = os.environ.get("LURE_HEADS_SPEND", "lure-heads-spend")


def cosine_lr(step, peak, warmup, total, floor):
    if step <= warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, p)))


@app.function(image=images()["tts"], gpu=GPU, volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=64 * 1024)
def train(run: str, name: str, talker: str, manifest: str, codes: str, out_root: str = "/vol/results/heads", init_from: str = "",
          steps: int = 600, batch: int = 32, lr: float = 1e-4, trunk_lr_mult: float = 1.0, warmup_frac: float = 0.05, floor_frac: float = 0.1,
          ce_w: float = 0.5, clip: float = 1.0, max_frames: int = 300, chunk: int = 2048, eval_every: int = 100, n_heldout: int = 64,
          seed: int = 0, budget_usd: float = 10.0, cap_usd: float = 40.0):
    import numpy as np
    import torch
    from ..model import Student
    from ..sft.data import Sampler, load_data, req_of, sha
    from .fastcp import HEADS, new_head
    t0 = time.time()
    usd, total = spend_guard(SPEND, f"{run}_{int(t0)}", GPU, cap_usd, t0)
    vol.reload()
    out = f"{vpath(out_root)}/{run}"
    os.makedirs(out, exist_ok=True)
    conf = dict(run=run, name=name, groups=HEADS[name], talker=talker, manifest=manifest, init_from=init_from or None, steps=steps, batch=batch,
                lr=lr, trunk_lr_mult=trunk_lr_mult, ce_w=ce_w, clip=clip, max_frames=max_frames, seed=seed, gpu=GPU,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(conf, open(f"{out}/config.json", "w"), indent=1)
    print("CONFIG", json.dumps(conf), flush=True)

    sp, codes_ = load_data(manifest, codes)
    TR = sp["train"]
    HO = sorted(sp.get("heldout", []) or TR, key=lambda r: sha(r["id"]))[:n_heldout]
    byid = {r["id"]: r for k in sp for r in sp[k]}
    samp = Sampler([r["id"] for r in TR], seed)
    te = Student("bfloat16")
    te.load_talker(talker)
    te.freeze()
    H = new_head(te, name).to(device="cuda:0", dtype=torch.float32).train()
    if init_from:
        miss = H.load_state_dict(torch.load(vpath(init_from), map_location="cuda:0"), strict=False)
        print(f"init from {init_from}: missing {len(miss.missing_keys)} (add-on), unexpected {len(miss.unexpected_keys)}", flush=True)
        H.float()
    for p in H.parameters():
        p.requires_grad_(True)
    add = [p for k, p in H.named_parameters() if k.startswith("addon")]
    trunk = [p for k, p in H.named_parameters() if not k.startswith("addon")]
    opt = torch.optim.AdamW([g for g in ({"params": add, "m": 1.0}, {"params": trunk, "m": trunk_lr_mult}) if g["params"]], lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    def frames_of(rows):
        """rows -> (h (N, H), codes (N, 16), stock CP log-probs (N, 15, 2048) bf16)."""
        hs, cs, lts = [], [], []
        for r, P in zip(rows, te.prompts([req_of(r) for r in rows])):
            c = codes_[r["id"]].cuda().long()[:max_frames]
            hp = te.hidden({"codes": c, "ended": False, **P})
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ld, _ = te.m.talker.forward_sub_talker_finetune(c, hp)
            hs.append(hp); cs.append(c); lts.append(ld.float().log_softmax(-1).bfloat16())
        return torch.cat(hs), torch.cat(cs), torch.cat(lts)

    def head_loss(hid, cc, lt, grad, acc, scale):
        for i in range(0, hid.shape[0], chunk):
            hh, c, l_t = hid[i:i + chunk], cc[i:i + chunk], lt[i:i + chunk].float()
            with torch.autocast("cuda", dtype=torch.bfloat16), (torch.enable_grad() if grad else torch.no_grad()):
                ls = H.logp(hh, c)
            kl = (l_t.exp() * (l_t - ls)).sum(-1)
            ce = -ls.gather(-1, c[:, 1:, None]).squeeze(-1)
            if grad:
                ((kl + ce_w * ce).sum() * scale).backward()
            acc["kl"] += float(kl.detach().sum())
            acc["nll"] += float(ce.detach().sum())
            acc["top1"] += float((ls.detach().argmax(-1) == l_t.argmax(-1)).sum())
            acc["t_nll"] += float(-l_t.gather(-1, c[:, 1:, None]).sum())

    def summ(acc, N):
        return {"frames": int(N), **{k: round(acc[k] / (N * (NCB - 1)), 4) for k in ("kl", "nll", "top1", "t_nll")}}

    EV = [frames_of(HO[i:i + 16]) for i in range(0, len(HO), 16)]

    def heldout(step):
        acc, N = collections.defaultdict(float), 0
        H.eval()
        for hid, cc, lt in EV:
            head_loss(hid, cc, lt, False, acc, 1.0)
            N += hid.shape[0]
        H.train()
        row = {"step": step, **summ(acc, N)}
        append(f"{out}/heldout.jsonl", [row])
        print("HELDOUT", json.dumps(row), flush=True)

    def save():
        torch.save({k: v.to(torch.bfloat16) for k, v in H.state_dict().items()}, f"{out}/{name}.pt")
        vol.commit()

    heldout(0)
    warmup, last = max(1, int(round(warmup_frac * steps))), 0
    for step in range(1, steps + 1):
        if usd() > budget_usd or total() > cap_usd - 1.0:
            print(f"budget reached at step {step - 1}: launch ${usd():.2f}, total ${total():.2f}", flush=True)
            break
        ts = time.time()
        hid, cc, lt = frames_of([byid[i] for i in samp.take(batch)])
        N = hid.shape[0]
        acc = collections.defaultdict(float)
        opt.zero_grad(set_to_none=True)
        head_loss(hid, cc, lt, True, acc, 1.0 / (N * (NCB - 1)))
        gn = float(torch.nn.utils.clip_grad_norm_(H.parameters(), clip))
        cur = cosine_lr(step, lr, warmup, steps, lr * floor_frac)
        if np.isfinite(gn):
            for g in opt.param_groups:
                g["lr"] = cur * g["m"]
            opt.step()
        met = {"step": step, "lr": cur, "grad_norm": round(gn, 3), "t_step": round(time.time() - ts, 2), "usd": round(usd(), 3), **summ(acc, N)}
        append(f"{out}/metrics.jsonl", [met])
        if step % 10 == 0 or step <= 3:
            print("STEP", json.dumps(met), flush=True)
        last = step
        if step % eval_every == 0 or step == steps:
            heldout(step)
            save()
    if last and last % eval_every and last != steps:
        heldout(last)
        save()
    print(f"HEAD DONE step {last} in {time.time() - t0:.0f}s, ${usd():.2f} on {GPU} (total ${total():.2f})", flush=True)


@app.local_entrypoint()
def main(run: str, name: str, talker: str, manifest: str, codes: str, init_from: str = "", steps: int = 600, lr: float = 1e-4,
         trunk_lr_mult: float = 1.0, budget_usd: float = 10.0):
    train.remote(run, name, talker, manifest, codes, init_from=init_from, steps=steps, lr=lr, trunk_lr_mult=trunk_lr_mult, budget_usd=budget_usd)
