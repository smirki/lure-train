"""Task-vector merge of specialist branches: merged = base + sum_i w_i * (branch_i - base). The cheap baseline that multi-teacher
OPD (lure_train.opd.train) should beat. Math in fp32, saved as bf16. Also reports per-branch delta norms and pairwise cosines.

  modal run --detach -m lure_train.merge --base /vol/ckpt/lure/talker.pt \
      --branches '{"V": "/vol/results/grpo/v1/ckpt/talker_step0150.pt", "R": "/vol/results/grpo/r1/ckpt/talker_step0120.pt"}' \
      --weights '{"V": 1.0, "R": 1.0}' --out /vol/results/merge/vr11
Then evaluate <out>/talker.pt with lure_train.evaluate. CPU only.
"""
import json
import os

import modal

from .common import VOL, images, vpath, volume

app = modal.App("lure-merge")
vol = volume()


def merge_state(base, branches, weights):
    """base: state dict, branches: {name: state dict}, weights: {name: float} -> (merged bf16 state dict, stats)."""
    import torch
    for n, sd in branches.items():
        assert set(sd) == set(base), f"key mismatch: {n}"
    names = sorted(branches)
    sq = {n: 0.0 for n in names}
    dot = {(a, b): 0.0 for i, a in enumerate(names) for b in names[i + 1:]}
    out = {}
    for k, v in base.items():
        b32 = v.float()
        d = {n: branches[n][k].float() - b32 for n in names}
        for n in names:
            sq[n] += float((d[n] * d[n]).sum())
        for a, b in dot:
            dot[(a, b)] += float((d[a] * d[b]).sum())
        out[k] = (b32 + sum(float(weights.get(n, 0.0)) * d[n] for n in names)).to(torch.bfloat16)
    stats = {"norm": {n: round(sq[n] ** 0.5, 4) for n in names},
             "cos": {f"{a}|{b}": round(v / max((sq[a] * sq[b]) ** 0.5, 1e-20), 4) for (a, b), v in dot.items()}, "weights": weights}
    return out, stats


@app.function(image=images()["tts"], cpu=4, memory=32 * 1024, volumes={VOL: vol}, timeout=3600)
def merge(base: str, branches: dict, weights: dict, out: str):
    import torch
    vol.reload()
    load = lambda p: torch.load(vpath(p), map_location="cpu")  # noqa: E731
    sd, stats = merge_state(load(base), {n: load(p) for n, p in branches.items()}, weights)
    d = vpath(out)
    os.makedirs(d, exist_ok=True)
    torch.save(sd, f"{d}/talker.pt")
    json.dump({"base": base, "branches": branches, **stats}, open(f"{d}/merge.json", "w"), indent=1)
    vol.commit()
    print("MERGE", json.dumps(stats), flush=True)
    return stats


@app.local_entrypoint()
def main(base: str, branches: str, weights: str, out: str):
    merge.remote(base, json.loads(branches), json.loads(weights), out)
