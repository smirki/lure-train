"""On-policy distillation (OPD) from one or more frozen teacher talkers into one student talker, optionally through a FastCP head.

Single teacher + V2s head = the Lure Turbo recipe: the student (talker + V2s head) samples on-policy, the frozen teacher (talker +
stock 15-pass CP) teacher-forces those codes, and the student is pulled toward the teacher at every sampled code.

Loss per sampled code: KL(S||T) + fwd_w * KL(T||S).
  cb1-15: S = head (or student CP) at that position given the sampled context, T = the branch teacher's stock CP.
  cb0 (weight w0): the talker's codebook-0 distribution incl. EOS.
Plus n_kd offline clips (rows that have codes) with forward KL only (weight kd_w), an anchor against on-policy drift.
Cosine lr; talker at lr_tal (text_embedding / text_projection frozen; stock CP frozen when a head is used), head at lr_head.

  modal run --detach -m lure_train.opd.train --config configs/opd_turbo.json --run opd_turbo
Outputs <out>/<run>/: config.json, metrics.jsonl, evals.jsonl, ckpt/talker_s####.pt (+ head_s####.pt), eval/s####/.
"""
import collections
import json
import math
import os
import time

import modal

from ..common import GPU, NCB, VOL, append, images, spend_guard, vpath, volume

app = modal.App("lure-opd")
vol = volume()
SPEND = os.environ.get("LURE_OPD_SPEND", "lure-opd-spend")
DEFAULTS = dict(teachers={}, student_init="", head="", head_init="", manifest="", codes="", eval_set="", calib="", branch_weights={},
                steps=120, n_on=16, n_kd=8, lr_head=3e-5, lr_tal=2e-6, train_talker=True, fwd_w=0.5, kd_w=0.5, w0=1.0, clip=1.0,
                max_frames=400, warmup_frac=0.05, floor_frac=0.1, evals_at=[0.5, 1.0], seed=1, budget_usd=12.0, cap_usd=40.0)


def cosine(step, warmup, total, floor):
    if step <= warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * min(1.0, p)))


@app.function(image=images()["scorer"], gpu=GPU, volumes={VOL: vol}, timeout=8 * 3600, cpu=8, memory=96 * 1024)
def train(cfg: dict, run: str, out_root: str = "/vol/results/opd", smoke: bool = False):
    import numpy as np
    import torch
    from ..model import Student, kl_terms
    from ..sft.data import Sampler, load_codes, load_manifest, plain_of, req_of
    c = {**DEFAULTS, **cfg}
    if smoke:
        run, c["steps"], c["n_on"], c["n_kd"], c["evals_at"] = run + "_smoke", 3, 4, 2, [1.0]
    assert c["teachers"], "cfg.teachers: {branch: talker checkpoint}"
    t0 = time.time()
    usd, total = spend_guard(SPEND, f"{run}_{int(t0)}", GPU, c["cap_usd"], t0)
    vol.reload()
    out = f"{vpath(out_root)}/{run}"
    os.makedirs(f"{out}/ckpt", exist_ok=True)
    json.dump({**c, "run": run, "gpu": GPU, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(f"{out}/config.json", "w"), indent=1)
    print("CONFIG", json.dumps(c), flush=True)

    branches = sorted(c["teachers"])
    rows = load_manifest(c["manifest"])
    for r in rows:
        r.setdefault("branch", branches[0])
    rows = [r for r in rows if r["branch"] in c["teachers"] and r["split"] == "train"]
    codes = load_codes(c["codes"]) if c["codes"] else {}
    byid = {r["id"]: r for r in rows}
    by_b = collections.defaultdict(list)
    for r in rows:
        by_b[r["branch"]].append(r["id"])
    w = {b: float(c["branch_weights"].get(b, 1.0)) for b in branches if by_b[b]}
    on_s = {b: Sampler(by_b[b], c["seed"] + 101 + i) for i, b in enumerate(sorted(w))}
    kd_ids = [i for i in byid if i in codes]
    kd_s = Sampler(kd_ids, c["seed"]) if kd_ids else None
    print(f"branches {dict((b, len(by_b[b])) for b in branches)}, kd clips {len(kd_ids)}", flush=True)

    T = {}
    for b in branches:
        T[b] = Student("bfloat16")
        T[b].load_talker(c["teachers"][b])
        T[b].freeze()
    st = Student("float32")
    st.load_talker(c["student_init"] or c["teachers"][branches[0]])
    use_head = bool(c["head"])
    tal_params = st.train_talker(cp=not use_head) if c["train_talker"] else []
    if not c["train_talker"]:
        st.freeze()
    groups = []
    H = None
    if use_head:
        from ..heads.fastcp import CPPatch, load_head
        H = load_head(st, c["head"], c["head_init"]).float().train()
        for p in H.parameters():
            p.requires_grad_(True)
        CPPatch(st).set(H, 0.9)
        groups.append({"params": list(H.parameters()), "lr": c["lr_head"], "base": c["lr_head"]})
    if tal_params:
        groups.append({"params": tal_params, "lr": c["lr_tal"], "base": c["lr_tal"]})
    assert groups, "nothing to train"
    allp = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    def student_cp_logp(hp, cc, grad):
        with torch.autocast("cuda", dtype=torch.bfloat16), (torch.enable_grad() if grad else torch.no_grad()):
            if H is not None:
                return H.logp(hp, cc)
            ld, _ = st.m.talker.forward_sub_talker_finetune(cc, hp)
            return ld.float().log_softmax(-1)

    def terms(s, pt, ps, te):
        """s: {codes, ended}; pt / ps: teacher / student prompt. -> summed KL terms (grad on the student side)."""
        with torch.no_grad():
            l0, ld = te.logits({**s, **pt})
            lt0, ltd = l0.log_softmax(-1), ld.log_softmax(-1)
        n = s["codes"].shape[0]
        hp = st.hidden({**s, **ps}, grad=bool(tal_params))
        lsd = student_cp_logp(hp[:n], s["codes"], True)
        rd, fd = kl_terms(lsd, ltd)
        r = {"rd": rd, "fd": fd, "top1": float((lsd.detach().argmax(-1) == ltd.argmax(-1)).sum())}
        if tal_params:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls0 = st.m.talker.codec_head(hp).float()[:, st.sup0].log_softmax(-1)
            r["r0"], r["f0"] = kl_terms(ls0, lt0)
        return r

    def est_frames(r):
        s = r.get("est_s") or len(plain_of(r["text"]).split()) / 2.5
        return int(12.5 * s * 1.5 + 25)

    def take_on(n):
        tot, ks = sum(w.values()), sorted(w)
        cnt = {k: int(n * w[k] / tot) for k in ks}
        for k in ks[: n - sum(cnt.values())]:
            cnt[k] += 1
        return [byid[i] for k in ks for i in on_s[k].take(cnt[k])]

    def prompts_by_branch(rs):
        """Teacher prompts, each row built by its own branch's teacher."""
        out_ = [None] * len(rs)
        for b in {r["branch"] for r in rs}:
            idx = [i for i, r in enumerate(rs) if r["branch"] == b]
            for i, p in zip(idx, T[b].prompts([req_of(rs[i]) for i in idx])):
                out_[i] = p
        return out_

    EV, sc = None, None

    def save_eval(step):
        nonlocal EV, sc
        st.save_talker(f"{out}/ckpt/talker_s{step:04d}.pt")
        if H is not None:
            torch.save({k: v.to(torch.bfloat16) for k, v in H.state_dict().items()}, f"{out}/ckpt/head_s{step:04d}.pt")
        vol.commit()
        if not c["eval_set"]:
            return
        from ..evaluate import load_calib, load_eval, run_eval, scorer
        EV = EV or load_eval(c["eval_set"])[: 8 if smoke else None]
        sc = sc or scorer()
        if H is not None:
            H.eval(); H._W = {}
        summ = {"step": step, **run_eval(st, sc, EV, f"{out}/eval/s{step:04d}", load_calib(c["calib"]))}
        if H is not None:
            H.train()
        append(f"{out}/evals.jsonl", [summ])
        vol.commit()
        print("EVAL", json.dumps(summ), flush=True)

    steps = c["steps"]
    ev_steps = sorted({max(1, round(steps * float(q))) for q in c["evals_at"]})
    warmup, last = max(1, int(round(c["warmup_frac"] * steps))), 0
    fwd_w, w0, kd_w = c["fwd_w"], c["w0"], c["kd_w"]
    for step in range(1, steps + 1):
        if usd() > c["budget_usd"] or total() > c["cap_usd"] - 0.6:
            print(f"budget reached at step {step - 1}: launch ${usd():.2f}, total ${total():.2f}", flush=True)
            break
        ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        on_rows = take_on(c["n_on"])
        mf = min(c["max_frames"], max(est_frames(r) for r in on_rows))
        if H is not None:
            H.eval(); H._W = {}
        gen = st.generate([req_of(r) for r in on_rows], mf, seed=c["seed"] * 100003 + step)
        if H is not None:
            H.train()
        t_gen = time.time() - ts
        on = [({"codes": g["codes"], "ended": g["ended"]}, pt, {"P": g["P"], "pad": g["pad"]}, T[r["branch"]], r["branch"])
              for r, g, pt in zip(on_rows, gen, prompts_by_branch(on_rows)) if g["codes"].shape[0] >= 2]
        kd = []
        if kd_s is not None and c["n_kd"]:
            kr = [byid[i] for i in kd_s.take(c["n_kd"])]
            for r, pt, ps in zip(kr, prompts_by_branch(kr), st.prompts([req_of(r) for r in kr])):
                cc = codes[r["id"]].cuda().long()
                kd.append(({"codes": cc[:c["max_frames"]], "ended": cc.shape[0] <= c["max_frames"]}, pt, ps, T[r["branch"]], r["branch"]))
        N0 = sum(s["codes"].shape[0] + int(s["ended"]) for s, *_ in on) or 1
        Nd = sum(s["codes"].shape[0] for s, *_ in on) * (NCB - 1) or 1
        K0 = sum(s["codes"].shape[0] + int(s["ended"]) for s, *_ in kd) or 1
        Kd = sum(s["codes"].shape[0] for s, *_ in kd) * (NCB - 1) or 1
        opt.zero_grad(set_to_none=True)
        acc = collections.defaultdict(float)
        for s, pt, ps, te, b in on:
            r = terms(s, pt, ps, te)
            loss = (r["rd"] + fwd_w * r["fd"]) / Nd
            if tal_params:
                loss = loss + w0 * (r["r0"] + fwd_w * r["f0"]) / N0
                acc["rkl0"] += float(r["r0"]); acc["fkl0"] += float(r["f0"])
            loss.backward()
            acc["rkl_cp"] += float(r["rd"]); acc["fkl_cp"] += float(r["fd"]); acc["top1"] += r["top1"]; acc["ended"] += float(s["ended"])
            acc[f"rkl_cp/{b}"] += float(r["rd"]); acc[f"frames/{b}"] += s["codes"].shape[0] * (NCB - 1)
        for s, pt, ps, te, _ in kd:
            r = terms(s, pt, ps, te)
            loss = kd_w * r["fd"] / Kd
            if tal_params:
                loss = loss + kd_w * w0 * r["f0"] / K0
                acc["kd_kl0"] += float(r["f0"])
            loss.backward()
            acc["kd_kl_cp"] += float(r["fd"])
        gn = float(torch.nn.utils.clip_grad_norm_(allp, c["clip"]))
        f = cosine(step, warmup, steps, c["floor_frac"])
        ok = bool(np.isfinite(gn))
        if ok:
            for g in opt.param_groups:
                g["lr"] = g["base"] * f
            opt.step()
        met = {"step": step, "rkl_cp": round(acc["rkl_cp"] / Nd, 4), "fkl_cp": round(acc["fkl_cp"] / Nd, 4), "rkl0": round(acc["rkl0"] / N0, 4),
               "fkl0": round(acc["fkl0"] / N0, 4), "top1_cp": round(acc["top1"] / Nd, 4), "kd_kl_cp": round(acc["kd_kl_cp"] / Kd, 4),
               "kd_kl0": round(acc["kd_kl0"] / K0, 4), "per_branch_rkl_cp": {b: round(acc[f"rkl_cp/{b}"] / acc[f"frames/{b}"], 4)
                                                                             for b in branches if acc[f"frames/{b}"]},
               "n_on": len(on), "ended_frac": round(acc["ended"] / max(1, len(on)), 3), "max_new": mf, "grad_norm": round(gn, 4), "skipped": not ok,
               "lr_f": round(f, 4), "t_gen": round(t_gen, 1), "t_step": round(time.time() - ts, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "usd": round(usd(), 3)}
        append(f"{out}/metrics.jsonl", [met])
        print("STEP", json.dumps(met), flush=True)
        last = step
        if step in ev_steps:
            save_eval(step)
    if last and last not in ev_steps:
        save_eval(last)
    print(f"OPD DONE step {last} in {time.time() - t0:.0f}s, ${usd():.2f} on {GPU} (total ${total():.2f})", flush=True)


@app.local_entrypoint()
def main(config: str, run: str, smoke: bool = False):
    train.remote(json.load(open(config)), run, smoke=smoke)
