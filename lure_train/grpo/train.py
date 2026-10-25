"""Train a specialist teacher branch with Dr.GRPO. Each branch is a config (configs/grpo_*.json) of this one trainer; the
specialists are later distilled back into one model by multi-teacher OPD (lure_train/opd/train.py).

  V  voice cloning fidelity (shipped as Lure: V step 150 on top of the SFT start point)
  R  robustness / long-form (hard text, 30-75 s pieces)
  events / emotion / intensity: planned branches; set judge weights in the config (judges run in their own venv, LURE_JUDGES=1)

Method: advantage = R - group mean (no std division), group 8, 16 prompts per step, one on-policy update per step, zero-variance
groups skipped. Loss per token = -A * logp + beta * k3(ref, policy), summed over cb0 (+EOS) and the 15 CP codes, divided by a
constant (rollouts x 16 x 250), not by the take's length. AdamW (0.9, 0.99), no weight decay, grad clip 1.0, full fine-tune of the
talker with text_embedding / text_projection frozen. Sampling = the inference path (temp 0.9, top-k 50, rep. penalty 1.05); log-probs
use the temperature-scaled distribution.

One GPU: policy (fp32) + frozen KL reference (bf16) + bf16 generation copy (re-synced after each update) here; the scorer
(Qwen3-ASR, Granite, ECAPA, WavLM-SV, DNSMOS, AST) in a subprocess on the same GPU. Pipelined: the scorer scores batch N while
batch N+1 is generated (so a batch is sampled one update behind).

Inputs (--inputs, a volume dir): voices.jsonl {voice_id, ref_wav, ref_text}, lines.jsonl {id, text, plain, est_s, direction?,
emotion?, intensity?}, long.jsonl and hard.jsonl (R; build with lure_train/grpo/prompts.py), eval.jsonl (see evaluate.py),
optional events_calib.json and passage_ref.wav (R long-passage eval). Eval voices are excluded from the training pool.
Outputs <out>/<run>/: config.json, metrics.jsonl, rollouts.jsonl (read these before changing a reward), evals.jsonl,
eval/s####/, samples/s####/, ckpt/talker_step####.pt (bf16), spend.json.

  modal run --detach -m lure_train.grpo.train --config configs/grpo_v.json --run v1 --init /vol/ckpt/sft/talker.pt
  modal run --detach -m lure_train.grpo.train --config configs/grpo_r.json --run r1 --init /vol/ckpt/sft/talker.pt
  LURE_JUDGES=1 modal run --detach -m lure_train.grpo.train --config configs/grpo_emotion.example.json --run e1 --init ...
Stop: modal app list, then modal app stop -y <app id>.
"""
import collections
import json
import os
import random
import threading
import time

import modal

from ..common import GPU, NCB, VOL, append, images, jl, price_h, vpath, volume
from ..evaluate import SCORER, load_calib, load_eval, run_eval, to16
from ..ipc import Client
from ..model import CapEOS, Student, tags_of, tf_logp
from ..rewards.reward import cap_frames, reward

GEN_BUDGET = 64000  # rows x frames per generate call (qwen-tts keeps every step's hidden states)
SPEND = os.environ.get("LURE_GRPO_SPEND", "lure-grpo-spend")  # modal.Dict shared by all branch runs (global cap)
JUDGES = ("/opt/judges/bin/python", "lure_train.judges.worker")
PASSAGE = (  # single-call long-form test (305 words), branch R evals
    "The storm had been building all afternoon, and by the time Mara reached the lighthouse, the sky was the color of a bruise. "
    "She pushed the heavy door open with her shoulder and stepped inside, shaking the rain from her coat. "
    "The keeper, an old man named Tobias, looked up from his chair by the stove and raised an eyebrow. "
    "\"You're either very brave or very lost,\" he said. "
    "Mara laughed, a little too loudly, and admitted that it was probably a bit of both. "
    "She had driven three hours along the coast road, following a letter that had arrived at her apartment with no return address. "
    "Inside the envelope was a single photograph of this lighthouse, and on the back, in her grandmother's handwriting, were four words: come before the winter. "
    "Tobias studied the photograph for a long time. Then he stood, slowly, and walked to a wooden cabinet against the far wall. "
    "He unlocked it with a brass key and took out a small tin box, dented and rusted at the corners. "
    "\"Your grandmother left this with me forty years ago,\" he said quietly. \"She told me someone would come for it one day. I stopped believing her a long time ago.\" "
    "Mara held the box in both hands. It was lighter than she expected. "
    "Outside, the wind howled against the glass, and somewhere far below, the waves crashed against the rocks. "
    "She opened the lid. Inside were dozens of letters, tied together with a faded blue ribbon, and a map drawn by hand on yellowed paper. "
    "At the center of the map was a small island she had never heard of, circled twice in red ink. "
    "Mara looked up at Tobias, and for the first time that night, he smiled. "
    "\"Well,\" he said, \"I suppose you'll be needing a boat.\""
)
DEFAULTS = dict(branch="V", steps=250, prompts=16, group=8, lr=4e-6, beta=0.1, clip=1.0, eval_every=25, n_long=0, n_hard=0, max_line_s=20.0,
                budget_usd=30.0, total_cap_usd=60.0, reserve_usd=1.5, seed=0, reward={}, judge_w={}, judges={}, passage_takes=0,
                passage_ref_text="")

app = modal.App(os.environ.get("GRPO_APP", "lure-grpo"))
vol = volume()


@app.function(image=images()["grpo"], gpu=GPU, volumes={VOL: vol}, timeout=12 * 3600, cpu=8, memory=64 * 1024)
def train(cfg: dict, run: str, init: str, ref: str = "", inputs: str = "/vol/inputs", out_root: str = "/vol/results/grpo", smoke: bool = False):
    import numpy as np
    import soundfile as sf
    import torch
    t0 = time.time()
    c = {**DEFAULTS, **cfg}
    br = c["branch"]
    if smoke:
        c.update(steps=3, prompts=4, group=4, eval_every=3, n_long=min(c["n_long"], 1), n_hard=min(c["n_hard"], 1), passage_takes=min(c["passage_takes"], 2))
        run += "_smoke"
    ref = ref or init  # KL reference = the start checkpoint unless continuing from another run
    vol.reload()
    out = f"{vpath(out_root)}/{run}"
    for d in ("ckpt", "samples", "eval"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    for f in ("metrics.jsonl", "rollouts.jsonl", "evals.jsonl"):
        if os.path.exists(f"{out}/{f}"):
            os.remove(f"{out}/{f}")
    ph = price_h(GPU)
    conf = dict(c, run=run, init=init, kl_ref=ref, inputs=inputs, gpu=GPU, price_h=round(ph, 3), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(conf, open(f"{out}/config.json", "w"), indent=1)
    print("CONFIG", json.dumps(conf), flush=True)
    rng = random.Random(c["seed"])
    spend = modal.Dict.from_name(SPEND, create_if_missing=True)
    usd = lambda: (time.time() - t0) / 3600 * ph  # noqa: E731

    def total_usd():
        spend[run] = round(usd(), 3)
        return sum(float(v) for v in spend.values())

    # ---------------------------------------------------------------- scorer (and judges) load while the TTS models load
    box = {}
    th = threading.Thread(target=lambda: box.update(sc=Client(*SCORER),
                                                    jg=Client(*JUDGES, env={"LURE_JUDGE_CFG": json.dumps(c["judges"])}) if c["judge_w"] else None))
    th.start()

    # ---------------------------------------------------------------- data
    inp = vpath(inputs)
    EV = load_eval(f"{inp}/eval.jsonl")
    calib = load_calib(f"{inp}/events_calib.json")
    ev_refs = {e["voice"]["ref_wav"] for e in EV}
    pool = [v for v in jl(f"{inp}/voices.jsonl") if v.get("ref_wav") and (v.get("ref_text") or "").strip() and v["ref_wav"] not in ev_refs]
    lines = []
    for r in jl(f"{inp}/lines.jsonl"):
        if r["est_s"] <= c["max_line_s"]:
            lines.append({**r, "plain": r.get("plain", r["text"]), "alts": r.get("alts", []), "n_tags": len(tags_of(r["text"]))})
    LONG = jl(f"{inp}/long.jsonl") if c["n_long"] else []
    HARD = jl(f"{inp}/hard.jsonl") if c["n_hard"] else []
    for L in (pool, lines, LONG, HARD):
        rng.shuffle(L)
    print(f"voices {len(pool)}, lines {len(lines)}, long {len(LONG)}, hard {len(HARD)}, eval {len(EV)}", flush=True)
    cur = collections.Counter()

    def take(name, L):
        cur[name] += 1
        return L[(cur[name] - 1) % len(L)]

    def make_batch():
        its = [dict(take("long", LONG), src="long") for _ in range(c["n_long"])] + [dict(take("hard", HARD), src="hard") for _ in range(c["n_hard"])]
        its += [dict(take("line", lines), src="line") for _ in range(c["prompts"] - len(its))]
        for it in its:
            it["voice"] = take("voice", pool)
        return its

    # ---------------------------------------------------------------- models
    pol = Student("float32")
    pol.load_talker(init)
    refm = Student("bfloat16")
    refm.load_talker(ref)
    gen = Student("bfloat16")
    gen.load_talker(init)
    refm.freeze(), gen.freeze()
    m = pol.m
    params = pol.train_talker()
    pnamed = dict(m.talker.named_parameters())
    gnamed = list(gen.m.talker.named_parameters())
    opt = torch.optim.AdamW(params, lr=c["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M, models up {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def sync():
        for n, p in gnamed:
            p.copy_(pnamed[n].detach())

    def generate(items, caps, seed_):
        gen.lp = CapEOS(caps, gen.EOS)
        try:
            outs = gen.generate([{"text": it["text"], "direction": it.get("direction"), "voice": it["voice"]} for it in items], max(caps), seed=seed_)
        finally:
            gen.lp = None
        for o, cp in zip(outs, caps):
            o["forced"] = o["T"] >= cp
            o["ended"] = bool(o["ended"]) and not o["forced"]  # a forced EOS was not sampled: keep it out of the log-probs
        return outs

    def microbatches(idx, ss, budget=6000):
        idx = sorted(idx, key=lambda i: ss[i]["T"])
        mbs, cur_, tok = [], [], 0
        for i in idx:
            L = ss[i]["T"] + ss[i]["P"].shape[0]
            if cur_ and (tok + L > budget or len(cur_) >= 16):
                mbs.append(cur_)
                cur_, tok = [], 0
            cur_.append(i)
            tok += L
        return mbs + ([cur_] if cur_ else [])

    def evaluate(step):
        sync()
        d = f"{out}/eval/s{step:04d}"
        summ = {"step": step, **run_eval(gen, box["sc"], EV[:8] if smoke else EV, f"{d}/set", calib)}
        if c["passage_takes"] and os.path.exists(f"{inp}/passage_ref.wav"):
            pv = {"voice_id": "passage_ref", "ref_wav": f"{inp}/passage_ref.wav", "ref_text": c["passage_ref_text"]}
            outs = gen.generate([{"text": PASSAGE, "direction": None, "voice": pv}] * c["passage_takes"], int(12.5 * 240), seed=2002)
            ws = gen.decode(outs)
            os.makedirs(f"{d}/passage", exist_ok=True)
            for k, w in enumerate(ws):
                sf.write(f"{d}/passage/take{k}.wav", w, 24000, subtype="PCM_16")
            res = box["sc"]({"clips": [{"id": f"p{k}", "x16": to16(w), "ref": pv["ref_wav"], "refs": [PASSAGE]} for k, w in enumerate(ws)]})
            pw = [0.5 * (r["wer_q"] + r["wer_g"]) for r in res]
            summ.update(passage_wer=pw, passage_wer_mean=round(float(np.mean(pw)), 4), passage_wer_max=round(max(pw), 4),
                        passage_dur=[round(len(w) / 24000, 1) for w in ws], passage_ended=[bool(o["ended"]) for o in outs])
        summ["usd"] = round(usd(), 3)
        append(f"{out}/evals.jsonl", [summ])
        vol.commit()
        print("EVAL", json.dumps(summ), flush=True)

    def save(step):
        pol.save_talker(f"{out}/ckpt/talker_step{step:04d}.pt")
        vol.commit()

    # ---------------------------------------------------------------- pipeline: produce batch k (generate + decode + submit)
    def produce(k):
        tp = time.time()
        its = make_batch()
        G = c["group"]
        items = [it for it in its for _ in range(G)]
        caps = [cap_frames(it) for it in items]
        order = sorted(range(len(its)), key=lambda g: caps[g * G])  # chunks of whole groups within the memory budget
        chunks, cur_ = [], []
        for g in order:
            if cur_ and (len(cur_) + 1) * G * max(caps[x * G] for x in cur_ + [g]) > GEN_BUDGET:
                chunks.append(cur_)
                cur_ = []
            cur_.append(g)
        chunks.append(cur_)
        outs = [None] * len(items)
        for ci, ch in enumerate(chunks):
            rows_ = [g * G + j for g in ch for j in range(G)]
            for i, o in zip(rows_, generate([items[i] for i in rows_], [caps[i] for i in rows_], c["seed"] * 100003 + k * 101 + ci)):
                outs[i] = o
            torch.cuda.empty_cache()
        x16 = [to16(w) for w in gen.decode(outs, bs=16)]
        clips = [{"id": f"s{k}_{i}", "x16": x, "ref": vpath(it["voice"]["ref_wav"]), "refs": [it["plain"]] + list(it.get("alts") or []),
                  "hard": it["src"] == "hard"} for i, (it, x) in enumerate(zip(items, x16))]
        return {"items": items, "outs": outs, "x16": x16, "clips": clips, "rid": box["sc"].submit({"clips": clips}), "t_gen": time.time() - tp}

    def judge_terms(b, res):
        if not box.get("jg"):
            return [None] * len(res)
        req = [{"x16": x, "text": it["text"], "plain": it["plain"], "asr": s["asr_q"], "direction": it.get("direction"),
                "emotion": it.get("emotion"), "intensity": it.get("intensity"), "ref": cl["ref"]}
               for it, x, s, cl in zip(b["items"], b["x16"], res, b["clips"])]
        return box["jg"]({"clips": req})

    def collect(b):
        try:
            res = box["sc"].result(b["rid"])
        except Exception as e:  # noqa: BLE001  infra failure: restart the scorer if needed, re-score once, else drop the batch
            print("SCORE FAILED", repr(e)[:300], flush=True)
            try:
                if not box["sc"].alive():
                    box["sc"] = Client(*SCORER)
                res = box["sc"]({"clips": b["clips"]})
            except Exception as e2:  # noqa: BLE001
                print("SCORE FAILED AGAIN, batch dropped", repr(e2)[:300], flush=True)
                return None, None
        return res, judge_terms(b, res)

    th.join()
    if "sc" not in box:
        raise RuntimeError("scorer did not start")
    evaluate(0)
    pending, last, stop = produce(1), 0, ""
    for step in range(1, c["steps"] + 1):
        b, pending = pending, None
        if step < c["steps"] and usd() < c["budget_usd"] - c["reserve_usd"] and total_usd() < c["total_cap_usd"] - c["reserve_usd"]:
            pending = produce(step + 1)  # sampled with the pre-update policy, scored while we wait for b
        tw = time.time()
        res, jt = collect(b)
        t_wait = time.time() - tw
        if res is None:
            if pending is None:
                break
            continue
        items, outs, G = b["items"], b["outs"], c["group"]
        durs = [len(x) / 16000 for x in b["x16"]]
        R, rows = np.zeros(len(items)), []
        for gi in range(len(items) // G):
            sl = range(gi * G, (gi + 1) * G)
            ok = [durs[i] for i in sl if not outs[i]["forced"]]
            gmed = float(np.median(ok)) if ok else 1e9
            for i in sl:
                R[i], fl = reward(br, items[i], res[i], durs[i], outs[i]["forced"], gmed, c["reward"], jt[i], c["judge_w"])
                rows.append({**fl, "R": round(float(R[i]), 4), "dur": round(durs[i], 2), "frames": outs[i]["T"], "group": gi, "judge": jt[i]})
        Rg = R.reshape(-1, G)
        A = (Rg - Rg.mean(1, keepdims=True)).reshape(-1)  # Dr.GRPO: no std normalisation
        live = np.repeat(Rg.std(1) > 1e-6, G)
        use = [i for i in range(len(items)) if live[i] and outs[i]["T"] >= 2]
        tu = time.time()
        m.train()
        opt.zero_grad(set_to_none=True)
        C = float(len(items) * NCB * 250)
        kls, ntok = [], 0
        for mb in microbatches(use, outs):
            ss = [outs[i] for i in mb]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rl = tf_logp(refm, refm.m.talker, ss)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pl = tf_logp(pol, m.talker, ss)
            loss = 0.0
            for i, lp, rlp in zip(mb, pl, rl):
                dlt = rlp - lp
                k3 = torch.exp(dlt) - dlt - 1
                loss = loss + (-float(A[i]) * lp.sum() + c["beta"] * k3.sum()) / C
                kls.append(float(k3.mean()))
                ntok += lp.numel()
            loss.backward()
            del pl, rl, loss
        gn = float(torch.nn.utils.clip_grad_norm_(params, c["clip"])) if use else 0.0
        if use and np.isfinite(gn):
            opt.step()
        sync()
        last = step
        mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
        bysrc = collections.defaultdict(list)
        for it, r in zip(items, rows):
            bysrc[it["src"]].append(r)
        met = {"step": step, "t": round(time.time() - t0), "R": mean(R), "R_group_sd": mean(Rg.std(1)), "groups_live": int(live.sum() // G),
               "wer": mean([r["wer"] for r in rows]), "ecapa": mean([s["ecapa"] for s in res]), "wavlm": mean([s["wavlm"] for s in res]),
               "dnsmos": mean([s.get("dnsmos_ovrl") or 0 for s in res]), "tail_ok": mean([s["tail_ok"] for s in res]),
               **{k: mean([r[k] for r in rows]) for k in ("bad", "forced", "runaway", "repeat")}, "early": mean([r["dur_pen"] > 0 for r in rows]),
               "dur": mean(durs), "by_src": {k: {"wer": mean([r["wer"] for r in v]), "bad": mean([r["bad"] for r in v]), "R": mean([r["R"] for r in v])}
                                             for k, v in bysrc.items()},
               "kl": mean(kls), "grad_norm": round(gn, 4), "n_upd": len(use), "tokens": ntok, "t_gen": round(b["t_gen"], 1),
               "t_wait": round(t_wait, 1), "t_upd": round(time.time() - tu, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
               "usd": round(usd(), 3), "usd_total": round(total_usd(), 3)}
        append(f"{out}/metrics.jsonl", [met])
        append(f"{out}/rollouts.jsonl", [{"step": step, "i": i, "src": it["src"], "id": it.get("id"), "text": it["text"], "direction": it.get("direction"),
                                          "voice": it["voice"]["voice_id"], **r, **{k: v for k, v in s.items() if k not in ("id", "ast")},
                                          "A": round(float(A[i]), 4)} for i, (it, r, s) in enumerate(zip(items, rows, res))])
        if step % 5 == 0 or step == 1 or smoke:  # best and worst take of two groups
            sd_ = f"{out}/samples/s{step:04d}"
            os.makedirs(sd_, exist_ok=True)
            for gi in range(min(2, len(items) // G)):
                g = list(range(gi * G, (gi + 1) * G))
                for tag, i in (("best", max(g, key=lambda i: R[i])), ("worst", min(g, key=lambda i: R[i]))):
                    sf.write(f"{sd_}/g{gi}_{tag}_{items[i]['src']}.wav", b["x16"][i], 16000, subtype="PCM_16")
        vol.commit()
        print("STEP", json.dumps(met), flush=True)
        if step % c["eval_every"] == 0 or pending is None:
            save(step)
            evaluate(step)
        if pending is None:
            stop = "steps done" if step >= c["steps"] else f"budget (run ${usd():.2f}, total ${total_usd():.2f})"
            break
    spend[run] = round(usd(), 3)
    json.dump({"run": run, "gpu": GPU, "usd": round(usd(), 3), "hours": round((time.time() - t0) / 3600, 3), "last_step": last, "stop": stop},
              open(f"{out}/spend.json", "w"), indent=1)
    vol.commit()
    print(f"TRAIN DONE step {last} ({stop}) in {time.time() - t0:.0f}s, ${usd():.2f} on {GPU}", flush=True)


@app.local_entrypoint()
def main(config: str, run: str, init: str, ref: str = "", inputs: str = "/vol/inputs", smoke: bool = False):
    train.remote(json.load(open(config)), run, init, ref, inputs, "/vol/results/grpo", smoke)
