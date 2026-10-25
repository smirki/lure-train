"""Held-out eval shared by every stage: render a fixed eval set with fixed seeds, score with the scorer subprocess, summarise.

Eval set jsonl rows: {key, text, direction|null, plain, est_s, voice: {voice_id, ref_wav, ref_text}, tags?}. Same seeds every run
(batches of 16, seed 1000 + offset + 50000 * take), so numbers compare across checkpoints of the same GPU type.
Reported: WER Qwen3-ASR / Granite, ECAPA, WavLM-SV, DNSMOS OVRL, ended, hit_max, mean duration, AST event hit.

Standalone (any talker checkpoint, optionally with a fast CP head):
  modal run --detach -m lure_train.evaluate --talker /vol/ckpt/lure/talker.pt --eval-set /vol/inputs/eval.jsonl --out /vol/results/eval/lure
  modal run --detach -m lure_train.evaluate --talker <opd>/ckpt/talker_s0120.pt --head <opd>/ckpt/head_s0120.pt --out ...
"""
import json
import os
import time

import modal

from .common import EVAL_GPU, VOL, images, jl, price_h, vpath, volume, write_rows
from .model import max_frames, tags_of
from .rewards.reward import event_hits

SCORER = ("/opt/sv/bin/python", "lure_train.rewards.worker")


def scorer():
    from .ipc import Client
    return Client(*SCORER)


def to16(w):
    import librosa
    import numpy as np
    return librosa.resample(w, orig_sr=24000, target_sr=16000).astype(np.float32) if len(w) else np.zeros(1600, np.float32)


def load_eval(path):
    items = jl(vpath(path))
    for e in items:
        e.setdefault("tags", tags_of(e["text"]))
        e.setdefault("plain", e["text"])
    return items


def mean(v):
    import numpy as np
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def run_eval(gen, sc, items, out_dir, calib=None, takes=1):
    """gen: Student (CP patched or not); sc: scorer Client. Writes out_dir/<key>_t<k>.wav + eval.jsonl, returns the summary."""
    import soundfile as sf
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    gen.lp = None
    rows, clips = [], []
    for t in range(takes):
        for s in range(0, len(items), 16):
            ch = items[s:s + 16]
            mf = max_frames([e["est_s"] for e in ch])
            outs = gen.generate([{"text": e["text"], "direction": e.get("direction"), "voice": e["voice"]} for e in ch], mf, seed=1000 + s + 50000 * t)
            for e, o, w in zip(ch, outs, gen.decode(outs)):
                fn = f"{e['key']}_t{t}.wav"
                sf.write(f"{out_dir}/{fn}", w, 24000, subtype="PCM_16")
                rows.append({"key": e["key"], "take": t, "file": fn, "frames": o["T"], "ended": bool(o["ended"]), "hit_max": o["T"] >= mf,
                             "dur": round(len(w) / 24000, 3), "tags": e["tags"], "plain": e["plain"]})
                clips.append({"id": f"{e['key']}_t{t}", "x16": to16(w), "ref": vpath(e["voice"]["ref_wav"]), "refs": [e["plain"]], "ast": True})
    t_render = time.time() - t0
    res = [x for i in range(0, len(clips), 32) for x in sc({"clips": clips[i:i + 32]})]
    for r, s in zip(rows, res):
        r.update({k: v for k, v in s.items() if k != "id"})
        r["event_hits"] = event_hits(r.get("ast"), r["tags"], calib)
    write_rows(f"{out_dir}/eval.jsonl", rows)
    eh = [h for r in rows for h in r["event_hits"]]
    return {"n": len(rows), "wer_qwen": mean([r["wer_q"] for r in rows]), "wer_granite": mean([r["wer_g"] for r in rows]),
            "ecapa": mean([r["ecapa"] for r in rows]), "wavlm": mean([r["wavlm"] for r in rows]),
            "dnsmos": mean([r.get("dnsmos_ovrl") for r in rows]), "ended": mean([r["ended"] for r in rows]),
            "hit_max": sum(r["hit_max"] for r in rows), "dur": mean([r["dur"] for r in rows]),
            "event_hit": mean(eh), "n_event_checks": len(eh), "t_render": round(t_render), "t_eval": round(time.time() - t0)}


def load_calib(path):
    p = vpath(path) if path else ""
    return json.load(open(p)) if p and os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------ standalone Modal eval
app = modal.App("lure-eval")
vol = volume()


@app.function(image=images()["scorer"], gpu=EVAL_GPU, volumes={VOL: vol}, timeout=4 * 3600, cpu=8, memory=64 * 1024)
def eval_ckpt(talker: str, out: str, eval_set: str = "/vol/inputs/eval.jsonl", head: str = "", head_name: str = "V2s",
              calib: str = "/vol/inputs/events_calib.json", takes: int = 1, limit: int = 0):
    from .model import Student
    t0 = time.time()
    vol.reload()
    gen = Student("bfloat16")
    gen.load_talker(talker)
    gen.freeze()
    if head:
        from .heads.fastcp import CPPatch, load_head
        CPPatch(gen).set(load_head(gen, head_name, head))
    items = load_eval(eval_set)[: limit or None]
    sc = scorer()
    summ = run_eval(gen, sc, items, vpath(out), load_calib(calib), takes)
    summ.update(talker=talker, head=head or None, gpu=EVAL_GPU, usd=round((time.time() - t0) / 3600 * price_h(EVAL_GPU), 3))
    json.dump(summ, open(f"{vpath(out)}/summary.json", "w"), indent=1)
    vol.commit()
    print("EVAL", json.dumps(summ), flush=True)
    return summ


@app.local_entrypoint()
def main(talker: str, out: str, eval_set: str = "/vol/inputs/eval.jsonl", head: str = "", head_name: str = "V2s", takes: int = 1, limit: int = 0):
    eval_ckpt.remote(talker, out, eval_set, head, head_name, "/vol/inputs/events_calib.json", takes, limit)
