"""Encode a manifest's audio to Qwen3-TTS 12 Hz codes: {id: int16 (T, 16)} in <codes_dir>/<shard>.pt. Fan-out over shards.

  modal run --detach -m lure_train.sft.encode --manifest /vol/data/sft/manifest.jsonl --codes /vol/data/sft/codes --shards 8
Resumable: shards whose file exists are skipped. max_containers caps the fan-out.
"""
import os
import time

import modal

from ..common import EVAL_GPU, SR, TTS, VOL, images, vpath, volume

app = modal.App("lure-sft-encode")
vol = volume()
MAX_CONTAINERS = int(os.environ.get("LURE_MAX_CONTAINERS", "4"))


@app.function(image=images()["tts"], gpu=EVAL_GPU, volumes={VOL: vol}, timeout=4 * 3600, cpu=8, memory=32768, max_containers=MAX_CONTAINERS)
def encode(manifest: str, codes: str, shard: int, nshards: int):
    import librosa
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from qwen_tts import Qwen3TTSModel
    from .data import load_manifest, sha
    vol.reload()
    out_p = f"{vpath(codes)}/{shard:03d}.pt"
    if os.path.exists(out_p):
        return 0
    t0 = time.time()
    rows = [r for r in load_manifest(manifest) if int(sha(r["id"]), 16) % nshards == shard]
    tok = Qwen3TTSModel.from_pretrained(TTS, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa").model.speech_tokenizer
    pool, out = ThreadPoolExecutor(16), {}
    ld = lambda r: librosa.load(vpath(r["audio"]), sr=SR, mono=True)[0].astype(np.float32)  # noqa: E731
    for i in range(0, len(rows), 64):
        ch = rows[i:i + 64]
        for r, x in zip(ch, pool.map(ld, ch)):
            with torch.inference_mode():
                out[r["id"]] = tok.encode(x, sr=SR).audio_codes[0].to(torch.int16).cpu().clone()
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    torch.save(out, out_p + ".tmp")
    os.replace(out_p + ".tmp", out_p)
    vol.commit()
    print(f"shard {shard}: {len(out)} clips in {time.time() - t0:.0f}s on {EVAL_GPU}", flush=True)
    return len(out)


@app.local_entrypoint()
def main(manifest: str, codes: str, shards: int = 8):
    n = sum(encode.starmap([(manifest, codes, s, shards) for s in range(shards)]))
    print(f"encoded {n} clips into {codes}")
