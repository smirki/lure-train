"""Shared config, Modal images, volume paths, jsonl helpers and the spend guard.

Everything that names infrastructure is configurable by environment variable at launch time:
  LURE_VOLUME      Modal volume mounted at /vol (default "lure-train")
  LURE_GPU         training GPU (default "RTX-PRO-6000")
  LURE_EVAL_GPU    eval / encode GPU (default "RTX-PRO-6000")
  LURE_JUDGES      "1" adds the judge venv (intensity / emotion / tags) to the GRPO image
Secrets are never read here. If a job needs a token (for example a gated Hugging Face repo), attach a Modal secret by name.
"""
import json
import os
import time

VOL = "/vol"
VOL_NAME = os.environ.get("LURE_VOLUME", "lure-train")
GPU = os.environ.get("LURE_GPU", "RTX-PRO-6000")
EVAL_GPU = os.environ.get("LURE_EVAL_GPU", "RTX-PRO-6000")
JUDGES = os.environ.get("LURE_JUDGES", "") == "1"
HF_SECRET = os.environ.get("LURE_HF_SECRET", "")  # name of a Modal secret holding HF_TOKEN, only if a gated repo is used

TTS = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CODES, NCB, FRAME_HZ, SR, SPF = 2048, 16, 12.5, 24000, 1920  # codebook size, codebooks, frames/s, codec rate, samples per frame
# Modal list price per hour: GPU + 8 cores + 64 GB (cores and memory are billed on top of the GPU)
GPU_PRICE = {"RTX-PRO-6000": 3.03, "H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80}


def price_h(gpu, cpu=8, mem_gb=64):
    return GPU_PRICE.get(gpu, 3.95) + cpu * 0.0473 + mem_gb * 0.008


def vpath(p):
    """Volume-relative path ("/data/x.wav" or "data/x.wav") -> container path under /vol. Absolute /vol paths pass through."""
    if p.startswith(VOL + "/"):
        return p
    return VOL + (p if p.startswith("/") else "/" + p)


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []


def append(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_rows(path, rows):
    """Atomic rewrite (one writer per file: shared appends from several containers lose rows on a volume)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(path + ".tmp", path)


def spend_guard(dict_name, launch, gpu, cap_usd, t0=None):
    """Per-launch cost from wall time, summed over every launch in a modal.Dict. Returns (usd(), total()); raises if the cap is
    already used up before starting."""
    import modal
    t0 = t0 or time.time()
    ph = price_h(gpu)
    spend = modal.Dict.from_name(dict_name, create_if_missing=True)

    def usd():
        return (time.time() - t0) / 3600 * ph

    def total():
        spend[launch] = round(usd(), 4)
        return sum(float(v) for v in spend.values())
    prior = sum(float(v) for k, v in spend.items() if k != launch)
    print(f"launch {launch} on {gpu}: prior spend ${prior:.2f} of cap ${cap_usd} ({dict_name})", flush=True)
    if prior > cap_usd - 0.3:
        raise SystemExit("budget cap reached before start")
    return usd, total


# ------------------------------------------------------------------------------------------------ Modal images (built lazily)
DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS"
QASR = "Qwen/Qwen3-ASR-1.7B-hf"
GRANITE = "ibm-granite/granite-speech-5.0-470m-turboctc"
AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
WAVLM_SV = "microsoft/wavlm-base-plus-sv"
SV_PKGS = ("torch==2.9.1 torchaudio==2.9.1 transformers==5.19.0 accelerate numpy scipy soundfile librosa jiwer onnxruntime "
           "'huggingface_hub[hf_xet]' speechbrain==1.1.1 num2words")
JUDGE_PKGS = ("torch==2.9.1 torchaudio==2.9.1 qwen-asr funasr modelscope numpy scipy soundfile soxr librosa pyloudnorm "
              "'huggingface_hub[hf_xet]'")


def images():
    """-> dict of Modal images: tts (qwen-tts, transformers 4.57), scorer (tts + /opt/sv venv with transformers 5 for Qwen3-ASR),
    grpo (scorer, plus the /opt/judges venv when LURE_JUDGES=1)."""
    import modal
    env = {"HF_HOME": "/opt/hf", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "LURE_GPU": GPU, "LURE_EVAL_GPU": EVAL_GPU, "LURE_VOLUME": VOL_NAME, "LURE_JUDGES": "1" if JUDGES else ""}
    tts = (modal.Image.debian_slim(python_version="3.11")
           .apt_install("git", "ffmpeg", "libsndfile1", "wget")
           .pip_install("torch==2.9.1", "torchaudio==2.9.1", "qwen-tts==0.1.1", "transformers==4.57.3", "numpy>=2.0", "soundfile>=0.13",
                        "librosa", "jiwer", "onnxruntime", "huggingface_hub[hf_xet]", "speechbrain==1.1.1", "pyloudnorm", "uv")
           .run_commands(f"mkdir -p /opt/dnsmos && wget -q -O /opt/dnsmos/sig_bak_ovr.onnx {DNSMOS_URL}/sig_bak_ovr.onnx "
                         f"&& wget -q -O /opt/dnsmos/model_v8.onnx {DNSMOS_URL}/model_v8.onnx",
                         "HF_HOME=/opt/hf python -c \"from huggingface_hub import snapshot_download as s; "
                         f"s('{TTS}'); s('speechbrain/spkrec-ecapa-voxceleb', local_dir='/opt/ecapa')\"")
           .env(env))
    scorer = tts.run_commands("uv venv /opt/sv --python 3.11", f"uv pip install --python /opt/sv/bin/python {SV_PKGS}",
                              "HF_HOME=/opt/hf /opt/sv/bin/python -c \"from huggingface_hub import snapshot_download as s; "
                              f"s('{QASR}'); s('{GRANITE}'); s('{AST}'); s('{WAVLM_SV}')\"")
    grpo = scorer
    if JUDGES:
        grpo = scorer.run_commands("uv venv /opt/judges --python 3.11", f"uv pip install --python /opt/judges/bin/python {JUDGE_PKGS}")
    src = lambda im: im.add_local_python_source("lure_train")  # noqa: E731  (local sources last: code edits do not rebuild layers)
    return {"tts": src(tts), "scorer": src(scorer), "grpo": src(grpo)}


def volume():
    import modal
    return modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def secrets():
    import modal
    return [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []
