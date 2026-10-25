# lure-train

Training code for Lure and Lure Turbo, the expressive TTS models built on Qwen3-TTS-12Hz-0.6B-Base. All training and eval jobs run on Modal.

The pipeline:

1. SFT on any corpus written as a jsonl manifest (text with inline tags plus an audio path).
2. Specialist GRPO branches. Each branch starts from the SFT model and is trained with Dr.GRPO toward one goal (voice fidelity, robustness, and later events, emotion and intensity). Each branch becomes a teacher.
3. Distillation. The teachers are distilled back into one model with on-policy distillation (OPD). The task-vector merge in `merge.py` is the simpler baseline.
4. Lure Turbo. A fast code-predictor head (V2s) replaces the 15-pass code predictor. It is warmed up against the stock predictor, then trained with OPD against the full model. OPD also trains the talker.

## Layout

```
lure_train/
  common.py        env config, Modal images, volume helpers, spend guard
  model.py         Qwen3-TTS wrapper: exact inference prompts, sampling, teacher-forced logits, decoding
  ipc.py           subprocess client/server (scorer and judges run in their own venvs)
  evaluate.py      fixed-seed held-out eval (WER Qwen3-ASR + Granite, ECAPA, WavLM-SV, DNSMOS, AST event hit)
  merge.py         task-vector merge of branches (baseline)
  sft/             data.py (generic manifest), encode.py (codec codes, fan-out), train.py
  grpo/            train.py (one trainer, a branch per config), prompts.py (hard-text and long-form prompt builder)
  rewards/         reward.py (branch rewards, pure Python), worker.py (scorer server)
  judges/          intensity.py, emotion.py, tags.py (inference + reward policy), worker.py (judge server)
  heads/           fastcp.py (V2 / V2s heads), train.py (head warm-up)
  opd/             train.py (single- or multi-teacher OPD)
configs/           branch, OPD and example configs; manifest and calibration examples
```

## Setup

```
pip install -e .
modal setup
```

Torch, qwen-tts, transformers, qwen-asr and funasr are installed only inside the Modal images in `lure_train/common.py`. Infrastructure is set with environment variables at launch time:

| variable | default | meaning |
|---|---|---|
| `LURE_VOLUME` | `lure-train` | Modal volume mounted at `/vol` |
| `LURE_GPU` | `RTX-PRO-6000` | training GPU |
| `LURE_EVAL_GPU` | `RTX-PRO-6000` | eval and encode GPU |
| `LURE_JUDGES` | unset | `1` adds the judge venv to the GRPO image |
| `LURE_HF_SECRET` | unset | name of a Modal secret holding `HF_TOKEN`, only if you use a gated repo |
| `LURE_MAX_CONTAINERS` | `4` | fan-out cap for `sft/encode.py` |
| `LURE_GRPO_SPEND`, `LURE_OPD_SPEND`, `LURE_HEADS_SPEND` | `lure-*-spend` | `modal.Dict` names used by the spend guards |

The code holds no tokens or keys. Paths are volume paths (`/vol/...` or volume-relative).

Conventions: always launch with `modal run --detach`. Every job writes its results to the volume, fan-outs set `max_containers`, and every trainer has a per-launch budget and a global spend cap. If you add an Audex-based judge, run it in bf16 only.

## 1. SFT

Write a manifest, one row per clip:

```json
{"audio": "data/sft/clips/chan_42/0001.flac", "text": "Okay (laugh) you got me.", "speaker": "chan_42", "direction": "warm, amused", "src": "yodas"}
```

`ref_audio`/`ref_text` are optional. A row without them clones from another clip by the same `speaker`. A row with neither field is dropped, because cloning a clip from itself teaches the model to copy the reference. `split` is `train` (default) or `heldout`. `src` labels the source for mixing weights. See `configs/manifest.example.jsonl` and the docstring in `sft/data.py`.

```
modal run --detach -m lure_train.sft.encode --manifest /vol/data/sft/manifest.jsonl --codes /vol/data/sft/codes --shards 8
modal run --detach -m lure_train.sft.train --run sft1 --manifest /vol/data/sft/manifest.jsonl --codes /vol/data/sft/codes \
    --eval-set /vol/inputs/eval.jsonl --mix '{"yodas": 0.75, "other": 0.25}'
```

The loss is cross-entropy on codebook 0 (including EOS) plus 0.5 times the mean cross-entropy on codebooks 1-15. Prompt positions are not trained. Gradients are clipped at 3 times the running median gradient norm. A run resumes from `ckpt/state.pt` when relaunched.

## 2. Specialist GRPO branches

`grpo/train.py` trains one specialist teacher branch. A branch is a config, not a separate script:

- `configs/grpo_v.json`: V, voice fidelity. The reward is ECAPA plus WavLM-SV similarity, scaled by the start model's within-group spread, with WER, DNSMOS and early-stop floors. Lure is V step 150.
- `configs/grpo_r.json`: R, robustness and long-form. The reward is -2·WER, plus a bonus for ending properly, minus an early-stop penalty, plus a small DNSMOS term and an ECAPA floor. It uses hard text and 30-75 s pieces built with `grpo/prompts.py`.
- `configs/grpo_emotion.example.json` and `configs/grpo_events.example.json`: planned branches. They add judge terms (see below) through `judge_w`.

WER is the mean of Qwen3-ASR-1.7B and Granite Speech 5 (no Whisper), computed on the tag-stripped text. Bad takes (no EOS, runaway length, word loops) get a fixed low reward before anything else is scored.

The method is Dr.GRPO: group 8, 16 prompts per step, advantage = reward minus the group mean, a k3 KL to a frozen reference (beta 0.1), and the loss divided by a constant instead of the take length. The scorer runs in a subprocess on the same GPU and scores batch N while batch N+1 is generated.

```
modal run --detach -m lure_train.grpo.train --config configs/grpo_v.json --run v1 --init /vol/results/sft/sft1/ckpt/talker_step1000.pt
LURE_JUDGES=1 modal run --detach -m lure_train.grpo.train --config configs/grpo_emotion.example.json --run e1 --init ...
```

Read `rollouts.jsonl` before you change a reward.

## 3. Judges used as rewards

These are inference and reward-policy code only. They run in the `/opt/judges` venv behind `judges/worker.py`.

- Intensity (int_v2): a CORAL ordinal head on the frozen Qwen3-ASR audio encoder. The reward is -|expected level - requested dial level|, so it pays for hitting the requested level and nothing extra for overacting. Gates on the raw take (clipping, too loud, silence, F0 drift, speaking rate) floor the reward.
- Emotion (e1): heads on emotion2vec+ large features (ModelScope). The reward is a log-margin of the requested class over the runner-up, with neighbour credit. The intensity judge supplies the "emotional at all" signal.
- Tags (TagNet): WavLM-large with a BiGRU, frame-level family and sub-tag heads. Each requested tag is looked for in the window around its forced-aligned position (Qwen3-ForcedAligner-0.6B). Unrequested events cost reward. A tag read aloud is a hard gate.

## 4. Distillation: multi-teacher OPD

`opd/train.py` takes `teachers: {branch: talker checkpoint}`. Each manifest row carries a `branch` field, and only that branch's teacher scores the student's takes for that row. The student samples on-policy. Its loss at every sampled code is reverse KL plus 0.5 forward KL to the routed teacher, on codebook 0 and on codebooks 1-15. Offline clips add a forward-KL anchor. With one teacher this is the Lure Turbo recipe below.

```
modal run --detach -m lure_train.opd.train --config configs/opd_multi.example.json --run mt1 --smoke   # 3-step smoke first
```

Baseline to beat (`merge.py`, CPU): merged = base + Σ wᵢ·(branchᵢ - base).

```
modal run --detach -m lure_train.merge --base /vol/ckpt/lure/talker.pt \
    --branches '{"V": ".../v1/ckpt/talker_step0150.pt", "R": ".../r1/ckpt/talker_step0120.pt"}' --weights '{"V": 1, "R": 1}' \
    --out /vol/results/merge/vr11
```

Compare both with `lure_train.evaluate` on the same eval set and GPU type.

## 5. Lure Turbo (V2s head + OPD)

The stock code predictor runs 15 sequential passes per 80 ms frame. V2 groups codebooks 1-3, 4-7 and 8-15 into 3 passes. V2s keeps those 3 passes and adds a small MLP that samples codebooks one at a time inside each group. The extra cost is a few small matmuls per codebook.

```
modal run --detach -m lure_train.heads.train --run head_v2 --name V2 --talker /vol/ckpt/lure/talker.pt \
    --manifest /vol/data/sft/manifest.jsonl --codes /vol/data/sft/codes
modal run --detach -m lure_train.heads.train --run head_v2s --name V2s --talker /vol/ckpt/lure/talker.pt \
    --manifest ... --codes ... --init-from /vol/results/heads/head_v2/V2.pt --trunk-lr-mult 0.3
modal run --detach -m lure_train.opd.train --config configs/opd_turbo.json --run turbo1
```

In OPD the student (talker at lr 2e-6, V2s head at lr 3e-5) samples on-policy. The frozen Lure model, with its stock code predictor, scores every sampled code. The run takes 120 steps of 16 on-policy and 8 anchor clips each. Evaluate the result with `lure_train.evaluate --talker <run>/ckpt/talker_s0120.pt --head <run>/ckpt/head_s0120.pt`.

## Eval

```
modal run --detach -m lure_train.evaluate --talker /vol/ckpt/lure/talker.pt --eval-set /vol/inputs/eval.jsonl --out /vol/results/eval/lure
```

Eval rows: `{key, text, direction, plain, est_s, voice: {voice_id, ref_wav, ref_text}}`. Seeds are fixed, so numbers are comparable across checkpoints on the same GPU type.

## License

Apache-2.0. See `LICENSE`.
