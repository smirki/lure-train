"""Judge server for GRPO specialist branches, run as `/opt/judges/bin/python -m lure_train.judges.worker` (LURE_JUDGES=1 image).

LURE_JUDGE_CFG (JSON, from the branch config's "judges" field); only the listed judges are loaded:
  {"intensity": {"ckpt": "/vol/judges/intensity/head_cal.pt", "thresholds": "/vol/judges/intensity/thresholds.json"},
   "emotion":   {"ckpt": "/vol/judges/emotion/head.pt"},
   "tags":      {"ckpt": "/vol/judges/tags/heads.pt", "policy": "/vol/judges/tags/policy.json", "spec_dir": null}}
Request {"clips": [{x16, text, plain, asr, direction, emotion, intensity (dial 0-3 or null), ref}]}
-> per clip {"intensity", "emotion", "tags", "gate_ok"}: reward terms (None when the judge is off or the prompt asks nothing of it)
and a hard gate (tag read aloud, clipping, too loud, near silent).
"""
import json
import os

from ..ipc import log, serve
from .intensity import HARD_GATES, preprocess


class Judges:
    def __init__(self):
        cfg = json.loads(os.environ.get("LURE_JUDGE_CFG") or "{}")
        self.ij = self.ej = self.tj = None
        self.refs = {}
        if cfg.get("intensity"):
            from .intensity import IntensityJudge
            self.ij = IntensityJudge(cfg["intensity"]["ckpt"], cfg["intensity"]["thresholds"])
        if cfg.get("emotion"):
            from .emotion import EmotionJudge
            self.ej = EmotionJudge(cfg["emotion"]["ckpt"])
        if cfg.get("tags"):
            from .tags import TagJudge
            t = cfg["tags"]
            self.tj = TagJudge(t["ckpt"], t["policy"], spec_dir=t.get("spec_dir"))
        log("judges loaded:", [k for k in ("intensity", "emotion", "tags") if cfg.get(k)])

    def ref16(self, p):
        if p not in self.refs:
            import librosa
            self.refs[p] = librosa.load(p, sr=16000, mono=True)[0]
        return self.refs[p]

    def score(self, req):
        from .emotion import emotion_reward, target_for
        clips = req["clips"]
        n = len(clips)
        out = [{"intensity": None, "emotion": None, "tags": None, "gate_ok": True} for _ in range(n)]
        E = None
        if self.ij is not None:
            items = [{"x16": c["x16"], "target": c.get("intensity"), "text": c["plain"], "ref16": self.ref16(c["ref"]) if c.get("ref") else None,
                      "steer": c.get("emotion")} for c in clips]
            res = self.ij.score(items)
            E = [r["expected"] for r in res]
            for o, r in zip(out, res):
                o["intensity"] = r.get("reward")
                o["gate_ok"] = not any(r.get(k) for k in HARD_GATES)
        if self.ej is not None:
            targets = [target_for(c) for c in clips]
            idx = [i for i, t in enumerate(targets) if t is not None]
            if idx:
                ans = self.ej.score([preprocess(clips[i]["x16"]) for i in idx], [E[i] for i in idx] if E else None)
                for i, a in zip(idx, ans):
                    out[i]["emotion"] = emotion_reward(a, targets[i])[0]
        if self.tj is not None:
            for o, r in zip(out, self.tj.score([{"x16": c["x16"], "text": c["text"], "asr": c["asr"]} for c in clips])):
                if r is not None:
                    o["tags"] = r["soft"]
                    o["gate_ok"] = o["gate_ok"] and r["gate_ok"]
        return out


if __name__ == "__main__":
    serve(Judges().score)
