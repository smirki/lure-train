"""Reward functions for the specialist GRPO branches. Pure Python + numpy, no models.

Hygiene first (both branches): a take that hit its frame cap (no EOS), ran away (> 2x the group's median and over the generous cap)
or loops words gets a fixed low reward and is not scored further. There is no "shorter is better" term anywhere.
WER is the mean of Qwen3-ASR and Granite, each capped at 1, on the tag-stripped text.

  R  robustness / long-form: -WER, +ended properly, early-stop penalty, small DNSMOS term, ECAPA floor
  V  voice cloning: ECAPA and WavLM-SV similarity (each scaled by the start model's within-group spread), minus floors
Optional judge terms (intensity / emotion / tags) are added with weights from the branch config; all default to 0.
"""
import re

# The V scales are the within-group SD of each speaker score measured on the start checkpoint, so neither model dominates.
BRANCHES = {
    "R": dict(bad=-3.0, wer_w=2.0, tail=0.3, dur_w=1.5, dns_w=0.25, ecapa_floor=0.45, ecapa_w=4.0),
    "V": dict(bad=-4.0, se=0.05, sw=0.02, e0=0.6, w0=0.93, wer_floor=0.10, wer_w=10.0, dns_floor=3.0, dns_w=2.0, pen_cap=3.0, dur_w=1.5),
}


def n_words(s, alts=()):
    """Spoken word count without num2words: letter words + 2.5 per digit group; max with spoken renderings."""
    n = len(re.findall(r"[A-Za-z']+", s or "")) + 2.5 * len(re.findall(r"\d+", s or ""))
    return max([n] + [len(a.split()) for a in alts or ()])


def cap_frames(it, hz=12.5):
    """Generous per-take cap: 2.0 words/s + 1.2 s per tag + 5 s."""
    return int(hz * (n_words(it["plain"], it.get("alts")) / 2.0 + 1.2 * it.get("n_tags", 0) + 5))


def lo_dur(it):
    """Shortest plausible duration (4.2 words/s, 0.3 s per tag): below it the take must have dropped words."""
    return n_words(it["plain"], it.get("alts")) / 4.2 + 0.3 * it.get("n_tags", 0)


def reward(branch, it, m, dur, forced, gmed, cfg=None, judge=None, judge_w=None):
    """-> (reward, flags). m = scorer metrics, gmed = median duration of the group's non-forced takes, judge = judge terms for the
    take ({"intensity", "emotion", "tags", "gate_ok"}) or None, judge_w = weights for those terms."""
    c = {**BRANCHES[branch], **(cfg or {})}
    wer = 0.5 * (min(m["wer_q"], 1.0) + min(m["wer_g"], 1.0))
    hi = n_words(it["plain"], it.get("alts")) / 2.0 + 1.2 * it.get("n_tags", 0) + 3
    runaway = dur > 2.0 * gmed and dur > hi
    bad = forced or runaway or m["repeat"] or (judge is not None and not judge.get("gate_ok", True))
    dns = m.get("dnsmos_ovrl") or 1.0
    dur_pen = max(0.0, 1.0 - dur / max(lo_dur(it), 0.5))
    fl = {"wer": wer, "bad": bad, "forced": forced, "runaway": runaway, "repeat": m["repeat"], "dur_pen": dur_pen}
    if bad:
        return c["bad"], fl
    if branch == "R":
        r = (-c["wer_w"] * wer + c["tail"] * float(m["tail_ok"]) - c["dur_w"] * dur_pen + c["dns_w"] * min(max(dns - 3.0, -1.0), 0.6)
             - c["ecapa_w"] * max(0.0, c["ecapa_floor"] - m["ecapa"]))
    else:
        sim = 0.5 * (m["ecapa"] - c["e0"]) / c["se"] + 0.5 * (m["wavlm"] - c["w0"]) / c["sw"]
        pen = c["wer_w"] * max(0.0, wer - c["wer_floor"]) + c["dns_w"] * max(0.0, c["dns_floor"] - dns) + c["dur_w"] * dur_pen
        r = sim - min(pen, c["pen_cap"])
    if judge and judge_w:
        r += sum(w * judge[k] for k, w in judge_w.items() if judge.get(k) is not None)
    return float(r), fl


# ------------------------------------------------------------------------------------------------ AST event hit (eval metric)
FAMILIES = {"laugh": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"], "sigh": ["Sigh"],
            "breath": ["Breathing", "Gasp", "Pant", "Wheeze", "Sigh"], "gasp": ["Gasp"], "cough": ["Cough"], "throat": ["Throat clearing", "Cough"],
            "sniff": ["Sniff"], "snort": ["Snort"], "groan": ["Groan", "Grunt"], "cry": ["Crying, sobbing", "Whimper"], "hum": ["Humming"]}


def fam_score(ast, fam):
    return max(ast.get(lbl, 0.0) for lbl in FAMILIES[fam])


def event_hits(ast, tags, calib):
    """calib = {"thresholds": {family: thr}, "per_tag": {tag: {"family", "validated"}}} (thresholds = p95 of the family score on
    untagged control takes; only tags whose detector passed validation are checked). -> list of bools, one per checked tag."""
    if not calib or not ast:
        return []
    thr, pt = calib["thresholds"], calib["per_tag"]
    return [fam_score(ast, pt[e]["family"]) > thr[pt[e]["family"]] for e in set(tags)
            if (pt.get(e) or {}).get("validated") and e != "pause"]
