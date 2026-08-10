#!/usr/bin/env python3
"""Targeted re-transcription of specific windows with the known lyrics as prompt.

usage: probe.py <out.json> <version:t0:t1:promptkey> ...
"""
import json
import sys
import numpy as np
import soundfile as sf
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"

PROMPTS = {
    "verse1": ("踏着高墙一重一重冰冷的雕栏。走过朱门走过深宫走过万人间。"
               "以为把名字写进了日月和山川。天光就该只照我这边。"),
    "bridge": ("谁在殿外磨亮了近身的那把刀剑。谁把忠诚兑成活着兑成一句谎言。"
               "最后的敌人从来不在那城墙外面。他就站在我的影子里边。"),
    "outro": "天不再借借期已满。大江东去涛声依旧在人间。",
}


def load16k(path, t0, t1):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)[int(t0 * sr):int(t1 * sr)]
    f = sr // 16000
    n = (len(mono) // f) * f
    return np.ascontiguousarray(mono[:n].reshape(-1, f).mean(axis=1), dtype=np.float32)


out = {}
for spec in sys.argv[2:]:
    v, t0, t1, key = spec.split(":")
    t0, t1 = float(t0), float(t1)
    audio = load16k(f"audio/{v}.mp3", t0, t1)
    r = mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language="zh", word_timestamps=True,
        condition_on_previous_text=False, initial_prompt=PROMPTS[key], temperature=0.0,
    )
    segs = [{"start": round(s["start"] + t0, 2), "end": round(s["end"] + t0, 2),
             "text": s["text"].strip(),
             "words": [{"w": w.get("word", ""), "s": round(w.get("start", 0) + t0, 2),
                        "e": round(w.get("end", 0) + t0, 2)} for w in (s.get("words") or [])]}
            for s in r.get("segments", [])]
    segs = [s for s in segs if "Zither" not in s["text"] and s["text"]]
    out.setdefault(v, []).extend(segs)
    print(f"=== {v} [{t0:.0f}-{t1:.0f}] {key} ===")
    for s in segs:
        print(f"  {s['start']:7.2f} - {s['end']:7.2f}  {s['text'][:42]}")

prev = {}
try:
    prev = json.load(open(sys.argv[1], encoding="utf-8"))
except FileNotFoundError:
    pass
for k, v in out.items():
    prev.setdefault(k, [])
    prev[k] = sorted(prev[k] + v, key=lambda s: s["start"])
json.dump(prev, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
