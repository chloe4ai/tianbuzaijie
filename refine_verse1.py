#!/usr/bin/env python3
"""Re-transcribe the 主歌一 window with the known lyrics as a prompt."""
import json
import numpy as np
import soundfile as sf
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
W0, W1 = 18.0, 70.0
PROMPT = ("踏着高墙一重一重冰冷的雕栏。走过朱门走过深宫走过万人间。"
          "以为把名字写进了日月和山川。天光就该只照我这边。")


def load16k(path, t0, t1):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)[int(t0 * sr):int(t1 * sr)]
    f = sr // 16000
    n = (len(mono) // f) * f
    return np.ascontiguousarray(mono[:n].reshape(-1, f).mean(axis=1), dtype=np.float32)


out = {}
for v in ("v1", "v2", "v3"):
    audio = load16k(f"audio/{v}.mp3", W0, W1)
    r = mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language="zh",
        word_timestamps=True, condition_on_previous_text=False,
        initial_prompt=PROMPT, temperature=0.0,
    )
    segs = [{"start": round(s["start"] + W0, 2), "end": round(s["end"] + W0, 2),
             "text": s["text"].strip(),
             "words": [{"w": w.get("word", ""), "s": round(w.get("start", 0) + W0, 2),
                        "e": round(w.get("end", 0) + W0, 2)} for w in (s.get("words") or [])]}
            for s in r.get("segments", [])]
    out[v] = segs
    print(f"=== {v} ===")
    for s in segs:
        print(f"  {s['start']:7.2f} - {s['end']:7.2f}  {s['text'][:40]}")

json.dump(out, open("verse1_probe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
