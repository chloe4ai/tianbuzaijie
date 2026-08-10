#!/usr/bin/env python3
"""Transcribe each take with word-level timestamps -> transcript_vN.json"""
import json
import sys
import numpy as np
import soundfile as sf
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"


def load16k(path):
    """Decode via libsndfile (no ffmpeg) -> mono float32 @16kHz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr % 16000 == 0:  # 48000 -> exact 3x: box-filter then decimate (anti-alias)
        f = sr // 16000
        n = (len(mono) // f) * f
        mono = mono[:n].reshape(-1, f).mean(axis=1)
    else:
        idx = np.linspace(0, len(mono) - 1, int(len(mono) * 16000 / sr))
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


for v in ("v1", "v2", "v3"):
    path = f"audio/{v}.mp3"
    print(f"=== {path} ===", flush=True)
    audio = load16k(path)
    print(f"  decoded {len(audio)/16000:.1f}s @16kHz", flush=True)
    r = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language="zh",
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt="这是一首中文古风史诗歌曲，男中音演唱。",
    )
    segs = []
    for s in r.get("segments", []):
        segs.append({
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "text": s["text"],
            "words": [
                {"w": w.get("word", ""), "s": round(w.get("start", 0), 2), "e": round(w.get("end", 0), 2)}
                for w in (s.get("words") or [])
            ],
        })
    with open(f"transcript_{v}.json", "w", encoding="utf-8") as f:
        json.dump({"segments": segs}, f, ensure_ascii=False, indent=1)
    nw = sum(len(s["words"]) for s in segs)
    print(f"  {len(segs)} segments, {nw} words -> transcript_{v}.json", flush=True)

print("done")
