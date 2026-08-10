#!/bin/bash
# 《天不再借》成片合成
#
#   ./build.sh 你的歌.mp3        把音频和水墨动画合成为 天不再借_成片.mp4
#   ./build.sh 你的歌.mp3 --redo 先重新渲染画面（改过 timeline.json 之后用这个）
#
set -euo pipefail
cd "$(dirname "$0")"

AUDIO="${1:-}"
VIDEO="天不再借_画面.mp4"
OUT="天不再借_成片.mp4"

if [ -z "$AUDIO" ]; then
  echo "用法: ./build.sh <人声音频文件> [--redo]"
  echo "例:   ./build.sh 天不再借.mp3"
  exit 1
fi
if [ ! -f "$AUDIO" ]; then
  echo "找不到音频文件: $AUDIO"; exit 1
fi

if [ "${2:-}" = "--redo" ] || [ ! -f "$VIDEO" ]; then
  echo "▸ 渲染画面（按 timeline.json）…"
  .venv/bin/python render.py
fi

# 以音频时长为准：画面不够就定格最后一帧补足，画面多了就裁掉
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
echo "▸ 音频时长 ${DUR}s，合成中…"

ffmpeg -y -loglevel error \
  -i "$VIDEO" -i "$AUDIO" \
  -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=600[v]" \
  -map "[v]" -map 1:a \
  -t "$DUR" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart \
  "$OUT"

echo "▸ 完成: $OUT"
ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name -of default=nw=1 "$OUT"
