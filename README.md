# 天不再借

**试听页：** https://chloe4ai.github.io/tianbuzaijie/

---

## 内容

| 文件 | 说明 |
|---|---|
| `index.html` | 试听页（歌词 / 编曲 / 三版音频 / 剧照） |
| `歌词与编曲.md` | 完整歌词、押韵说明、曲式与配器方案 |
| `歌词.srt` | 带时间轴的字幕 |
| `audio/v1–v3.mp3` | 三个演唱版本，同词同曲、不同随机种子 |
| `img/` | 水墨画面关键帧 |
| `render.py` / `build.sh` / `timeline.json` | 画面渲染与音画合成 |

## 规格

- g 小调，74 BPM，4/4，3′50″
- 男中音，主歌 G2–D4 胸腔实声，副歌冲至 G4–A4
- 副歌第二遍升全音至 a 小调

## 音频怎么来的

三版均由 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)（MIT）在本地 Apple Silicon 上生成，
未使用任何在线音乐服务。词、编曲方案、画面为原创。

三版响度对比（主歌 → 末段副歌）：

| 版本 | 主歌 | 末段副歌 | 落差 |
|---|---|---|---|
| v1 | −19.0 dB | −14.7 dB | 4.3 dB |
| v2 | −21.1 dB | −14.0 dB | 7.1 dB |
| v3 | −23.2 dB | −14.9 dB | 8.3 dB |

## 合成成片

```bash
./build.sh audio/v3.mp3
```

需要 `ffmpeg`，以及未收录进本仓库的 `天不再借_画面.mp4`（265 MB，超过 GitHub 单文件上限）。
