#!/usr/bin/env python3
"""Align the known 30 lyric lines to each take, using global DP alignment.

Whisper mis-hears sung Chinese, so we ignore WHAT it heard and keep only WHEN.
The full known lyric character sequence is globally aligned (Needleman-Wunsch)
against the full heard character sequence; each line's start time = timestamp of
its first character that landed on a real match.

Lines with no anchor are interpolated between their neighbours and flagged
low-confidence, so the page can choose not to display them.
"""
import json
import re

LINES = [
    ("主歌一", "踏着高墙一重一重冰冷的雕栏", "踏着高墙　一重一重　冰冷的雕栏"),
    ("主歌一", "走过朱门走过深宫走过万人间", "走过朱门　走过深宫　走过万人间"),
    ("主歌一", "以为把名字写进了日月和山川", "以为把名字写进了日月和山川"),
    ("主歌一", "天光就该只照我这边", "天光就该只照我这边"),
    ("主歌二", "数着更鼓一声一声敲碎的夜寒", "数着更鼓　一声一声　敲碎的夜寒"),
    ("主歌二", "山河不语风雷不语谁在替我瞒", "山河不语　风雷不语　谁在替我瞒"),
    ("主歌二", "史册翻页从来不问功过与恩怨", "史册翻页从来不问功过与恩怨"),
    ("主歌二", "只把王侯写成一缕烟", "只把王侯写成一缕烟"),
    ("预副歌", "有心锁住万里江山", "有心锁住万里江山"),
    ("预副歌", "无奈天意从来不由人算", "无奈天意从来不由人算"),
    ("副歌", "大厦将倾风未起满城灯火先寒", "大厦将倾　风未起　满城灯火先寒"),
    ("副歌", "我真的还想再借五百年", "我真的还想再借五百年"),
    ("副歌", "天不再借不再借借条烧成了烟", "天不再借　不再借　借条烧成了烟"),
    ("副歌", "九重宫阙一夜换了天", "九重宫阙　一夜换了天"),
    ("桥段", "谁在殿外磨亮了近身的那把刀剑", "谁在殿外磨亮了近身的那把刀剑"),
    ("桥段", "谁把忠诚兑成活着兑成一句谎言", "谁把忠诚兑成活着　兑成一句谎言"),
    ("桥段", "最后的敌人从来不在那城墙外面", "最后的敌人从来不在那城墙外面"),
    ("桥段", "他就站在我的影子里边", "他就站在我的影子里边"),
    ("预副歌", "求过神明拜过青天", "求过神明　拜过青天"),
    ("预副歌", "换不来龙椅底下多半年", "换不来龙椅底下　多半年"),
    ("副歌", "大厦将倾风未起满城灯火先寒", "大厦将倾　风未起　满城灯火先寒"),
    ("副歌", "我真的还想再借五百年", "我真的还想再借五百年"),
    ("副歌", "天不再借不再借借条烧成了烟", "天不再借　不再借　借条烧成了烟"),
    ("副歌", "九重宫阙一夜换了天", "九重宫阙　一夜换了天"),
    ("副歌", "大厦将倾风未起满城灯火先寒", "大厦将倾　风未起　满城灯火先寒"),
    ("副歌", "我真的还想再借五百年", "我真的还想再借五百年"),
    ("副歌", "天不再借不再借借条烧成了烟", "天不再借　不再借　借条烧成了烟"),
    ("副歌", "九重宫阙一夜换了天", "九重宫阙　一夜换了天"),
    ("尾声", "天不再借借期已满", "天不再借　借期已满"),
    ("尾声", "大江东去涛声依旧在人间", "大江东去　涛声依旧在人间"),
]

CJK = re.compile(r"[一-鿿]")
MATCH, MIS, GAP = 3, -2, -1


def char_stream(tr):
    """[(char, start_time)] spread evenly across each word/segment."""
    out = []
    for seg in tr["segments"]:
        units = seg.get("words") or []
        if units:
            for w in units:
                chars = CJK.findall(w["w"])
                if chars:
                    span = max(w["e"] - w["s"], 0.01) / len(chars)
                    for i, c in enumerate(chars):
                        out.append((c, w["s"] + i * span))
        else:
            chars = CJK.findall(seg["text"])
            if chars:
                span = max(seg["end"] - seg["start"], 0.01) / len(chars)
                for i, c in enumerate(chars):
                    out.append((c, seg["start"] + i * span))
    return out


def nw_align(a, b):
    """Global alignment. Returns list of (i,j) for aligned (matched) pairs."""
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * GAP
    for j in range(1, m + 1):
        score[0][j] = j * GAP
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = score[i], score[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1] + (MATCH if ai == b[j - 1] else MIS)
            row[j] = d if d >= row[j - 1] + GAP else row[j - 1] + GAP
            up = prev[j] + GAP
            if up > row[j]:
                row[j] = up
    # traceback
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j - 1] + (MATCH if a[i - 1] == b[j - 1] else MIS):
            if a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def main():
    report = {}
    try:
        probe = json.load(open("verse1_probe.json", encoding="utf-8"))
    except FileNotFoundError:
        probe = {}

    # Lines the take demonstrably never sings (verified by vocal-band energy:
    # v2 131.5-151.5s matches the instrumental profile, not the vocal one).
    NOT_SUNG = {"v2": {14, 15, 16, 17}}

    for v in ("v1", "v2", "v3"):
        tr = json.load(open(f"transcript_{v}.json", encoding="utf-8"))
        if v in probe:
            # targeted passes win inside the windows they cover
            pr = [s for s in probe[v] if "Zither" not in s["text"] and s["text"].strip()]
            keep = [s for s in tr["segments"]
                    if not any(s["start"] < p["end"] and p["start"] < s["end"] for p in pr)]
            tr = {"segments": sorted(keep + pr, key=lambda s: s["start"])}
        stream = char_stream(tr)
        heard = "".join(c for c, _ in stream)
        times = [t for _, t in stream]

        known = "".join(l[1] for l in LINES)
        owner, off = [], 0
        for idx, l in enumerate(LINES):
            owner += [idx] * len(l[1])

        pairs = nw_align(known, heard)

        # per line: matched chars and the earliest matched time
        hits = {i: [] for i in range(len(LINES))}
        for ki, hi in pairs:
            hits[owner[ki]].append((ki, times[hi]))

        # median per-character duration, for extrapolating unmatched line heads
        allt = sorted(t for h in hits.values() for _, t in h)
        deltas = [b - a for a, b in zip(allt, allt[1:]) if 0 < b - a < 1.5]
        char_dur = sorted(deltas)[len(deltas) // 2] if deltas else 0.45

        line_start = {}
        for idx, (_, plain, _) in enumerate(LINES):
            offs = 0
            for j in range(idx):
                offs += len(LINES[j][1])
            line_start[idx] = offs

        cues = []
        for i, (label, plain, disp) in enumerate(LINES):
            h = hits[i]
            n_match = len(h)
            conf = round(n_match / len(plain), 2)
            t = None
            if h:
                # positions within this line, and their times
                pos = [ki - line_start[i] for ki, _ in h]
                ts = [tt for _, tt in h]
                if len(h) >= 2:
                    # least squares: time = a + b*pos  ->  onset = a (pos 0)
                    n = len(pos)
                    mx = sum(pos) / n
                    my = sum(ts) / n
                    den = sum((p - mx) ** 2 for p in pos)
                    b = (sum((p - mx) * (q - my) for p, q in zip(pos, ts)) / den) if den else char_dur
                    b = min(max(b, 0.15), 1.2)
                    t = my - b * mx
                else:
                    t = ts[0] - pos[0] * char_dur
                t = round(max(0.0, t), 2)
            cues.append({"i": i, "label": label, "text": disp,
                         "t": t, "conf": conf, "matched": n_match,
                         "len": len(plain), "src": "align" if t else "gap"})

        # reject anchors too weak to trust (noise matching a stray character)
        for c in cues:
            if c["t"] is not None and c["conf"] < 0.25:
                c["t"], c["src"] = None, "weak"

        # enforce monotonic order; drop out-of-order anchors as unreliable
        last = -1.0
        for c in cues:
            if c["t"] is not None and c["t"] <= last:
                c["t"], c["src"], c["conf"] = None, "nonmono", 0.0
            elif c["t"] is not None:
                last = c["t"]

        # interpolate gaps between known anchors
        anchors = [(c["i"], c["t"]) for c in cues if c["t"] is not None]
        for c in cues:
            if c["t"] is not None:
                continue
            prev = [a for a in anchors if a[0] < c["i"]]
            nxt = [a for a in anchors if a[0] > c["i"]]
            if prev and nxt:
                (i0, t0), (i1, t1) = prev[-1], nxt[0]
                c["t"] = round(t0 + (t1 - t0) * (c["i"] - i0) / (i1 - i0), 2)
                c["src"] = "interp"
            elif nxt:
                i1, t1 = nxt[0]
                # local spacing measured from the next two anchors
                step = ((nxt[1][1] - t1) / (nxt[1][0] - i1)) if len(nxt) > 1 else 6.0
                step = min(max(step, 3.0), 9.0)
                c["t"] = round(max(0.0, t1 - step * (i1 - c["i"])), 2)
                c["src"] = "extrap"
            elif prev:
                i0, t0 = prev[-1]
                step = ((t0 - prev[-2][1]) / (i0 - prev[-2][0])) if len(prev) > 1 else 6.0
                step = min(max(step, 3.0), 9.0)
                c["t"] = round(t0 + step * (c["i"] - i0), 2)
                c["src"] = "extrap"

        # duration = until next line starts (capped), so a line clears when the next begins
        END = 230.0
        # spread any trailing extrapolated lines over the remaining time
        tail = [c for c in cues if c["src"] == "extrap" and c["i"] > max(
            [x["i"] for x in cues if x["src"] == "align"] or [-1])]
        if tail:
            t0 = max((c["t"] for c in cues if c["src"] == "align"), default=0.0)
            step = (END - t0) / (len(tail) + 1)
            for k, c in enumerate(tail, start=1):
                c["t"] = round(t0 + step * k, 2)

        for c in cues:
            c["t"] = round(min(c["t"], END - 1.5), 2)

        # lines this take never sings are dropped, not guessed
        cues = [c for c in cues if c["i"] not in NOT_SUNG.get(v, set())]

        for k, c in enumerate(cues):
            nxt_t = cues[k + 1]["t"] if k + 1 < len(cues) else END
            c["d"] = round(min(max(nxt_t - c["t"], 1.5), 9.0), 2)

        solid = sum(1 for c in cues if c["src"] == "align")
        report[v] = {"heard_chars": len(stream), "anchored": solid,
                     "interp": [c["i"] for c in cues if c["src"] != "align"]}
        json.dump(cues, open(f"cues_{v}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

        print(f"--- {v}: 听到 {len(stream)} 字 | 实测锚定 {solid}/30 句 | "
              f"推算 {len(report[v]['interp'])} 句")
        for c in cues:
            mark = {"align": " ", "interp": "~", "extrap": "^", "gap": "!", "nonmono": "x"}[c["src"]]
            print(f" {mark}{c['i']:2d} {c['t']:7.2f} +{c['d']:4.1f}  "
                  f"conf={c['conf']:.2f} {c['matched']:2d}/{c['len']:2d}  {c['text'][:15]}")
    json.dump(report, open("align_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
