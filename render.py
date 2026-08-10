#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《天不再借》水墨山河歌词动画渲染器

读 timeline.json，逐帧生成画面，用管道喂给 ffmpeg 编码成 MP4。
纯 numpy + Pillow 绘制，没有任何外部素材。

用法：
    .venv/bin/python render.py            # 输出无声动画 天不再借_画面.mp4
    .venv/bin/python render.py --preview   # 只渲染 12 秒抽样，用来快速看效果
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

FONT_BLACK = ("/System/Library/Fonts/Supplemental/Songti.ttc", 0)   # Songti SC Black
FONT_REG = ("/System/Library/Fonts/Supplemental/Songti.ttc", 6)     # Songti SC Regular
FONT_LIGHT = ("/System/Library/Fonts/Supplemental/Songti.ttc", 3)   # Songti SC Light


def font(spec, size):
    path, idx = spec
    return ImageFont.truetype(path, size, index=idx)


# ---------------------------------------------------------------- 场景配色
# sky_top / sky_bot 天空渐变；ink 山峦墨色；sun 日轮颜色（None 表示无）
# sun_y 日轮中心高度（0=顶 1=底）；sun_r 半径比例；mist 云雾浓度；text 歌词颜色
SCENES = {
    "intro":  dict(sky_top=(10, 14, 26),  sky_bot=(30, 38, 54),   ink=(4, 6, 12),
                   sun=(196, 208, 224), sun_y=0.20, sun_r=0.052, sun_glow=0.35,
                   mist=0.30, text=(226, 230, 238), ember=(150, 170, 200), ember_n=90, shaft=0.0),
    "dawn":   dict(sky_top=(28, 40, 60),  sky_bot=(148, 138, 126), ink=(16, 20, 30),
                   sun=(236, 200, 152), sun_y=0.72, sun_r=0.075, sun_glow=0.55,
                   mist=0.42, text=(238, 236, 228), ember=(210, 190, 160), ember_n=120, shaft=0.0),
    "dusk":   dict(sky_top=(14, 18, 30),  sky_bot=(88, 66, 62),   ink=(7, 9, 15),
                   sun=(158, 92, 72), sun_y=0.80, sun_r=0.068, sun_glow=0.45,
                   mist=0.58, text=(232, 226, 216), ember=(180, 130, 100), ember_n=160, shaft=0.0),
    "rise":   dict(sky_top=(38, 44, 66),  sky_bot=(198, 152, 104), ink=(20, 18, 26),
                   sun=(252, 224, 164), sun_y=0.66, sun_r=0.090, sun_glow=0.75,
                   mist=0.40, text=(255, 250, 238), ember=(240, 210, 160), ember_n=150, shaft=0.0),
    "blaze":  dict(sky_top=(64, 24, 26),  sky_bot=(222, 118, 58),  ink=(26, 10, 10),
                   sun=(255, 96, 52), sun_y=0.58, sun_r=0.140, sun_glow=1.00,
                   mist=0.32, text=(255, 246, 232), ember=(255, 170, 90), ember_n=260, shaft=0.0),
    "storm":  dict(sky_top=(9, 9, 13),    sky_bot=(66, 42, 38),   ink=(3, 4, 8),
                   sun=(120, 60, 44), sun_y=0.50, sun_r=0.100, sun_glow=0.25,
                   mist=0.82, text=(230, 222, 212), ember=(150, 100, 78), ember_n=320, shaft=0.0),
    "shadow": dict(sky_top=(5, 6, 9),     sky_bot=(26, 24, 28),   ink=(2, 2, 5),
                   sun=None, sun_y=0.40, sun_r=0.05, sun_glow=0.0,
                   mist=0.22, text=(222, 216, 210), ember=(90, 88, 96), ember_n=70, shaft=1.0),
    "blaze2": dict(sky_top=(88, 24, 24),  sky_bot=(255, 152, 72),  ink=(30, 10, 10),
                   sun=(255, 112, 58), sun_y=0.54, sun_r=0.175, sun_glow=1.25,
                   mist=0.28, text=(255, 250, 240), ember=(255, 190, 110), ember_n=340, shaft=0.0),
    "river":  dict(sky_top=(228, 224, 214), sky_bot=(250, 248, 242), ink=(96, 104, 116),
                   sun=(226, 214, 196), sun_y=0.30, sun_r=0.085, sun_glow=0.18,
                   mist=0.34, text=(26, 26, 30), ember=(150, 150, 152), ember_n=60, shaft=0.0),
}

SEAL_RED = (176, 42, 38)
XFADE = 2.6  # 场景交叉淡入秒数


# ---------------------------------------------------------------- 地形生成
def ridge(width, roughness, seed, octaves=9):
    """中点位移法生成一条山脊线，返回 [0,1] 的高度数组。"""
    rng = np.random.default_rng(seed)
    n = 2 ** octaves + 1
    h = np.zeros(n, dtype=np.float64)
    h[0] = rng.random()
    h[-1] = rng.random()
    step = n - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        idx = np.arange(half, n - 1, step)
        h[idx] = (h[idx - half] + h[idx + half]) / 2.0 + (rng.random(idx.size) - 0.5) * scale
        scale *= roughness
        step = half
    h -= h.min()
    if h.max() > 0:
        h /= h.max()
    # 重采样到需要的宽度（首尾接得上，方便无缝滚动）
    x = np.linspace(0, n - 1, width, endpoint=False)
    out = np.interp(x, np.arange(n), h)
    # 环形融合，消除接缝
    blend = min(width // 6, 400)
    w = np.linspace(0, 1, blend)
    out[:blend] = out[:blend] * w + out[-blend:][::-1] * (1 - w)
    return out


def value_noise(h, w, cells_y, cells_x, seed, octaves=4):
    """多倍频 value noise，返回 [0,1] 的 float32 (h,w)。"""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    cy, cx = cells_y, cells_x
    for _ in range(octaves):
        grid = rng.random((cy, cx)).astype(np.float32)
        # 横向环形平铺，保证滚动无缝
        img = Image.fromarray((grid * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)
        layer = np.asarray(img, dtype=np.float32) / 255.0
        acc += layer * amp
        total += amp
        amp *= 0.5
        cy, cx = cy * 2, cx * 2
    acc /= total
    return acc


# ---------------------------------------------------------------- 文字缓存
class TextSprite:
    """把一句歌词渲染成 alpha 图，连同它的墨晕一起缓存。"""

    def __init__(self, text, fnt, max_w):
        pad = 70
        tmp = Image.new("L", (10, 10))
        box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=fnt)
        w, h = box[2] - box[0], box[3] - box[1]
        img = Image.new("L", (w + pad * 2, h + pad * 2), 0)
        ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, font=fnt, fill=255)
        if img.width > max_w:  # 太长就整体缩一点，别撞边
            k = max_w / img.width
            img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
        self.alpha = np.asarray(img, dtype=np.float32) / 255.0
        self.bleed = np.asarray(img.filter(ImageFilter.GaussianBlur(9)),
                                dtype=np.float32) / 255.0
        self.h, self.w = self.alpha.shape


def vertical_sprite(text, fnt, gap=14):
    """竖排文字（右侧段落标签用）。"""
    sizes = []
    tmp = ImageDraw.Draw(Image.new("L", (10, 10)))
    for ch in text:
        b = tmp.textbbox((0, 0), ch, font=fnt)
        sizes.append((b[2] - b[0], b[3] - b[1], b[0], b[1]))
    w = max(s[0] for s in sizes)
    h = sum(s[1] for s in sizes) + gap * (len(text) - 1)
    img = Image.new("L", (w + 8, h + 8), 0)
    d = ImageDraw.Draw(img)
    y = 4
    for ch, (cw, ch_, ox, oy) in zip(text, sizes):
        d.text((4 + (w - cw) // 2 - ox, y - oy), ch, font=fnt, fill=255)
        y += ch_ + gap
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- 渲染器
class Renderer:
    def __init__(self, tl):
        self.W = tl["width"]
        self.H = tl["height"]
        self.fps = tl["fps"]
        self.duration = tl["duration"]
        self.tl = tl

        W, H = self.W, self.H
        self.yy = np.arange(H, dtype=np.float32)[:, None]
        self.xx = np.arange(W, dtype=np.float32)[None, :]

        # ---- 五层山峦（越远越淡越慢）
        span = W * 2
        self.layers = []
        specs = [
            # (基线y比例, 起伏幅度比例, 视差速度 px/s, 墨色不透明度, 粗糙度, seed)
            (0.615, 0.135, 3.5,  0.30, 0.58, 11),
            (0.665, 0.170, 6.5,  0.45, 0.55, 27),
            (0.735, 0.205, 11.0, 0.62, 0.53, 43),
            (0.830, 0.235, 18.0, 0.80, 0.50, 61),
            (0.945, 0.260, 30.0, 1.00, 0.47, 89),
        ]
        for base, amp, spd, op, rough, sd in specs:
            hmap = ridge(span, rough, sd)
            self.layers.append(dict(h=hmap, base=base * H, amp=amp * H,
                                    spd=spd, op=op, span=span))

        # ---- 云雾 / 纸纹 / 暗角 / 颗粒
        self.cloud = value_noise(H, W * 2, 3, 5, seed=7, octaves=5)
        self.cloud = np.clip((self.cloud - 0.38) * 2.1, 0, 1)
        self.cloud2 = value_noise(H, W * 2, 2, 3, seed=19, octaves=4)
        self.cloud2 = np.clip((self.cloud2 - 0.42) * 2.4, 0, 1)

        paper = value_noise(H, W, 90, 160, seed=5, octaves=3)
        fiber = value_noise(H, W, 200, 12, seed=13, octaves=2)
        self.paper = ((paper - 0.5) * 0.05 + (fiber - 0.5) * 0.035).astype(np.float32)

        nx = (self.xx / W - 0.5) * 2.0
        ny = (self.yy / H - 0.5) * 2.0
        r = np.sqrt(nx ** 2 * 0.85 + ny ** 2)
        self.vignette = np.clip(1.0 - np.clip(r - 0.55, 0, None) * 0.95, 0.30, 1.0).astype(np.float32)

        rng = np.random.default_rng(101)
        self.grain = [(rng.random((H, W)).astype(np.float32) - 0.5) * 0.030 for _ in range(8)]

        # ---- 飞灰粒子（用最大量预分配，按场景取前 n 个）
        self.PMAX = 360
        g = np.random.default_rng(202)
        self.p_x = g.random(self.PMAX).astype(np.float32) * W
        self.p_y = g.random(self.PMAX).astype(np.float32) * H
        self.p_vy = -(6 + g.random(self.PMAX).astype(np.float32) * 34)
        self.p_vx = (g.random(self.PMAX).astype(np.float32) - 0.35) * 16
        self.p_sz = (g.random(self.PMAX) < 0.22).astype(np.int32) + 1
        self.p_ph = g.random(self.PMAX).astype(np.float32) * 6.28
        self.p_a = 0.25 + g.random(self.PMAX).astype(np.float32) * 0.6

        # ---- 文字
        self.f_lyric = font(FONT_BLACK, 68)
        self.f_title = font(FONT_BLACK, 168)
        self.f_sub = font(FONT_LIGHT, 42)
        self.f_label = font(FONT_REG, 30)
        self.f_seal = font(FONT_BLACK, 40)

        maxw = int(W * 0.86)
        self.sprites = [TextSprite(c["text"], self.f_lyric, maxw) for c in tl["cues"]]
        t = tl["title"]
        self.sp_title = TextSprite(t["main"], self.f_title, maxw)
        self.sp_sub = TextSprite(t["sub"], self.f_sub, maxw)
        self.labels = {}
        for c in tl["cues"]:
            if c["label"] not in self.labels:
                self.labels[c["label"]] = vertical_sprite(c["label"], self.f_label)
        self.seal = self._make_seal()

        # ---- 场景时间轴
        self.sections = sorted(tl["sections"], key=lambda s: s["t"])

    def _make_seal(self):
        """右下角朱文印：终局（红框 + 红字，框内留白）"""
        s = 126
        img = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([2, 2, s - 3, s - 3], radius=6, outline=255, width=7)
        f = font(FONT_BLACK, 44)
        y = 14
        for ch in "终局":
            b = d.textbbox((0, 0), ch, font=f)
            cw, chh = b[2] - b[0], b[3] - b[1]
            d.text((s // 2 - cw // 2 - b[0], y - b[1]), ch, font=f, fill=255)
            y += chh + 9
        arr = np.asarray(img, np.float32) / 255.0
        # 做旧：随机剥蚀
        g = np.random.default_rng(33)
        arr *= (g.random(arr.shape) > 0.13).astype(np.float32)
        arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
        return arr

    # ------------------------------------------------------------ 配色插值
    def palette(self, t):
        cur = self.sections[0]
        nxt = None
        for i, s in enumerate(self.sections):
            if t >= s["t"]:
                cur = s
                nxt = self.sections[i + 1] if i + 1 < len(self.sections) else None
        a = SCENES[cur["scene"]]
        if nxt and t >= nxt["t"] - XFADE:
            b = SCENES[nxt["scene"]]
            k = (t - (nxt["t"] - XFADE)) / XFADE
            k = k * k * (3 - 2 * k)
            return self._mix(a, b, k)
        return dict(a)

    @staticmethod
    def _mix(a, b, k):
        out = {}
        for key in a:
            va, vb = a[key], b[key]
            if va is None or vb is None:
                # 日轮出现/消失：用 glow 淡出，颜色取存在的那个
                out[key] = vb if va is None else va
            elif isinstance(va, tuple):
                out[key] = tuple(va[i] + (vb[i] - va[i]) * k for i in range(3))
            else:
                out[key] = va + (vb - va) * k
        if a["sun"] is None:
            out["sun_glow"] = b["sun_glow"] * k
        elif b["sun"] is None:
            out["sun_glow"] = a["sun_glow"] * (1 - k)
        return out

    # ------------------------------------------------------------ 单帧
    def frame(self, t, fi):
        W, H = self.W, self.H
        p = self.palette(t)

        # 天空渐变
        g = (self.yy / (H - 1)).astype(np.float32)
        g = g ** 1.15
        top = np.array(p["sky_top"], np.float32)
        bot = np.array(p["sky_bot"], np.float32)
        img = top[None, None, :] + (bot - top)[None, None, :] * g[:, :, None]
        img = np.repeat(img, W, axis=1) if img.shape[1] == 1 else img

        # 日轮：先画光晕再画实心盘，压在山后面
        if p["sun_glow"] > 0.01:
            cx = W * 0.5 + math.sin(t * 0.045) * W * 0.055
            cy = H * p["sun_y"]
            rr = np.sqrt((self.xx - cx) ** 2 + (self.yy - cy) ** 2)
            R = H * p["sun_r"]
            sc = np.array(p["sun"], np.float32)
            glow = np.exp(-(rr / (R * 4.6)) ** 2) * p["sun_glow"] * 0.85
            img += sc[None, None, :] * glow[:, :, None]
            disc = np.clip(1.0 - (rr - R) / 7.0, 0, 1) * min(1.0, p["sun_glow"] * 1.4)
            img = img * (1 - disc[:, :, None]) + sc[None, None, :] * disc[:, :, None]

        # 远景云雾（在山前山后各一层）
        off = int((t * 9.0) % W)
        mist = self.cloud[:, off:off + W]
        mfade = np.clip((self.yy / H - 0.10) * 2.0, 0, 1) * np.clip((0.82 - self.yy / H) * 3.2, 0, 1)
        m = mist * mfade * p["mist"] * 0.85
        haze = np.array(p["sky_bot"], np.float32) * 0.55 + np.array(p["sun"] or (200, 200, 200),
                                                                   np.float32) * 0.45
        img = img * (1 - m[:, :, None]) + haze[None, None, :] * m[:, :, None]

        # 桥段的一道斜光：殿门缝里漏进来的冷光，刀就在这道光里
        if p.get("shaft", 0.0) > 0.01:
            slant = 0.34
            cxs = W * 0.335 + math.sin(t * 0.11) * 26
            dist = np.abs(self.xx - (cxs + (self.yy - H * 0.1) * slant))
            beam = np.exp(-(dist / 92.0) ** 2) * np.clip(1.6 - self.yy / H * 1.5, 0, 1)
            beam = beam * p["shaft"] * 46.0
            img += np.array((190, 202, 224), np.float32)[None, None, :] * (beam / 255.0)[:, :, None]

        # 山峦
        ink = np.array(p["ink"], np.float32)
        for i, L in enumerate(self.layers):
            o = int((t * L["spd"]) % L["span"])
            hs = np.concatenate([L["h"], L["h"]])[o:o + W]
            ytop = L["base"] - hs * L["amp"]
            a = np.clip((self.yy - ytop[None, :]) / 2.2, 0, 1) * L["op"]
            # 远山偏淡偏蓝，近山浓黑
            tint = ink * (0.55 + 0.45 * L["op"]) + np.array(p["sky_bot"], np.float32) * (
                0.45 * (1 - L["op"]))
            img = img * (1 - a[:, :, None]) + tint[None, None, :] * a[:, :, None]

        # 近景流云（压在山上，做出"云海"）
        off2 = int((t * 16.0) % W)
        m2 = self.cloud2[:, off2:off2 + W]
        band = np.clip(1 - np.abs(self.yy / H - 0.70) * 5.5, 0, 1)
        m2 = m2 * band * p["mist"] * 0.62
        img = img * (1 - m2[:, :, None]) + haze[None, None, :] * m2[:, :, None]

        # 飞灰 / 火星
        n = int(p["ember_n"])
        if n > 0:
            px = (self.p_x[:n] + self.p_vx[:n] * t + np.sin(t * 0.9 + self.p_ph[:n]) * 26) % W
            py = (self.p_y[:n] + self.p_vy[:n] * t) % H
            ix = px.astype(np.int32)
            iy = py.astype(np.int32)
            ec = np.array(p["ember"], np.float32)
            aa = self.p_a[:n] * (0.55 + 0.45 * np.sin(t * 2.1 + self.p_ph[:n]))
            for dy in range(2):
                for dx in range(2):
                    yq = np.clip(iy + dy, 0, H - 1)
                    xq = np.clip(ix + dx, 0, W - 1)
                    w = aa[:, None] * 0.55
                    img[yq, xq] = img[yq, xq] * (1 - w) + ec[None, :] * w

        # 尾声：江水横向波光
        if p is not None and t >= self.sections[-1]["t"] - XFADE:
            k = np.clip((t - (self.sections[-1]["t"] - XFADE)) / (XFADE * 1.6), 0, 1)
            wb = np.clip(1 - np.abs(self.yy / H - 0.88) * 7.0, 0, 1)
            wave = (np.sin(self.xx * 0.010 + t * 1.1) * 0.5 + 0.5) * \
                   (np.sin(self.xx * 0.031 - t * 0.7 + self.yy * 0.05) * 0.5 + 0.5)
            wv = wave * wb * 0.30 * k
            img = img * (1 - wv[:, :, None]) + 255.0 * wv[:, :, None]

        # ---- 歌词
        self._draw_cues(img, t, p)

        # ---- 标题
        T = self.tl["title"]
        if T["t"] <= t < T["t"] + T["d"]:
            lt = t - T["t"]
            fade = min(1.0, lt / 2.2) * min(1.0, (T["d"] - lt) / 2.4)
            col = np.array(SCENES["intro"]["text"], np.float32)
            self._blit(img, self.sp_title, (H - self.sp_title.h) // 2 - 70, fade,
                       col, reveal=min(1.0, lt / 2.6))
            if lt > 1.6:
                f2 = min(1.0, (lt - 1.6) / 1.6) * min(1.0, (T["d"] - lt) / 2.4)
                self._blit(img, self.sp_sub, (H - self.sp_title.h) // 2 + self.sp_title.h - 40,
                           f2 * 0.85, col, reveal=1.0)

        # ---- 印章（副歌起出现）
        if t > 78:
            sa = min(1.0, (t - 78) / 2.5) * 0.88
            sh, sw = self.seal.shape
            y0, x0 = H - sh - 74, W - sw - 84
            a = self.seal * sa
            reg = img[y0:y0 + sh, x0:x0 + sw]
            img[y0:y0 + sh, x0:x0 + sw] = reg * (1 - a[:, :, None]) + \
                np.array(SEAL_RED, np.float32)[None, None, :] * a[:, :, None]

        # ---- 纸纹 / 暗角 / 颗粒
        img *= self.vignette[:, :, None]
        img += (self.paper * 255.0)[:, :, None]
        img += (self.grain[fi % 8] * 255.0)[:, :, None]

        # ---- 首尾黑场
        if t < 1.6:
            img *= t / 1.6
        tail = self.duration - t
        if tail < 3.2:
            img *= max(0.0, tail / 3.2)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _draw_cues(self, img, t, p):
        H = self.H
        for c, sp in zip(self.tl["cues"], self.sprites):
            if not (c["t"] - 0.05 <= t < c["t"] + c["d"]):
                continue
            lt = t - c["t"]
            fin, fout = 0.85, 0.75
            fade = min(1.0, lt / fin) * min(1.0, (c["d"] - lt) / fout)
            if fade <= 0.01:
                continue
            reveal = min(1.0, lt / 1.05)          # 毛笔从左往右扫出来
            drift = (1.0 - min(1.0, lt / 1.6)) * 16   # 轻微上浮
            y = int(H * 0.775 - sp.h / 2 + drift)
            col = np.array(p["text"], np.float32)
            self._blit(img, sp, y, fade, col, reveal=reveal)

            lb = self.labels.get(c["label"])
            if lb is not None:
                lh, lw = lb.shape
                ly, lx = int(H * 0.20), self.W - lw - 92
                a = lb * fade * 0.42
                reg = img[ly:ly + lh, lx:lx + lw]
                img[ly:ly + lh, lx:lx + lw] = reg * (1 - a[:, :, None]) + \
                    col[None, None, :] * a[:, :, None]

    def _blit(self, img, sp, y, fade, col, reveal=1.0):
        """把文字 sprite 居中贴上去，带墨晕和毛笔扫出效果。"""
        H, W = self.H, self.W
        x = (W - sp.w) // 2
        y0, y1 = max(0, y), min(H, y + sp.h)
        x0, x1 = max(0, x), min(W, x + sp.w)
        if y1 <= y0 or x1 <= x0:
            return
        sy0, sx0 = y0 - y, x0 - x
        a = sp.alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        b = sp.bleed[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]

        if reveal < 1.0:
            edge = 110.0
            xs = np.arange(x0, x1, dtype=np.float32)
            cut = x + sp.w * reveal
            mask = np.clip((cut - xs) / edge, 0, 1)[None, :]
            a = a * mask
            b = b * mask

        reg = img[y0:y1, x0:x1]
        ab = np.clip(b * 0.55 * fade, 0, 1)      # 墨晕先铺
        reg = reg * (1 - ab[:, :, None]) + col[None, None, :] * 0.55 * ab[:, :, None]
        af = np.clip(a * fade, 0, 1)             # 实笔画
        img[y0:y1, x0:x1] = reg * (1 - af[:, :, None]) + col[None, None, :] * af[:, :, None]


_WORKER = None


def _init_worker(tl):
    global _WORKER
    _WORKER = Renderer(tl)


def _render_one(fi):
    return _WORKER.frame(fi / _WORKER.fps, fi).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "天不再借_画面.mp4"))
    ap.add_argument("--preview", action="store_true", help="只渲染几段抽样，快速看效果")
    ap.add_argument("--stills", action="store_true", help="导出关键帧 PNG，不出视频")
    ap.add_argument("--srt", action="store_true", help="按时间轴导出 歌词.srt")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="并行进程数")
    args = ap.parse_args()

    with open(os.path.join(HERE, "timeline.json"), encoding="utf-8") as f:
        tl = json.load(f)

    if args.srt:
        def ts(x):
            h, x = divmod(x, 3600)
            m, s = divmod(x, 60)
            return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")
        lines = []
        for i, c in enumerate(tl["cues"], 1):
            lines.append(f"{i}\n{ts(c['t'])} --> {ts(c['t'] + c['d'])}\n"
                         f"{c['text'].replace(chr(0x3000), '  ')}\n")
        path = os.path.join(HERE, "歌词.srt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("完成：", path)
        return

    r = Renderer(tl)

    if args.stills:
        os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
        for t in (8, 30, 55, 76, 92, 114, 136, 152, 200, 224):
            Image.fromarray(r.frame(t, int(t * r.fps))).save(
                os.path.join(HERE, "stills", f"t{t:03d}.png"))
            print("still", t)
        return

    if args.preview:
        segs = [(4, 16), (28, 38), (88, 100), (130, 140), (196, 206), (218, 230)]
        frames = []
        for a, b in segs:
            frames += list(range(int(a * r.fps), int(b * r.fps)))
    else:
        frames = list(range(int(tl["duration"] * r.fps)))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{r.W}x{r.H}", "-r", str(r.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = len(frames)
    t0 = time.time()

    def report(i):
        if i % 96 and i != total - 1:
            return
        el = time.time() - t0
        eta = el / max(1, i + 1) * (total - i - 1)
        sys.stderr.write(f"\r渲染 {i + 1}/{total}  {(i + 1) / total * 100:5.1f}%  "
                         f"剩余约 {eta / 60:4.1f} 分钟   ")
        sys.stderr.flush()

    if args.jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(tl,)) as pool:
            for i, buf in enumerate(pool.imap(_render_one, frames, chunksize=8)):
                proc.stdin.write(buf)
                report(i)
    else:
        for i, fi in enumerate(frames):
            proc.stdin.write(r.frame(fi / r.fps, fi).tobytes())
            report(i)

    proc.stdin.close()
    proc.wait()
    sys.stderr.write("\n")
    if proc.returncode != 0:
        sys.exit("ffmpeg 编码失败")
    print("完成：", args.out)


if __name__ == "__main__":
    main()
