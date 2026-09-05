# -*- coding: utf-8 -*-
"""爻级强弱：把每根 K 线的强度以『爻』的方式记录（不改变卦的 6 爻结构）。

思路：一爻不只阴阳，还有『动静/强弱』（四象：少阴/少阳/老阴/老阳）。
  - 阳爻 + 力度大（老） -> 老阳（动，趋势强）
  - 阳爻 + 力度小（少） -> 少阳（静，温和）
  - 阴爻 + 力度大（老） -> 老阴（动，强）
  - 阴爻 + 力度小（少） -> 少阴（静，弱）
阴阳决定卦的体位（64 卦、卦序、错综互、变卦全保留）；力度用『当根 K 线相对涨跌幅
超过该品种历史分位』判定老/少。

对外提供：
  yao_threshold(close, q)      该品种『老』的力度阈值（|涨跌幅| 的 q 分位）
  yao_strength(close, s, block, thresh)  某 6 日块每爻的连续力度(0~1)
  yao_4state(yang_bits, close, s, block, thresh)  每爻四象(0少阴/1少阳/2老阴/3老阳)
纯计算，零外部依赖。
"""
import numpy as np


def _returns(close):
    c = np.asarray(close, dtype=float)
    r = np.zeros_like(c)
    if len(c) > 1:
        r[1:] = c[1:] / c[:-1] - 1.0
    return r


def yao_threshold(close, q=0.75):
    """该品种『老/强』阈值 = 当根涨跌幅绝对值的 q 分位。"""
    r = _returns(close)
    a = np.abs(r[1:])
    return float(np.quantile(a, q)) if len(a) else 0.0


def yao_strength(close, s, block=6, thresh=None):
    """第 s 天起连续 block 根的『爻力度』(0~1)。1=达到老/强阈值。"""
    r = _returns(close)
    if thresh is None:
        thresh = yao_threshold(close)
    seg = np.abs(r[s:s + block])
    if thresh > 0:
        return np.clip(seg / thresh, 0.0, 1.0).astype(np.float32)
    return np.zeros(block, dtype=np.float32)


def yao_4state(yang_bits, close, s, block=6, thresh=None):
    """每爻四象：0=少阴 1=少阳 2=老阴 3=老阳。"""
    r = _returns(close)
    if thresh is None:
        thresh = yao_threshold(close)
    seg = r[s:s + block]
    out = []
    for i, y in enumerate([int(v) for v in yang_bits]):
        old = int(abs(seg[i]) >= thresh) if thresh > 0 else 0
        out.append((3 if old else 1) if y else (2 if old else 0))
    return np.array(out, dtype=np.int64)


if __name__ == "__main__":
    # 验证：同样『乾』(连阳)，大阳=老阳动爻多；小阳=少阳静爻
    import numpy as np
    base = 100.0
    big = base * np.cumprod([1.04] * 6)        # 连续大阳（每根+4%）
    small = big[-1] * np.cumprod([1.002] * 6)  # 连续小阳（每根+0.2%，接续 big）
    close = np.concatenate([[base], big, small])   # 前置一天给首根算涨跌
    yang = (close[1:] > close[:-1]).astype(int)    # 每根阴阳（涨=1）
    th = yao_threshold(close)
    print(f"『老/强』阈值(|涨跌幅| 75分位) = {th:.4f}")
    for name, s, ybits in (("连续大阳-乾", 1, yang[0:6]), ("连续小阳-乾", 7, yang[6:12])):
        mask = yao_4state(ybits, close, s, thresh=th)       # 6 位
        str_ = yao_strength(close, s, thresh=th)
        labels = {0: "少阴", 1: "少阳", 2: "老阴", 3: "老阳"}
        four = " ".join(labels[v] for v in mask)
        print(f"{name}: 四象=[{four}]  爻力度=({', '.join(f'{x:.2f}' for x in str_)})")
