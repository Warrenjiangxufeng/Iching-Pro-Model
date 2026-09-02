# -*- coding: utf-8 -*-
from _win_compat import setup_utf8
setup_utf8()
"""易经 64 卦 -> 涨跌序列：为「K线阴阳爻 -> 卦象」的训练/匹配做准备。

约定（本模块唯一口径，请全项目保持一致）：
  - 每一卦 = 上下两个三爻卦（八卦），共 6 爻。
  - 爻序：从下到上读（初爻 -> 上爻），也就是 下卦三爻(自下而上) + 上卦三爻(自下而上)。
  - 编码：阳爻（—） = 1 = 涨；阴爻（- -） = 0 = 跌。
  - 因此一个六位 0/1 串（自下而上）唯一对应一卦；64 卦正好覆盖 2^6 种全部组合（双射）。

用法示例：
    from iching import HEXAGRAMS, match_hexagram, bars_to_hexagrams
    hexs = match_hexagram("涨跌涨跌涨跌")     # 或 "101010" / "阳阴阳阴阳阴"
    h = HEXAGRAMS[0]                          # 第 1 卦：乾为天
    # 若你不是"涨/跌"输入，而是某个价格序列，可以先算每日阴阳，再窗口化匹配。
"""

import json


# --------------------------------------------------------------------------
# 八卦（三爻，自下而上；阳=1，阴=0）
# --------------------------------------------------------------------------
TRIGRAMS = {
    "乾": {"symbol": "☰", "bits": (1, 1, 1)},
    "兑": {"symbol": "☱", "bits": (1, 1, 0)},
    "离": {"symbol": "☲", "bits": (1, 0, 1)},
    "震": {"symbol": "☳", "bits": (1, 0, 0)},
    "巽": {"symbol": "☴", "bits": (0, 1, 1)},
    "坎": {"symbol": "☵", "bits": (0, 1, 0)},
    "艮": {"symbol": "☶", "bits": (0, 0, 1)},
    "坤": {"symbol": "☷", "bits": (0, 0, 0)},
}


# --------------------------------------------------------------------------
# 64 卦（文王卦序）：(序号, 简称, 全名, 上卦, 下卦, 拼音)
# --------------------------------------------------------------------------
_HEX_TABLE = [
    (1, "乾", "乾为天", "乾", "乾", "qián"),
    (2, "坤", "坤为地", "坤", "坤", "kūn"),
    (3, "屯", "水雷屯", "坎", "震", "zhūn"),
    (4, "蒙", "山水蒙", "艮", "坎", "méng"),
    (5, "需", "水天需", "坎", "乾", "xū"),
    (6, "讼", "天水讼", "乾", "坎", "sòng"),
    (7, "师", "地水师", "坤", "坎", "shī"),
    (8, "比", "水地比", "坎", "坤", "bǐ"),
    (9, "小畜", "风天小畜", "巽", "乾", "xiǎo xù"),
    (10, "履", "天泽履", "乾", "兑", "lǚ"),
    (11, "泰", "地天泰", "坤", "乾", "tài"),
    (12, "否", "天地否", "乾", "坤", "pǐ"),
    (13, "同人", "天火同人", "乾", "离", "tóng rén"),
    (14, "大有", "火天大有", "离", "乾", "dà yǒu"),
    (15, "谦", "地山谦", "坤", "艮", "qiān"),
    (16, "豫", "雷地豫", "震", "坤", "yù"),
    (17, "随", "泽雷随", "兑", "震", "suí"),
    (18, "蛊", "山风蛊", "艮", "巽", "gǔ"),
    (19, "临", "地泽临", "坤", "兑", "lín"),
    (20, "观", "风地观", "巽", "坤", "guān"),
    (21, "噬嗑", "火雷噬嗑", "离", "震", "shì kè"),
    (22, "贲", "山火贲", "艮", "离", "bì"),
    (23, "剥", "山地剥", "艮", "坤", "bō"),
    (24, "复", "地雷复", "坤", "震", "fù"),
    (25, "无妄", "天雷无妄", "乾", "震", "wú wàng"),
    (26, "大畜", "山天大畜", "艮", "乾", "dà xù"),
    (27, "颐", "山雷颐", "艮", "震", "yí"),
    (28, "大过", "泽风大过", "兑", "巽", "dà guò"),
    (29, "坎", "坎为水", "坎", "坎", "kǎn"),
    (30, "离", "离为火", "离", "离", "lí"),
    (31, "咸", "泽山咸", "兑", "艮", "xián"),
    (32, "恒", "雷风恒", "震", "巽", "héng"),
    (33, "遁", "天山遁", "乾", "艮", "dùn"),
    (34, "大壮", "雷天大壮", "震", "乾", "dà zhuàng"),
    (35, "晋", "火地晋", "离", "坤", "jìn"),
    (36, "明夷", "地火明夷", "坤", "离", "míng yí"),
    (37, "家人", "风火家人", "巽", "离", "jiā rén"),
    (38, "睽", "火泽睽", "离", "兑", "kuí"),
    (39, "蹇", "水山蹇", "坎", "艮", "jiǎn"),
    (40, "解", "雷水解", "震", "坎", "xiè"),
    (41, "损", "山泽损", "艮", "兑", "sǔn"),
    (42, "益", "风雷益", "巽", "震", "yì"),
    (43, "夬", "泽天夬", "兑", "乾", "guài"),
    (44, "姤", "天风姤", "乾", "巽", "gòu"),
    (45, "萃", "泽地萃", "兑", "坤", "cuì"),
    (46, "升", "地风升", "坤", "巽", "shēng"),
    (47, "困", "泽水困", "兑", "坎", "kùn"),
    (48, "井", "水风井", "坎", "巽", "jǐng"),
    (49, "革", "泽火革", "兑", "离", "gé"),
    (50, "鼎", "火风鼎", "离", "巽", "dǐng"),
    (51, "震", "震为雷", "震", "震", "zhèn"),
    (52, "艮", "艮为山", "艮", "艮", "gèn"),
    (53, "渐", "风山渐", "巽", "艮", "jiàn"),
    (54, "归妹", "雷泽归妹", "震", "兑", "guī mèi"),
    (55, "丰", "雷火丰", "震", "离", "fēng"),
    (56, "旅", "火山旅", "离", "艮", "lǚ"),
    (57, "巽", "巽为风", "巽", "巽", "xùn"),
    (58, "兑", "兑为泽", "兑", "兑", "duì"),
    (59, "涣", "风水涣", "巽", "坎", "huàn"),
    (60, "节", "水泽节", "坎", "兑", "jié"),
    (61, "中孚", "风泽中孚", "巽", "兑", "zhōng fú"),
    (62, "小过", "雷山小过", "震", "艮", "xiǎo guò"),
    (63, "既济", "水火既济", "坎", "离", "jì jì"),
    (64, "未济", "火水未济", "离", "坎", "wèi jì"),
]


def _trigram_bits(name):
    """下卦/上卦三爻（自下而上）0/1 元组。"""
    return TRIGRAMS[name]["bits"]


def _updown(bits):
    return "".join("涨" if b else "跌" for b in bits)


def _yin_yang(bits):
    return "".join("阳" if b else "阴" for b in bits)


def build_hexagrams():
    """把 64 卦表展开，每个卦带上 六爻bits / 涨跌串 / 阴阳串。"""
    out = []
    for num, name, full, upper, lower, pinyin in _HEX_TABLE:
        bits = tuple(_trigram_bits(lower)) + tuple(_trigram_bits(upper))  # 自下而上
        out.append({
            "num": num,
            "name": name,           # 简称（如 乾）
            "full": full,           # 全名（如 乾为天）
            "upper": upper,
            "lower": lower,
            "pinyin": pinyin,
            "symbol": TRIGRAMS[upper]["symbol"] + TRIGRAMS[lower]["symbol"],  # 上卦符号 + 下卦符号
            "bits": bits,           # (初爻...上爻) 0/1
            "yin_yang": _yin_yang(bits),   # 自下而上 阳/阴
            "updown": _updown(bits),       # 自下而上 涨/跌（= 对应 K 线序列）
            "value": int("".join(map(str, bits)), 2),  # 十进制 0~63
        })
    return out


HEXAGRAMS = build_hexagrams()
BY_VALUE = {h["value"]: h for h in HEXAGRAMS}          # 0~63 -> 卦
BY_BITS = {h["bits"]: h for h in HEXAGRAMS}            # bits 元组 -> 卦
BY_NAME = {h["name"]: h for h in HEXAGRAMS}            # 简称 -> 卦
BY_FULL = {h["full"]: h for h in HEXAGRAMS}            # 全名 -> 卦


# --------------------------------------------------------------------------
# 编码 / 匹配工具
# --------------------------------------------------------------------------
_UP_TOKENS = {"1", "阳", "涨", "+", "up", "upward", "yang", "true", "bull", "bullish", "是"}
_DOWN_TOKENS = {"0", "阴", "跌", "-", "down", "downward", "yin", "false", "bear", "bearish", "否"}


def to_bit(token):
    """把任意单个标记归一成 0/1。支持：1/0、阳/阴、涨/跌、+/-, True/False 等。"""
    if isinstance(token, bool):
        return 1 if token else 0
    if isinstance(token, (int, float)):
        return 1 if int(token) else 0
    s = str(token).strip().lower()
    if s in _UP_TOKENS:
        return 1
    if s in _DOWN_TOKENS:
        return 0
    raise ValueError(f"无法识别涨跌标记：{token!r}")


def to_bits(pattern):
    """把 6 个标记归一成 (0/1...) 自下而上元组。"""
    tokens = list(pattern)
    if len(tokens) != 6:
        raise ValueError(f"需要 6 个标记（一卦六爻），实际 {len(tokens)}：{pattern}")
    return tuple(to_bit(t) for t in tokens)


def match_hexagram(pattern):
    """给定一个 6 位涨跌/阴阳序列，返回对应的卦象记录。

    pattern 可以是："涨跌涨跌涨跌" / "101010" / "阳阴阳阴阳阴" / [1,0,1,0,1,0] 等。
    """
    bits = to_bits(pattern)
    h = BY_BITS.get(bits)
    if h is None:
        raise ValueError(f"未找到对应卦象：{bits}")
    return h


def updown_series(df, rule="close>prev_close", window=6):
    """把清洗后的行情 DataFrame 转成「每日阴阳」+「滚动6日 -> 卦象」序列。

    df 需含 date、close 列。rule：
      - "close>prev_close"：当日收盘 > 昨日收盘 => 阳(涨)
      - "close>open"     ：收盘 > 开盘 => 阳(涨)
    返回 list[dict]：{date, is_yang, bit, hexagram*}
    """
    import pandas as pd

    cols = [c for c in ("date", "close", "open") if c in df.columns]
    if "date" not in cols or "close" not in cols:
        raise ValueError("df 至少需要 date、close 列")
    df = df[cols].reset_index(drop=True)
    if rule == "close>open":
        if "open" not in df.columns:
            raise ValueError("rule=close>open 需要 open 列")
        yang = df["close"] > df["open"]
    else:
        yang = df["close"] > df["close"].shift(1)
    yang = yang.fillna(False).astype(bool)
    bits = yang.astype(int).tolist()
    dates = df["date"].tolist()
    window = int(window)
    rows = []
    for i in range(len(dates)):
        start = i - window + 1
        if start < 0:
            continue
        seg = bits[start:i + 1]
        h = BY_BITS.get(tuple(seg))
        rows.append({
            "date": str(dates[i]),
            "is_yang": bool(yang.iloc[i]),
            "bit": int(bits[i]),
            "hexagram": h,
            "window": "".join("阳" if b else "阴" for b in seg),
            "updown_window": "".join("涨" if b else "跌" for b in seg),
        })
    return rows


def dump(path=None):
    """把 64 卦导出为 JSON（默认打印）。"""
    data = [
        {
            "num": h["num"], "name": h["name"], "full": h["full"],
            "upper": h["upper"], "lower": h["lower"], "pinyin": h["pinyin"],
            "bits": list(h["bits"]), "yin_yang": h["yin_yang"],
            "updown": h["updown"], "value": h["value"],
        }
        for h in HEXAGRAMS
    ]
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("已写出", path)
    else:
        print(text)


def main():
    print(f"共 {len(HEXAGRAMS)} 卦；唯一 bits 组合 {len(BY_BITS)}（应为 64，双射验证）")
    print(f"{'#':>3} {'卦名':<10} {'上下':<8} {'六爻(下->上)':<10} {'涨跌序列':<12} {'值':<4}")
    print("-" * 64)
    for h in HEXAGRAMS:
        print(f"{h['num']:>3} {h['full']:<9} {h['upper']+h['lower']:<6} "
              f"{h['yin_yang']:<10} {h['updown']:<12} {h['value']:<4}")


if __name__ == "__main__":
    main()
