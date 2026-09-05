# -*- coding: utf-8 -*-
"""易经结构特征：错卦/综卦/互卦/文王卦序/上下卦/自综。

把每卦的"结构关系"先算成一个定长向量，供模型当作结构 embedding 注入（即"推演一卦"，
而非猜编号）。纯计算，零依赖，输入 0..63 的卦 value，输出结构向量。

口径（与 iching.py 保持一致）：
  - 六爻自下而上：bits[0]=初爻 ... bits[5]=上爻；阳/涨=1，阴/跌=0。
  - 错卦 = 6 位全取反；综卦 = 6 位上下颠倒(reverse)；互卦 = 取 2,3,4 爻为下卦 + 3,4,5 爻为上卦。
  - 文王卦序 = 1..64（用 iching.HEXAGRAMS 中的文王序号/列表顺序）。
"""
import numpy as np

from iching import HEXAGRAMS, TRIGRAMS, BY_VALUE, BY_NAME


# 八卦 -> 三爻 bits（自下而上）
_TRIG_BITS = {name: t["bits"] for name, t in TRIGRAMS.items()}
_TRIG_NAMES = list(TRIGRAMS.keys())   # 顺序：乾兑离震巽坎艮坤


def _bits_to_value(bits):
    return int("".join(str(int(b)) for b in bits), 2)


def _trigram_index(bits3):
    bits3 = tuple(int(b) for b in bits3)
    for idx, name in enumerate(_TRIG_NAMES):
        if tuple(_TRIG_BITS[name]) == bits3:
            return idx
    return 0


def _seq_of(value):
    """文王卦序(1..64)。展开里已保留 num=文王序号。"""
    h = BY_VALUE.get(int(value))
    if h is None:
        return 1
    return int(h.get("num", 1))


def structure_vector(value):
    """输入卦 value(0..63)，返回 numpy float32 结构向量。"""
    h = BY_VALUE.get(int(value))
    if h is None:
        bits = [0] * 6
    else:
        bits = list(h["bits"])
    # 错 / 综 / 互
    cuobits = [1 - b for b in bits]
    zongbits = bits[::-1]
    hubits = bits[1:4] + bits[2:5]          # 下卦=2,3,4爻；上卦=3,4,5爻（自下而上）
    cuo = _bits_to_value(cuobits)
    zong = _bits_to_value(zongbits)
    hu = _bits_to_value(hubits)
    seq = _seq_of(value)
    lo = _trigram_index(bits[0:3])
    up = _trigram_index(bits[3:6])
    self_zh = 1.0 if bits == bits[::-1] else 0.0
    return np.array([
        cuo / 63.0, zong / 63.0, hu / 63.0,
        (seq - 1) / 63.0, lo / 7.0, up / 7.0, self_zh,
    ], dtype=np.float32)


# 结构向量维度
STRUCT_DIM = 7


# 预计算 0..63 号卦的结构表 (64, STRUCT_DIM)，供模型注册为 buffer
STRUCT_TABLE = np.stack([structure_vector(v) for v in range(64)], axis=0).astype(np.float32)


if __name__ == "__main__":
    # 自测：验证经典卦的关系是否正确
    for name in ("乾", "坤", "屯", "蒙", "泰", "否"):
        h = BY_NAME[name]
        v = h["value"]
        vec = structure_vector(v)
        cuo = BY_VALUE[int(vec[0] * 63)]
        zong = BY_VALUE[int(round(vec[1] * 63))]
        hu = BY_VALUE[int(round(vec[2] * 63))]
        print(f"{name}({v:>2}): 错={cuo['name']} 综={zong['name']} 互={hu['name']} 文王序={_seq_of(v)}")
