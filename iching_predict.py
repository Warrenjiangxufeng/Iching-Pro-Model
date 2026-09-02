# -*- coding: utf-8 -*-
from _win_compat import setup_utf8
setup_utf8()
"""对多个品种跑方案 D（64 卦 token 序列 + Transformer），输出「下一卦 + 未来6天涨跌 + 逐日涨跌概率」。

供 Flask 看板 / 脚本调用。逻辑复用 exp_iching_seq 的取数与建模。
"""

import time

import numpy as np

from iching import HEXAGRAMS, BY_BITS
from config import SYMBOLS
from exp_iching_seq import (
    N_CTX,
    build_tokens,
    make_windows,
    Net,
    train,
    marg_day,
    load_prices,
)

# 6 天趋势按「平均上涨概率 avg_prob」(0~1) 判定：高于 50% 看涨、低于 50% 看跌。
# NEUTRAL_BAND 是中性带半宽（概率单位），0.015 = ±1.5%，即 48.5%~51.5% 视为均衡。
# 想更保守（更多中性）就调大，如 0.03；想更敢于表态就调小，如 0.005。
NEUTRAL_BAND = 0.015


# 品种清单统一来自 config.SYMBOLS（20 个，可在 config.py 追加）
# 下方保留 NAME_OF 供外层（含主项目嵌入）反向查询

NAME_OF = {h["value"]: {"num": h["num"], "full": h["full"], "name": h["name"]} for h in HEXAGRAMS}


def _predict_one(model, tokens, feats):
    """model 对某品种最后 N_CTX 个卦 token 预测下一卦（严格未来 6 天）。"""
    import torch

    xt = torch.tensor(tokens[-N_CTX:][None, :], dtype=torch.long)
    xf = torch.tensor(feats[-N_CTX:][None, :], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(xt, xf), dim=1).numpy()[0]
    pred_value = int(probs.argmax())
    pred_bits = []
    for h in HEXAGRAMS:
        if h["value"] == pred_value:
            pred_bits = list(h["bits"])
            break
    per_day = marg_day(probs[None, :])[0]           # (6,) 逐爻为阳(涨)概率
    updown = [("阳" if b else "阴") for b in pred_bits]
    zhifu = ["涨" if b else "跌" for b in pred_bits]

    # 6 天趋势：用逐日上涨概率做「加权累积漂移」。
    # 每天贡献 E[方向] = 2p-1 ∈ [-1,1]；权重近端更高（趋势惯性/近因性），
    # 得到一个 [-1,1] 的看涨-看跌强度分：>0 看涨、<0 看跌，|score| 为强度。
    p = np.asarray(per_day, dtype=np.float64)
    avg_prob = float(p.mean())                        # 6 天平均上涨概率
    # 三态：平均上涨概率高于 50%+带 → 看涨，低于 50%-带 → 看跌，否则中性
    if avg_prob >= 0.5 + NEUTRAL_BAND:
        trend6 = "看涨"
    elif avg_prob <= 0.5 - NEUTRAL_BAND:
        trend6 = "看跌"
    else:
        trend6 = "中性"
    trend6_score = round(abs(avg_prob - 0.5) * 2, 4)  # 看涨/看跌强度 0~1

    return {
        "pred_value": pred_value,
        "pred_hex": NAME_OF.get(pred_value, {}),
        "bits": pred_bits,
        "updown": updown,
        "zhifu": zhifu,
        "perday_prob": per_day.tolist(),
        "avg_prob": float(per_day.mean()),
        "trend6": trend6,
        "trend6_score": trend6_score,
        "probs": probs.tolist(),
    }


def predict_all(demo=False, symbols=None):
    """对给定品种训练 pooled Transformer，返回每品种的最新「下一卦」预测。"""
    syms = symbols or list(SYMBOLS.values())
    name_of = {v: k for k, v in SYMBOLS.items()}
    data = []
    for s in syms:
        try:
            df = load_prices(s, demo)
            tokens, feats = build_tokens(df)
        except Exception as e:
            print(f"[{s}] 取数失败，跳过：{str(e)[:60]}")
            continue
        if len(tokens) <= N_CTX + 5:
            print(f"[{s}] 样本不足，跳过")
            continue
        data.append({"sym": s, "name": name_of.get(s, s),
                     "tokens": tokens, "feats": feats})
    if not data:
        raise RuntimeError("没有足够样本，无法训练")

    # 合并所有品种窗口做 pooled 训练
    all_wins = []
    for d in data:
        w = make_windows(d["tokens"], d["feats"])
        if w is not None:
            all_wins.append(w)
    xt = np.concatenate([w[0] for w in all_wins])
    xf = np.concatenate([w[1] for w in all_wins])
    y = np.concatenate([w[2] for w in all_wins])
    try:
        model = train((xt, xf, y))
    except Exception as e:
        print("[易模型] 训练失败：", str(e)[:80])
        return []

    out = []
    for d in data:
        # 当前最后已知卦（展示用）
        cur_value = int(d["tokens"][-1])
        cur = NAME_OF.get(cur_value, {})
        p = _predict_one(model, d["tokens"], d["feats"])
        out.append({
            "name": d["name"],
            "symbol": d["sym"],
            "current_hex": cur,
            "pred": p,
            "updated": time.strftime("%Y-%m-%d %H:%M"),
        })
    return out


if __name__ == "__main__":
    import json
    res = predict_all(demo=True)
    for r in res:
        p = r["pred"]
        print(f"{r['name']}({r['symbol']}) 当前= {r['current_hex'].get('full')} "
              f"|- 预测下卦= {p['pred_hex'].get('full')} "
              f"(值{p['pred_value']}) 未来6天= {''.join(p['zhifu'])} "
              f"逐日概率= {[round(x,2) for x in p['perday_prob']]}")
