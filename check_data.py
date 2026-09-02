# -*- coding: utf-8 -*-
"""本地数据质量检查：用健康模板(6品种)比对全量,定位会向训练注入 NaN/Inf 的品种。
只读本地 data/*.csv + data/win/*.npz,不联网。输出精简(只报问题+汇总)。"""
from _win_compat import setup_utf8
setup_utf8()
import warnings, importlib.util, os, sys
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

MARKETS = None
gen = Path("markets_generated.py")
if gen.exists():
    s = importlib.util.spec_from_file_location("_mg", gen)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    MARKETS = m.MARKETS
else:
    from config import MARKETS

# 健康模板
HEALTHY = {"FG0", "SA0", "M0", "600519", "601318", "000300"}

problems = []   # (code, 问题)
healthy_nan = 0
total = 0
for name, code, kind in MARKETS:
    total += 1
    csv = Path(f"data/{code}.csv")
    if not csv.exists():
        continue
    try:
        df = pd.read_csv(csv)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        close = pd.to_numeric(df["close"], errors="coerce") if "close" in df else None
        if close is None:
            problems.append((code, "无close列")); continue
        if (close <= 0).any():
            problems.append((code, f"close<=0 共{(close<=0).sum()}行"))
        if close.isna().any():
            problems.append((code, f"close NaN {close.isna().sum()}行"))
        # 用 build_tokens 检查 feats 是否引入 NaN
        from exp_iching_seq import build_tokens, make_windows, N_CTX
        tokens, feats = build_tokens(df)
        if len(tokens) <= N_CTX + 5:
            problems.append((code, f"token不足({len(tokens)})"))
        elif np.isnan(feats).any():
            problems.append((code, f"feats含NaN {int(np.isnan(feats).sum())}个"))
        elif np.isinf(feats).any():
            problems.append((code, f"feats含Inf {int(np.isinf(feats).sum())}个"))
        else:
            w = make_windows(tokens, feats)
            if w is None:
                problems.append((code, "window为空"))
            else:
                if np.isnan(w[0]).any() or np.isnan(w[1]).any() or np.isnan(w[2]).any():
                    problems.append((code, "窗口含NaN"))
    except Exception as e:
        problems.append((code, f"异常:{str(e)[:40]}"))

print(f"=== 检查 {total} 个品种, 发现问题 {len(problems)} 个 ===")
for c, issue in problems[:40]:
    tag = "【模板】" if c in HEALTHY else "        "
    print(f"  {tag} {c}: {issue}")
if len(problems) > 40:
    print(f"  ... 还有 {len(problems)-40} 个问题品种(仅列前40)")
print("=== 完成 ===")
