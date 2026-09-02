# -*- coding: utf-8 -*-
"""批量抓取品种日线数据并落盘到 data/{代码}.csv（断点续传）。

用法：
    python download_markets.py --kinds futures            # 抓全部期货
    python download_markets.py --kinds stock              # 抓全部 A股
    python download_markets.py --kinds futures,stock --limit 20
"""
from _win_compat import setup_utf8
setup_utf8()

import argparse
import importlib.util
import time
from pathlib import Path

from fetch_data import fetch_data


def load_markets():
    gen = Path(__file__).parent / "markets_generated.py"
    if gen.exists():
        spec = importlib.util.spec_from_file_location("_mg", gen)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MARKETS
    from config import MARKETS
    return MARKETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="futures", help="comma 分隔: futures/stock/index")
    ap.add_argument("--limit", type=int, default=0, help="仅抓前 N 个(调试用)")
    ap.add_argument("--sleep", type=float, default=1.0, help="每抓一个的间隔秒数(防限流)")
    args = ap.parse_args()

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    markets = load_markets()
    selected = [m for m in markets if m[2] in kinds]
    if args.limit:
        selected = selected[:args.limit]

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0
    print(f"== 共 {len(selected)} 个({sorted(kinds)})开始抓取 ==")
    for i, (name, code, kind) in enumerate(selected, 1):
        csv = data_dir / f"{code}.csv"
        if csv.exists():
            skip += 1
            continue
        try:
            df = fetch_data(code, kind=kind)
            ok += 1
            print(f"[{i}/{len(selected)}] {name}({code}) OK {len(df)} 行")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(selected)}] {name}({code}) 失败: {str(e)[:50]}")
        time.sleep(args.sleep)

    print(f"\n完成: 新增 {ok} | 已缓存跳过 {skip} | 失败 {fail}")


if __name__ == "__main__":
    main()
