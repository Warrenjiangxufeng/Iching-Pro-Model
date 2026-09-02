# -*- coding: utf-8 -*-
"""自动扩品种：从 akshare 拉取「全部期货主连 + 沪深300成分股」生成 MARKETS。

产物：markets_generated.py（内含 MARKETS 列表，供训练/预测用）。
用法：
    python build_markets.py --futures-all --stock-n 300 --out markets_generated.py
    python build_markets.py --futures-n 60 --stock-n 100     # 限制数量
"""
from _win_compat import setup_utf8
setup_utf8()

import argparse
import re
from pathlib import Path


def clean_name(name: str) -> str:
    """去掉「连续」等后缀，如 '玻璃连续' -> '玻璃'。"""
    return re.sub(r"(连续|主力|指数期货合约|期货)$", "", name).strip()


def fetch_futures(all_n: int) -> list:
    """从新浪获取期货主连清单，返回 [(name, symbol, 'futures'), ...]。"""
    import akshare as ak
    df = ak.futures_display_main_sina()   # symbol/exchange/name
    out = []
    for _, r in df.iterrows():
        sym = str(r["symbol"])
        name = clean_name(str(r["name"]))
        out.append((name, sym, "futures"))
    if all_n and len(out) > all_n:
        out = out[:all_n]
    return out


def fetch_stocks(n: int) -> list:
    """从沪深300成分股获取清单，返回 [(name, code, 'stock'), ...]。"""
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol="000300")
    code_key = "成分券代码" if "成分券代码" in df.columns else df.columns[3]
    name_key = "成分券名称" if "成分券名称" in df.columns else df.columns[4]
    out = []
    for _, r in df.iterrows():
        code = str(r[code_key]).zfill(6)
        name = str(r[name_key]).strip()
        out.append((name, code, "stock"))
    if n and len(out) > n:
        out = out[:n]
    return out


def write_markets(markets, out_path):
    """把 MARKETS 写入指定 py 文件。"""
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""由 build_markets.py 自动生成的品种清单（勿手改，重新生成即可）。"""',
        "MARKETS = [",
    ]
    for name, code, kind in markets:
        lines.append(f'    ("{name}", "{code}", "{kind}"),')
    lines.append("]")
    lines.append('')
    lines.append("# 代码 -> 类型 映射（供 load_prices / --symbols 推断）")
    lines.append("MARKET_KIND = {code: kind for (_n, code, kind) in MARKETS}")
    lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 已写入 {len(markets)} 个品种 -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--futures-all", action="store_true", help="全部期货主连")
    ap.add_argument("--futures-n", type=int, default=0, help="期货数量上限(默认全部)")
    ap.add_argument("--stock-n", type=int, default=300, help="沪深300取前 N 只(默认300)")
    ap.add_argument("--out", default="markets_generated.py")
    args = ap.parse_args()

    futures = fetch_futures(args.futures_n if not args.futures_all else 0)
    stocks = fetch_stocks(args.stock_n)
    markets = futures + stocks
    write_markets(markets, args.out)


if __name__ == "__main__":
    main()
