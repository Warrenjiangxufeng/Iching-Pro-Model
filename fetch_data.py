# -*- coding: utf-8 -*-
"""独立抓数模块：用 akshare 抓「期货/股票」日线，带内存 + 磁盘双层缓存。

统一把 A股(stock_zh_a_hist) 与 期货(futures_main_sina) 归一到
date/open/high/low/close/volume，(hold=持仓量仅期货有；A股有 amount=成交额)。
数据落盘到 data/{symbol}.csv，断网时也能用上次缓存训练。
"""
from _win_compat import setup_utf8
setup_utf8()

from pathlib import Path

import pandas as pd


_CACHE = {}   # 进程内缓存，避免重复联网


# 各类数据源 -> 统一字段名的映射
RENAME_MAP = {
    # 期货（futures_main_sina / futures_zh_daily_sina）
    "日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low",
    "收盘价": "close", "成交量": "volume", "持仓量": "hold", "动态结算价": "settle",
    # A股（stock_zh_a_hist）
    "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount",
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名 + 数值化 + 清洗，返回标准 date/open/high/low/close/volume 序列。"""
    df = df.rename(columns=RENAME_MAP)
    df.columns = [str(c).lower().strip() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    for col in ("open", "high", "low", "close", "volume", "settle", "hold", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [date_col] + [c for c in ("open", "high", "low", "close", "volume",
                                     "hold", "settle", "amount") if c in df.columns]
    df = (df[keep].dropna(subset=["close"])
          .query("close > 0")                # 过滤掉非正收盘价，避免 log(0)
          .drop_duplicates(subset=[date_col])
          .sort_values(date_col).reset_index(drop=True))
    return df


def _read_local(symbol: str) -> pd.DataFrame:
    """读取磁盘缓存 data/{symbol}.csv；成功返回 DataFrame，否则 None。"""
    local_csv = Path(__file__).parent / "data" / f"{symbol}.csv"
    if not local_csv.exists():
        return None
    try:
        ldf = normalize(pd.read_csv(local_csv))
        if len(ldf) > 0:
            _CACHE[symbol] = ldf
            print(f"[抓数] 读取本地缓存 {local_csv}，共 {len(ldf)} 行")
            return ldf.copy()
    except Exception as e:
        print(f"[抓数] 本地缓存读取失败({e})，尝试联网…")
    return None


def fetch_stock(symbol: str) -> pd.DataFrame:
    """A股日线。优先东财(前复权)，失败回退新浪；symbol 为 6 位代码，如 '600519'。"""
    import akshare as ak
    # 东财：前复权，支持 6 位纯数字代码
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date="20180101", end_date="20261231", adjust="qfq")
        if df is not None and len(df) > 0:
            return normalize(df)
    except Exception as e:
        print(f"[抓数] 东财A股 {symbol} 失败({str(e)[:60]})，回退新浪…")
    # 新浪：需带市场前缀 sh/sz/bj
    prefix = "sh" if symbol.startswith(("6", "9")) else ("bj" if symbol.startswith(("4", "8")) else "sz")
    sina_symbol = prefix + symbol
    df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date="20180101", end_date="20261231", adjust="qfq")
    return normalize(df)


def fetch_index(symbol: str) -> pd.DataFrame:
    """指数日线。优先东财 index_zh_a_hist，失败回退新浪 stock_zh_index_daily。
    symbol 为指数代码，如 '000300'。"""
    import akshare as ak
    try:
        df = ak.index_zh_a_hist(symbol=symbol, period="daily",
                                start_date="20180101", end_date="20261231")
        if df is not None and len(df) > 0:
            return normalize(df)
    except Exception as e:
        print(f"[抓数] 东财指数 {symbol} 失败({str(e)[:60]})，回退新浪…")
    # 新浪：需前缀。沪=sh000xxx，深=sz399xxx
    prefix = "sh" if symbol.startswith("000") else "sz"
    df = ak.stock_zh_index_daily(symbol=prefix + symbol)
    return normalize(df)


def fetch_futures(symbol: str) -> tuple:
    """期货主力连续日线。返回 (df, src)。"""
    import akshare as ak
    attempts = [
        ("futures_main_sina", lambda: ak.futures_main_sina(symbol=symbol, start_date="20180101", end_date="20261231")),
        ("futures_zh_daily_sina", lambda: ak.futures_zh_daily_sina(symbol=symbol)),
    ]
    for name, fn in attempts:
        try:
            return normalize(fn()), name
        except Exception as e:
            print(f"[抓数] {name} 失败：{e}")
    raise RuntimeError(f"未能抓到期货 {symbol} 数据，请检查网络。")


def fetch_data(symbol: str = "FG0", kind: str = "futures") -> pd.DataFrame:
    """按类型抓日线。kind: 'futures' | 'stock'。优先本地缓存。"""
    if symbol in _CACHE:
        return _CACHE[symbol].copy()
    cand = _read_local(symbol)
    if cand is not None:
        return cand

    if kind == "stock":
        df = fetch_stock(symbol)
        src = "stock_zh_a_hist"
    elif kind == "index":
        df = fetch_index(symbol)
        src = "index_zh_a_hist"
    else:
        df, src = fetch_futures(symbol)

    _CACHE[symbol] = df
    try:
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / f"{symbol}.csv", index=False)
    except Exception as e:
        print(f"[抓数] 落盘失败({e})")
    date_col = df.columns[0]
    print(f"[抓数] 数据源={src}，共 {len(df)} 个交易日，区间 {df[date_col].min().date()} ~ {df[date_col].max().date()}")
    return df.copy()


if __name__ == "__main__":
    for sym, k in [("FG0", "futures"), ("600519", "stock")]:
        try:
            df = fetch_data(sym, kind=k)
            print(f"== {sym} ({k}) ==\n{df.tail(3).to_string()}\n")
        except Exception as e:
            print(f"== {sym} 失败: {e}")
