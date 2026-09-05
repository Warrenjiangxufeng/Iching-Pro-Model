# -*- coding: utf-8 -*-
"""干支纪时：把 K 线日期转成『时间特征向量』，让每个卦有时间概念。

给出（均为归一化到 0~1，供模型当伴随特征）：
  year_gz   年干支六十甲子序号(0~59)；用公历年近似(立春前略有偏差)
  month_stem 月干(0~9，五虎遁，寅月起)
  month_zhi  月支(0=寅 ... 11=丑)
  day_gz    日干支六十甲子序号(0~59，JDN 法，2000-01-01=戊午=54)
  month     公历月份/12
  day       公历日/31
  doy       一年中的第几天/366
  season    季节(0春/1夏/2秋/3冬)/3
纯计算，零外部依赖。
"""
import datetime as _dt

import numpy as np


TIME_DIM = 8


def _jdn(y, m, d):
    """公历->儒略日数(Gregorian)。"""
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524


def day_gz(y, m, d):
    """日干支六十甲子序号(0=甲子)。"""
    return (_jdn(y, m, d) + 49) % 60


def year_gz(y):
    """年干支六十甲子序号(0=甲子)，公历年近似。"""
    return (y - 4) % 60


def month_gz(y, m):
    """月干 + 月支(支: 0=寅 ... 11=丑)。五虎遁定寅月干。"""
    yg = (y - 4) % 10
    zhi_delta = (m - 2) % 12                  # 相对寅月的偏移(寅=0)，用于月干推进
    first_stem = ((yg % 5) * 2 + 2) % 10      # 甲己之年丙寅作首...
    stem = (first_stem + zhi_delta) % 10
    return stem, m % 12                        # 月支绝对地支序号(子=0, 寅=2)


def time_features(date_str):
    """日期串 -> 8 维时间特征(0~1)。解析失败给全 0。"""
    try:
        y, m, d = (int(x) for x in str(date_str)[:10].split("-"))
    except Exception:
        return np.zeros(TIME_DIM, dtype=np.float32)
    yg = year_gz(y)
    dg = day_gz(y, m, d)
    ms, mz = month_gz(y, m)
    doy = _dt.date(y, m, d).timetuple().tm_yday
    season = (m % 12) // 3
    return np.array([
        yg / 59.0, ms / 9.0, mz / 11.0, dg / 59.0,
        m / 12.0, d / 31.0, doy / 366.0, season / 3.0,
    ], dtype=np.float32)


if __name__ == "__main__":
    _GAN = "甲乙丙丁戊己庚辛壬癸"
    _ZHI = "子丑寅卯辰巳午未申酉戌亥"
    def gz(idx):
        return _GAN[idx % 10] + _ZHI[idx % 12]
    print("2000-01-01 日干支 =", gz(day_gz(2000, 1, 1)), "(应 戊午)")
    print("2024 年干支 =", gz(year_gz(2024)), "(应 甲辰)")
    ms, mz = month_gz(2024, 2)
    print("2024-02(建寅月) 月干支 =", _GAN[ms] + _ZHI[mz])
    print("2024-02-10 时间向量 =", time_features("2024-02-10").round(3))
