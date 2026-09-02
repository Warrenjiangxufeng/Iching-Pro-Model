# -*- coding: utf-8 -*-
"""跨平台兼容小工具：统一把 stdout/stderr 设为 UTF-8，避免 Windows(GBK) 控制台
在 print 中文时报 UnicodeEncodeError，同时兼顾其他平台。

用法：在任何会 print 中文的模块顶部 `from _win_compat import setup_utf8; setup_utf8()`。
"""
import sys


def setup_utf8():
    """把 stdout/stderr 重配置为 UTF-8。Py3.7+ 支持 reconfigure，Windows 下最稳。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


setup_utf8()
