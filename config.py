# -*- coding: utf-8 -*-
"""集中配置：品种清单（可扩展）+ 建模超参。独立项目统一从这里取数。"""
from _win_compat import setup_utf8
setup_utf8()

# ---- 统一品种清单（可扩展）----
# 每项格式：(中文名, 代码, 类型)，类型 futures=期货主力连续 / stock=A股(6位代码) / index=指数
# 后续追加这里即可；训练默认使用全部 MARKETS。
MARKETS = [
    # ===== 期货（主力连续）=====
    ("玻璃", "FG0", "futures"),
    ("纯碱", "SA0", "futures"),
    ("豆粕", "M0", "futures"),
    ("菜粕", "RM0", "futures"),
    ("焦煤", "JM0", "futures"),
    ("白糖", "SR0", "futures"),
    ("螺纹钢", "RB0", "futures"),
    ("热卷", "HC0", "futures"),
    ("铁矿石", "I0", "futures"),
    ("焦炭", "J0", "futures"),
    ("沪铜", "CU0", "futures"),
    ("沪铝", "AL0", "futures"),
    ("沪金", "AU0", "futures"),
    ("沪银", "AG0", "futures"),
    ("原油", "SC0", "futures"),
    ("甲醇", "MA0", "futures"),
    ("PTA", "TA0", "futures"),
    ("玉米", "C0", "futures"),
    ("棕榈油", "P0", "futures"),
    ("郑棉", "CF0", "futures"),
    # ===== A股（6 位代码，前复权）=====
    ("贵州茅台", "600519", "stock"),
    ("五粮液", "000858", "stock"),
    ("宁德时代", "300750", "stock"),
    ("比亚迪", "002594", "stock"),
    ("中国平安", "601318", "stock"),
    ("招商银行", "600036", "stock"),
    ("工商银行", "601398", "stock"),
    ("万科A", "000002", "stock"),
    ("隆基绿能", "601012", "stock"),
    ("东方财富", "300059", "stock"),
    # ===== 指数 =====
    ("沪深300", "000300", "index"),
    ("上证指数", "000001", "index"),
    ("创业板指", "399006", "index"),
    # ===== 追加区：新品类写这里，格式 ("名称","代码","类型")，逗号结尾 =====
]


# ---- 兼容旧引用：品种名 -> 代码（仅期货，供 iching_predict/train_iching_pooled 等旧代码使用）----
SYMBOLS = {
    "玻璃": "FG0",
    "纯碱": "SA0",
    "豆粕": "M0",
    "菜粕": "RM0",
    "焦煤": "JM0",
    "白糖": "SR0",
    "螺纹钢": "RB0",
    "热卷": "HC0",
    "铁矿石": "I0",
    "焦炭": "J0",
    "沪铜": "CU0",
    "沪铝": "AL0",
    "沪金": "AU0",
    "沪银": "AG0",
    "原油": "SC0",
    "甲醇": "MA0",
    "PTA": "TA0",
    "玉米": "C0",
    "棕榈油": "P0",
    "郑棉": "CF0",
}


# 代码 -> 类型 的映射（从 MARKETS 自动生成），用于抓数时按类型路由
MARKET_KIND = {code: kind for (_name, code, kind) in MARKETS}

# ---- 建模超参（如需调参，改这里，勿散落各处）----
BLOCK = 6            # 6 日不重叠 -> 1 卦 token
NUM = 64            # 卦 token 总数
N_CTX = 20          # 上下文 token 数（约 120 天）
EMB = 16            # embedding 维度
FEAT = 3            # 每 token 连续特征维
SEED = 42
EPOCHS = 80
LR = 1e-3           # 备用学习率（Transformer 实际用 GPT_LR）
BATCH = 64

# ---- Transformer (GPT 风格 decoder-only) 超参 ----
GPT_D_MODEL = 64       # 模型宽度
GPT_NHEAD = 4
GPT_NLAYERS = 2
GPT_DIM_FF = 128
GPT_DROPOUT = 0.1
GPT_NORM_EPS = 1e-5
GPT_LR = 3e-4          # 学习率（默认 1e-3 易发散为 nan）
GPT_GRAD_CLIP = 1.0    # 梯度裁剪范数，防数值爆炸

# ---- 模型/数据路径（相对本项目）----
MODEL_DIR = "models"
DATA_DIR = "data"
