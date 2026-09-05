# -*- coding: utf-8 -*-
from _win_compat import setup_utf8
setup_utf8()
"""方案 D：把 64 卦当 token，多品种合并 -> Transformer 序列建模 -> 预测下一卦。

逻辑：
  1) 每个 6 日(不重叠)窗口 = 1 个卦 token(0..63)。
  2) 一条时间轴 => 一串卦 token；对每个 token 取前面 N_CTX 个 token 作为上下文。
  3) 每个 token 配少量连续特征(收益/波动/末次收益)，与 embedding 拼接送入 Transformer。
  4) Transformer decoder-only -> Linear -> 64 类 softmax = P(下一卦)。
  5) 评估：留一品种验证(训练其它品种、测试该品种)，报告逐日方向 AUC(3天/6天)与整卦命中率。
说明：沙箱无网时用多品种合成序列(种子不同)验证流程；真实数据用 --symbols 指定。

运行：
    python exp_iching_seq.py --symbols FG0,SA0,JM0            # 真实数据
    ICHING_DEMO=1 python exp_iching_seq.py                   # 合成 3 品种
"""

import os
import zlib
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from iching import HEXAGRAMS, BY_BITS
from iching_structure import STRUCT_DIM, STRUCT_TABLE
from iching_yao import yao_strength
from iching_time import time_features

from config import (
    BLOCK, NUM, N_CTX, EMB, FEAT, SEED, EPOCHS, LR, BATCH, MARKET_KIND,
    GPT_D_MODEL, GPT_NHEAD, GPT_NLAYERS, GPT_DIM_FF, GPT_DROPOUT, GPT_NORM_EPS,
    GPT_LR, GPT_GRAD_CLIP, USE_STRUCT, USE_YAO, YAO_Q, USE_TIME, USE_CLASS_WEIGHT, USE_FOCAL,
    LABEL_SMOOTHING, FOCAL_GAMMA,
)


def make_demo(symbol, n=1400):
    """按 symbol 生成不同参数的多品种合成序列（演示用）。"""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()) % 100000)
    dt = pd.date_range("2018-01-01", periods=n, freq="D")
    vol_d = np.zeros(n)
    ret_d = np.zeros(n)
    vol_d[0] = 0.010
    mom = 0.12 + (rng.random() * 0.12)         # 每品种动量不完全一样
    for i in range(1, n):
        vol_d[i] = np.sqrt(0.00005 + 0.85 * vol_d[i - 1] ** 2 + 0.12 * ret_d[i - 1] ** 2)
        ret_d[i] = mom * ret_d[i - 1] + rng.normal(0, vol_d[i])
    period = 48 + int(rng.random() * 48)
    seasonal = 0.004 * np.sin(2 * np.pi * np.arange(n) / period)
    close = 2000 * np.exp(np.cumsum(ret_d + seasonal))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(500000, 2000000, n)
    return pd.DataFrame({"date": dt, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def load_prices(symbol, demo=False, kind=None):
    # demo：直接合成数据，不联网（用于离线验证流程/调参）
    if demo:
        return make_demo(symbol)
    # 从统一清单推断类型（futures/stock/index），未匹配默认 futures
    if kind is None:
        kind = MARKET_KIND.get(symbol, "futures")
    # 抓取（内部先读磁盘缓存 data/{symbol}.csv，缺失才联网；统一列名）
    try:
        from fetch_data import fetch_data
        df = fetch_data(symbol, kind=kind)
        print(f"[{symbol}] fetch_data 成功，{len(df)} 行")
        return df
    except Exception as e:
        print(f"[{symbol}] 抓数失败：{str(e)[:60]}")
    raise RuntimeError(f"无本地数据且网络不可用：{symbol}")


def yang_series(df):
    return (df["close"].astype(float) > df["close"].astype(float).shift(1)) \
        .fillna(False).astype(int).to_numpy()


def build_tokens(df):
    """不重叠 6 日块 -> 卦 token 序列 + 每块特征。
    特征：3 基础(ret/vol/last) + 6 爻力度(以『爻』记录每根 K 线强弱，老/少)。"""
    yang = yang_series(df)
    n = len(yang)
    nb = (n - BLOCK + 1) // BLOCK
    close_all = df["close"].astype(float).to_numpy()
    dates = df["date"].astype(str).str[:10].to_numpy() if "date" in df else None
    tokens, feats = [], []
    for k in range(nb):
        s = k * BLOCK
        bits = tuple(int(yang[s + i]) for i in range(BLOCK))
        h = BY_BITS.get(bits)
        if h is None:
            continue
        seg = df.iloc[s:s + BLOCK]
        close = seg["close"].astype(float).to_numpy()
        ret = close[-1] / close[0] - 1
        logret = np.diff(np.log(close)) if len(close) > 1 else np.array([0.0])
        vol = float(np.std(logret)) if len(logret) > 1 else 0.0
        last = (close[-1] / close[-2] - 1) if len(close) > 1 else 0.0
        tokens.append(h["value"])
        feats.append([ret, vol, last])
        j = s + BLOCK - 1
        if USE_YAO:
            # 用『截至块末』的历史算爻力度（老/少），不做未来泄漏
            ystr = yao_strength(close_all[:j + 1], s, BLOCK, q=YAO_Q)
            feats[-1].extend([float(v) for v in ystr])
        if USE_TIME and dates is not None:
            tv = time_features(dates[j])
            feats[-1].extend([float(v) for v in tv])
    return np.asarray(tokens, dtype=np.int64), np.asarray(feats, dtype=np.float32)


def make_windows(tokens, feats, n_ctx=N_CTX):
    """从 token 序列生成 (上下文token, 上下文特征, 目标token)。"""
    Xt, Xf, Y = [], [], []
    for i in range(n_ctx, len(tokens)):
        Xt.append(tokens[i - n_ctx:i])
        Xf.append(feats[i - n_ctx:i])
        Y.append(tokens[i])
    if not Xt:
        return None
    return (np.asarray(Xt, dtype=np.int64), np.asarray(Xf, dtype=np.float32),
            np.asarray(Y, dtype=np.int64))


def _apply_rope(x, cos, sin):
    """标准 RoPE：x (B,L,D)，cos/sin (L,D/2)。拆成相邻 pair 旋转。"""
    # 相邻两两配对：(B,L,D/2,2)
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = xr[..., 0], xr[..., 1]      # (B,L,D/2)
    cos = cos[None, None]                # (1,1,L,D/2)
    sin = sin[None, None]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack((xr1, xr2), dim=-1).reshape(*x.shape)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=GPT_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class _SwiGLUFFN(nn.Module):
    """SwiGLU 门控 FFN（GPT 系标配）。"""
    def __init__(self, d_model, dim_ff, dropout):
        super().__init__()
        # 3*d_model: 门控变量 (swish(gate) * up)
        self.gate = nn.Linear(d_model, dim_ff)
        self.up = nn.Linear(d_model, dim_ff)
        self.down = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        g = torch.nn.functional.silu(self.gate(x))
        h = g * self.up(x)
        return self.dropout(self.down(h))


class _GPTLayer(nn.Module):
    """decoder-only 层：因果自注意力 + SwiGLU FFN，均带 RMSNorm + 残差。"""
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = _RMSNorm(d_model)
        self.ffn = _SwiGLUFFN(d_model, dim_ff, dropout)

    def forward(self, x, causal_mask):
        import torch.nn.functional as F
        B, L, D = x.shape
        normed = self.norm1(x)
        H = self.nhead
        q = self.q_proj(normed).reshape(B, L, H, D // H).permute(0, 2, 1, 3)  # (B,H,L,hd)
        k = self.k_proj(normed).reshape(B, L, H, D // H).permute(0, 2, 1, 3)
        v = self.v_proj(normed).reshape(B, L, H, D // H).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=self.dropout.p if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, L, D)   # (B, L, D)
        a = self.proj(a)
        x = x + a
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class _TransformerNet(nn.Module):
    """GPT 风格 decoder-only：卦 token 序列 -> 自回归预测下一卦。
    含 RoPE 位置编码 + RMSNorm + SwiGLU FFN + 因果掩码 + weight tying。"""
    def __init__(self, d_model=GPT_D_MODEL, nhead=GPT_NHEAD, nlayers=GPT_NLAYERS,
                 dim_ff=GPT_DIM_FF, dropout=GPT_DROPOUT):
        super().__init__()
        self.arch = "transformer"
        self.d_model = d_model
        self.n_ctx = N_CTX
        self.emb = nn.Embedding(NUM, EMB)
        # 易经结构注入：错/综/互/文王序/上下卦 -> 结构向量，叠加到卦 embedding
        self.struct_emb = nn.Linear(STRUCT_DIM, EMB) if USE_STRUCT else None
        if USE_STRUCT:
            self.register_buffer("struct_table", torch.from_numpy(STRUCT_TABLE).float(), persistent=False)
        self.feat_proj = nn.Linear(FEAT, d_model - EMB) if d_model > EMB else nn.Identity()
        self.layers = nn.ModuleList([
            _GPTLayer(d_model, nhead, dim_ff, dropout) for _ in range(nlayers)
        ])
        self.norm = _RMSNorm(d_model)
        self.head = nn.Linear(d_model, NUM)
        # 预计算 RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        positions = torch.arange(N_CTX, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)        # (L, D/2)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, xt, xf):
        L = xt.size(1)
        e = self.emb(xt)                       # (B, L, EMB)
        if USE_STRUCT:
            se = self.struct_emb(self.struct_table[xt])   # (B, L, EMB)
            e = e + se
        fp = self.feat_proj(xf)                # (B, L, d_model-EMB)
        x = torch.cat([e, fp], dim=-1)         # (B, L, d_model)
        x = _apply_rope(x, self.cos[:L], self.sin[:L])   # RoPE 位置编码
        # 因果掩码：预测第 i 位只能看前 i 位（0=允许看，-inf=遮住未来）
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.float32), diagonal=1).to(x.device) * -1e9
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.norm(x)
        return self.head(x[:, -1])             # (B, NUM)


def Net():
    """卦象序列 Transformer（GPT 风格 decoder-only）。"""
    return _TransformerNet()


def _focal(out, y, weight=None, gamma=2.0):
    """focal loss：聚焦模型总猜错的样本（长尾/难样本）。"""
    import torch.nn.functional as F
    logp = F.log_softmax(out, dim=1)
    logpt = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    w = (1.0 - pt) ** gamma
    if weight is not None:
        w = w * weight[y]
    return -(w * logpt).mean()


def train(win, epochs=EPOCHS, lr=LR, seed=SEED):
    torch.manual_seed(seed)
    xt = torch.tensor(win[0], dtype=torch.long)
    xf = torch.tensor(win[1], dtype=torch.float32)
    y = torch.tensor(win[2], dtype=torch.long)
    model = Net()
    # Transformer 用较小学习率 + 梯度裁剪，避免数值发散成 nan
    opt = torch.optim.Adam(model.parameters(), lr=GPT_LR)
    # 类别加权（拉平 64 卦长尾）：低频卦权重高，逼模型学会冷门卦
    cls_w = None
    if USE_CLASS_WEIGHT:
        freq = np.bincount(y.numpy(), minlength=NUM).astype(np.float64)
        inv = 1.0 / (freq + 1.0)
        inv = inv / inv.sum() * NUM
        cls_w = torch.tensor(inv, dtype=torch.float32)
    if USE_FOCAL:
        lossf = lambda o, yy: _focal(o, yy, weight=cls_w, gamma=FOCAL_GAMMA)
    else:
        lossf = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=LABEL_SMOOTHING)
    n = len(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + 64]
            opt.zero_grad()
            loss = lossf(model(xt[idx], xf[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GPT_GRAD_CLIP)
            opt.step()
    model.eval()
    return model


def marg_day(probs):
    probs = np.asarray(probs, dtype=np.float64)
    out = np.zeros((probs.shape[0], BLOCK), dtype=np.float64)
    for h in HEXAGRAMS:
        for j in range(BLOCK):
            if h["bits"][j] == 1:
                out[:, j] += probs[:, h["value"]]
    return out


def roc_auc(y01, score):
    from sklearn.metrics import roc_auc_score
    y01 = np.asarray(y01)
    score = np.asarray(score)
    return float(roc_auc_score(y01, score)) if np.unique(y01).size > 1 else None


def predict(win, model):
    xt = torch.tensor(win[0], dtype=torch.long)
    xf = torch.tensor(win[1], dtype=torch.float32)
    with torch.no_grad():
        return torch.softmax(model(xt, xf), dim=1).numpy()


def eval_win(win, model):
    probs = predict(win, model)
    y = win[2]
    bits = np.array([_bits_of_value(v) for v in y], dtype=int)
    mday = marg_day(probs)
    aucs = np.asarray([a if (a := roc_auc(bits[:, j], mday[:, j])) is not None else np.nan
                       for j in range(BLOCK)])
    avg3 = float(np.nanmean(aucs[:3]))
    avg6 = float(np.nanmean(aucs))
    top1 = float((probs.argmax(1) == y).mean())
    base = float(np.bincount(y, minlength=NUM).max() / len(y))
    return {"aucs": aucs, "avg3": avg3, "avg6": avg6, "top1": top1, "base": base}


def _bits_of_value(v):
    bits = [0] * BLOCK
    for h in HEXAGRAMS:
        if h["value"] == int(v):
            return list(h["bits"])
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=os.environ.get("ICHING_SYMBOLS", "FG0,SA0,JM0"))
    ap.add_argument("--demo", action="store_true", default=os.environ.get("ICHING_DEMO") == "1")
    args = ap.parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    wins = []
    for i, s in enumerate(syms):
        df = load_prices(s, args.demo)
        tokens, feats = build_tokens(df)
        w = make_windows(tokens, feats)
        if w is None:
            print(f"[{s}] 样本不足，跳过")
            continue
        wins.append({"sym": s, "win": w})
    if len(wins) < 2:
        raise RuntimeError("有效品种少于 2 个，无法做留一验证")

    print(f"\n各品种样本（不重叠卦 token 窗口）：")
    for w in wins:
        print(f"  {w['sym']}: {len(w['win'][0])} 窗口")

    print("\n== 留一品种验证（训练其它品种，测试该品种） ==")
    for held in wins:
        tr_wins = [w["win"] for w in wins if w["sym"] != held["sym"]]
        # 合并其它品种的窗口作训练
        xt = np.concatenate([w[0] for w in tr_wins])
        xf = np.concatenate([w[1] for w in tr_wins])
        y = np.concatenate([w[2] for w in tr_wins])
        model = train((xt, xf, y))
        r = eval_win(held["win"], model)
        print(f"\n-- 测试品种 {held['sym']} --")
        print(f"  逐日AUC: " + " ".join(f"{a:.3f}" for a in r["aucs"]))
        print(f"  3天平均 {r['avg3']:.4f} | 6天平均 {r['avg6']:.4f} | 整卦top1 {r['top1']:.2%} (基率 {r['base']:.2%})")


if __name__ == "__main__":
    main()
