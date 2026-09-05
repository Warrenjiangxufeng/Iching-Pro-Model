# -*- coding: utf-8 -*-
"""用本地全部品种数据训练一轮，看整体命中率(Top1 / 基率 / 方向 AUC)。
做法：每个品种按时间取末尾 20% 窗口做样本外验证，其余合并训练；
     训练循环带进度打印(每 5 epoch 一次)，便于远程盯进度。
用法：venv/bin/python run_full.py [--epochs 30]
"""
import glob
import os
import argparse

import numpy as np
import torch
import torch.nn as nn

import exp_iching_seq as M
from config import NUM, BATCH, GPT_LR, GPT_GRAD_CLIP


def _cat(parts):
    if not parts:
        return (np.zeros((0, 20), dtype=np.int64), np.zeros((0, 20, M.FEAT), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _train_wp(win, epochs=30):
    torch.manual_seed(M.SEED)
    xt = torch.tensor(win[0], dtype=torch.long)
    xf = torch.tensor(win[1], dtype=torch.float32)
    y = torch.tensor(win[2], dtype=torch.long)
    model = M.Net()
    opt = torch.optim.Adam(model.parameters(), lr=GPT_LR)
    cls_w = None
    if M.USE_CLASS_WEIGHT:
        freq = np.bincount(y.numpy(), minlength=NUM).astype(np.float64)
        inv = 1.0 / (freq + 1.0)
        inv = inv / inv.sum() * NUM
        cls_w = torch.tensor(inv, dtype=torch.float32)
    if M.USE_FOCAL:
        lossf = lambda o, yy: M._focal(o, yy, weight=cls_w, gamma=M.FOCAL_GAMMA)
    else:
        lossf = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=M.LABEL_SMOOTHING)
    n = len(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(model(xt[idx], xf[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GPT_GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={total/n:.4f}", flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--holdout", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 个品种(0=全部)，调试用")
    args = ap.parse_args()

    paths = sorted(glob.glob("data/*.csv"))
    symbols = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"===== 本地全部数据训练：{len(symbols)} 个品种 =====", flush=True)

    train_parts, test_parts = [], []
    nok = 0
    for i, s in enumerate(symbols):
        try:
            df = M.load_prices(s)
            tokens, feats = M.build_tokens(df)
            w = M.make_windows(tokens, feats)
        except Exception as e:
            print(f"  [skip {s}] {str(e)[:50]}", flush=True)
            continue
        if w is None:
            continue
        xt, xf, y = w
        cut = int(len(y) * (1 - args.holdout))
        if cut > 0:
            train_parts.append((xt[:cut], xf[:cut], y[:cut]))
        if len(y) - cut > 0:
            test_parts.append((xt[cut:], xf[cut:], y[cut:]))
        nok += 1
        if (i + 1) % 60 == 0:
            print(f"  构建窗口 {i+1}/{len(symbols)} (有效 {nok})", flush=True)

    tw, wv = _cat(train_parts), _cat(test_parts)
    print(f"===== 窗口构建完成：训练 {len(tw[2])} / 样本外 {len(wv[2])} =====", flush=True)
    print(f"===== 开始训练 {args.epochs} epochs =====", flush=True)
    model = _train_wp(tw, epochs=args.epochs)

    tr = M.eval_win(tw, model)
    te = M.eval_win(wv, model)
    print("===== 结果 =====", flush=True)
    print(f"  训练集  : top1 {tr['top1']*100:.2f}% (基率 {tr['base']*100:.2f}%) | avg3 {tr['avg3']:.4f} | avg6 {tr['avg6']:.4f}", flush=True)
    print(f"  样本外  : top1 {te['top1']*100:.2f}% (基率 {te['base']*100:.2f}%) | avg3 {te['avg3']:.4f} | avg6 {te['avg6']:.4f}", flush=True)


if __name__ == "__main__":
    main()
