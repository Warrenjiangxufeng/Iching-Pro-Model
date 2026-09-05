# -*- coding: utf-8 -*-
"""易经独立模型（Iching-Pro-Model）—— 独立训练 / 预测 / 评估 CLI。

目标：把易经模型从主项目脱耦，单独训练并落盘版本化；后续优化完再回嵌主项目。

用法：
    python main.py train [--symbols FG0,SA0] [--demo] [--epochs 80] [--out models/iching.pt]
    python main.py predict [--symbols FG0] [--model models/iching.pt] [--demo]
    python main.py eval   [--symbols FG0,SA0,JM0] [--demo]
    (默认使用 config.SYMBOLS 的 20 个品种；--symbols 可追加/覆盖，为后续扩品做准备)
"""
from _win_compat import setup_utf8
setup_utf8()

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import SYMBOLS, MODEL_DIR, DATA_DIR, EPOCHS, LR, SEED, MARKETS, MARKET_KIND, FEAT
from exp_iching_seq import (
    N_CTX, build_tokens, make_windows, Net, train,
    marg_day, predict, eval_win, load_prices,
)
from iching_predict import _predict_one, NAME_OF


def _load_markets():
    """加载品种清单：优先 markets_generated.py（自动扩品种），否则 config.MARKETS。"""
    gen = Path(__file__).parent / "markets_generated.py"
    if gen.exists():
        spec = importlib.util.spec_from_file_location("_mg", gen)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "MARKETS", None) or MARKETS
    return MARKETS


def _resolve_symbols(symbols_arg):
    """把 --symbols 转成 [(中文名, 代码, 类型), ...]。
    未传则用 config.MARKETS（含期货/A股/指数）。传"代码"自动推断类型；传"名:代码"指定。"""
    if symbols_arg:
        # 用完整清单（含 markets_generated.py 的 381 个）构建名称/类型映射
        full_markets = _load_markets()
        code_to_name = {_c: _n for _n, _c, _k in full_markets}
        name_to_code = {_n: _c for _n, _c, _k in full_markets}
        kind_of_code = {_c: _k for _n, _c, _k in full_markets}
        out = []
        for raw in symbols_arg.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if ":" in raw:                      # 形如 贵州茅台:600519
                name, code = raw.split(":", 1)
                name, code = name.strip(), code.strip()
            elif raw in code_to_name:            # 传代码
                code = raw
                name = code_to_name[raw]
            elif raw in name_to_code:            # 传中文名
                name = raw
                code = name_to_code[raw]
            else:                               # 未知，按代码对待
                code = raw
                name = code_to_name.get(raw, raw)
            kind = kind_of_code.get(code, "futures")
            out.append((name, code, kind))
        return out
    return list(_load_markets())


def _prepare_data(symbols, demo=False):
    """抓数 + 建窗口，返回 {name: (tokens, feats, win)}, 供训练/预测共用。
    symbols: [(name, code, kind), ...]
    窗口缓存到 data/win/{code}.npz，第二次起秒读，避免反复重建。"""
    data = []
    wins = []
    win_dir = Path("data") / "win"
    for name, s, kind in symbols:
        win_path = win_dir / f"{s}_f{FEAT}.npz"
        # 缓存命中：直接读窗口
        if win_path.exists():
            try:
                z = np.load(win_path, allow_pickle=True)
                tokens, feats = z["tokens"], z["feats"]
                w = (z["xt"], z["xf"], z["y"]) if "xt" in z else None
                data.append({"name": name, "sym": s, "kind": kind,
                             "tokens": tokens, "feats": feats})
                wins.append(w)
                continue
            except Exception:
                pass
        try:
            df = load_prices(s, demo, kind=kind)
            tokens, feats = build_tokens(df)
        except Exception as e:
            print(f"[{name} {s}] 取数失败，跳过：{str(e)[:60]}")
            continue
        if len(tokens) <= N_CTX + 5:
            print(f"[{name} {s}] 样本不足，跳过")
            continue
        w = make_windows(tokens, feats)
        if w is None:
            continue
        data.append({"name": name, "sym": s, "kind": kind, "tokens": tokens, "feats": feats})
        wins.append(w)
        # 落盘窗口缓存
        try:
            win_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(win_path, tokens=tokens, feats=feats, xt=w[0], xf=w[1], y=w[2])
        except Exception:
            pass
    return data, wins


def _last_data_date(symbols, demo=False):
    """取所有品种数据最后一天的并集最大值，作为版本指纹。"""
    dates = []
    for _, s, kind in symbols:
        try:
            df = load_prices(s, demo, kind=kind)
            dates.append(str(df["date"].iloc[-1].date()))
        except Exception:
            pass
    return max(dates) if dates else None


def _save_model(model, meta_tag, out_path, extras=None):
    from config import BATCH
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    meta_path = str(Path(out_path).with_suffix(".json"))
    meta = {"model_path": out_path, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_ctx": N_CTX, "seed": SEED}
    if extras:
        meta.update(extras)
    meta.update(meta_tag)
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def _load_model(model_path):
    pt = Path(model_path)
    if not pt.exists():
        return None, None
    meta = None
    mp = pt.with_suffix(".json")
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            meta = None
    model = Net()
    model.load_state_dict(torch.load(pt, map_location="cpu"))
    model.eval()
    return model, meta


def cmd_train(args):
    name_code = _resolve_symbols(args.symbols)
    print(f"== 训练：{len(name_code)} 个品种 ==")
    t0 = time.time()
    data, wins = _prepare_data(name_code, args.demo)
    if len(wins) < 2:
        print("❌ 有效品种不足 2 个，无法训练")
        sys.exit(1)
    xt = np.concatenate([w[0] for w in wins])
    xf = np.concatenate([w[1] for w in wins])
    y = np.concatenate([w[2] for w in wins])
    print(f"   合并窗口：{len(y)} (训练 {time.time()-t0:.1f}s 抓数)")
    model = train((xt, xf, y), epochs=args.epochs, lr=args.lr, seed=args.seed)

    # 全量预测准确率（训练集内 top1 命中，仅供参考）
    probs = predict((xt, xf, y), model)
    top1 = float((probs.argmax(1) == y).mean())
    base = float(np.bincount(y, minlength=64).max() / len(y))
    last_date = _last_data_date(name_code, args.demo)
    extras = {
        "n_varieties": len(data), "n_windows": int(len(y)),
        "train_top1": round(top1, 4), "base_rate": round(base, 4),
        "symbols": [{"name": d["name"], "sym": d["sym"]} for d in data],
        "last_date": last_date,
    }
    meta_path = _save_model(model, {}, args.out, extras)
    print(f"✅ 训练完成 {time.time()-t0:.1f}s | top1={top1:.2%} (基率{base:.2%})")
    print(f"   已保存模型: {args.out}")
    print(f"   元数据  : {meta_path}")


def cmd_predict(args):
    model, meta = _load_model(args.model)
    if model is None:
        print("❌ 未找到模型，请先运行 train")
        sys.exit(1)
    name_code = _resolve_symbols(args.symbols)
    print(f"== 预测：{len(name_code)} 个品种 ==")
    print(f"   模型: {args.model} | 训练于 {meta.get('trained_at') if meta else '未知'}")
    if meta:
        print(f"   版本数据截止: {meta.get('last_date')} | 品种 {meta.get('n_varieties')} | top1 {meta.get('train_top1')}")

    data, _ = _prepare_data(name_code, args.demo)
    for d in data:
        p = _predict_one(model, d["tokens"], d["feats"])
        cur = NAME_OF.get(int(d["tokens"][-1]), {})
        pred = p["pred_hex"] if isinstance(p, dict) else {}
        print(f"\n{d['name']}({d['sym']}) 当前={cur.get('full','?')} -> 预测={p.get('pred_hex',{}).get('full','?')}")
        print(f"   未来6天: {''.join(p['zhifu'])} | 逐日概率: {[round(x,2) for x in p['perday_prob']]}")
        print(f"   6天趋势: {p['trend6']} (强度 {p['trend6_score']})")


def cmd_eval(args):
    name_code = _resolve_symbols(args.symbols)
    print(f"== 留一品种验证：{len(name_code)} 个品种 ==")
    data, wins = _prepare_data(name_code, args.demo)
    for i, held in enumerate(data):
        # 用其它品种训练（直接读 windows 而非抓数）
        other_wins = [w for j, w in enumerate(wins) if j != i]
        if not other_wins:
            continue
        xt = np.concatenate([w[0] for w in other_wins])
        xf = np.concatenate([w[1] for w in other_wins])
        y = np.concatenate([w[2] for w in other_wins])
        m = train((xt, xf, y), epochs=args.epochs, lr=args.lr, seed=args.seed)
        r = eval_win(wins[i], m)
        print(f"\n-- 测试 {held['name']}({held['sym']}) --")
        print(f"  3天AUC {r['avg3']:.4f} | 6天AUC {r['avg6']:.4f} | top1 {r['top1']:.2%} (基率 {r['base']:.2%})")


def main():
    parser = argparse.ArgumentParser(description="易经独立模型 Iching-Pro-Model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="训练并保存模型")
    p_tr.add_argument("--symbols", default=None, help="品种代码列表(逗号分隔)，默认 config.SYMBOLS")
    p_tr.add_argument("--demo", action="store_true", help="合成数据演示")
    p_tr.add_argument("--epochs", type=int, default=EPOCHS)
    p_tr.add_argument("--lr", type=float, default=LR)
    p_tr.add_argument("--seed", type=int, default=SEED)
    p_tr.add_argument("--out", default="models/iching.pt")
    p_tr.set_defaults(func=cmd_train)

    p_pr = sub.add_parser("predict", help="加载模型预测")
    p_pr.add_argument("--symbols", default=None)
    p_pr.add_argument("--demo", action="store_true")
    p_pr.add_argument("--model", default="models/iching.pt")
    p_pr.set_defaults(func=cmd_predict)

    p_ev = sub.add_parser("eval", help="留一品种验证")
    p_ev.add_argument("--symbols", default=None)
    p_ev.add_argument("--demo", action="store_true")
    p_ev.add_argument("--epochs", type=int, default=EPOCHS)
    p_ev.add_argument("--lr", type=float, default=LR)
    p_ev.add_argument("--seed", type=int, default=SEED)
    p_ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
