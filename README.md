# 易经独立模型（Iching-Pro-Model）

从主项目 `Trading-model-web` 中**独立抽取**的易经（64 卦 token 序列 + Transformer）模型项目。
目的：把易经模型从主项目脱耦，单独训练、评估、优化；优化完成后可作为独立依赖回嵌主项目。

> ⚠️ 仅供学习研究，不构成投资建议。

## 模型说明

把**期货 / A股 / 指数**日线按**不重叠 6 日**切块，每块对应一个 0~63 的卦 token（64 卦与 6 位 0/1 串双射）：

- 6 日块 → 1 卦 token（0..63）
- 一条时间轴 → 一串卦 token；对每个 token 取前 `N_CTX=20` 个卦作上下文
- 每个卦搭配少量连续特征（收益 / 波动 / 末次收益），与卦 embedding 拼接送入 Transformer
- Transformer decoder-only → 64 类 softmax = P(下一卦)
- 预测下一卦后，拆成 6 爻，映射为未来 6 天涨/跌与逐日上涨概率

### 架构：纯 Transformer（GPT 风格 decoder-only）

模型为 GPT 风格 decoder-only：Embedding + RoPE 位置编码 + RMSNorm + SwiGLU FFN + 因果自注意力 → 64 类分类。

相关超参在 `config.py`：`GPT_D_MODEL`(64) / `GPT_NHEAD`(4) / `GPT_NLAYERS`(2) / `GPT_DIM_FF`(128) / `GPT_DROPOUT`(0.1) / `GPT_LR`(3e-4) / `GPT_GRAD_CLIP`(1.0)。

## 目录结构

```
Iching-Pro-Model/
├── main.py            # CLI 入口：train / predict / eval
├── config.py          # 集中配置：品种清单(SYMBOLS, 可扩展) + 超参 + 路径
├── build_markets.py   # 自动扩品种：从接口生成 markets_generated.py（全部期货+沪深300）
├── markets_generated.py  # 自动生成的品种清单（81期货 + 300 A股 = 381，勿手改）
├── exp_iching_seq.py  # 建模核心：Net / train / build_tokens / make_windows / 评估
├── iching.py          # 64 卦字典 + 编码/匹配工具（零外部依赖）
├── iching_predict.py  # 品种->卦 的预测封装（predict_all, _predict_one）
├── fetch_data.py      # 独立抓数 + 磁盘缓存（与主项目解耦）
├── _win_compat.py     # 跨平台 UTF-8 输出兼容
├── requirements.txt
├── data/              # 行情落盘缓存 data/{symbol}.csv
└── models/            # 训练产物 models/iching.pt + .json 元数据
```

## 安装

本项目**复用主项目 `Trading-model-web/venv`** 的依赖（numpy/pandas/torch/sklearn/akshare 已齐全），无需重复安装。

便捷入口（推荐）：`./run.sh` 会自动优先使用本项目 `venv`，否则回退到主项目 `Trading-model-web/venv`，再退到系统 `python3`：

```bash
./run.sh train --demo --symbols FG0,SA0,M0
./run.sh predict --model models/iching.pt
```

若确需独立环境（如主项目 venv 不可用），再按需创建：

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## 用法（CLI）

### 训练（默认 config.MARKETS：期货 + A股 + 指数）
```bash
python main.py train
# 指定品种 / 调参
python main.py train --symbols FG0,SA0,M0 --epochs 80 --out models/iching.pt
# 离线合成数据演示
python main.py train --demo --symbols FG0,SA0,M0
# 混合：期货 + A股 + 指数（--symbols 传代码，自动按类型路由）
python main.py train --symbols FG0,SA0,M0,600519,601318,000300
```
训练结果落盘为 `models/iching.pt` + `models/iching.json`（含数据截止日、品种数、train_top1），
供预测时直接加载、不重复训练。

### 预测（加载模型）
```bash
python main.py predict --model models/iching.pt
python main.py predict --symbols FG0 --model models/iching.pt
```
输出每个品种的当前卦、预测下一卦、未来 6 天涨跌、逐日上涨概率、6 天趋势。

### 评估（留一品种验证）
```bash
python main.py eval --symbols FG0,SA0,M0
```
报告逐日 AUC（3 天 / 6 天）与整卦 top1 命中率（相对基率）。

## 数据 / 品种（期货 / A股 / 指数）

数据由 `fetch_data.py` 统一抓取并归一化列名（`date/open/high/low/close/volume`），落盘 `data/{代码}.csv`，多次训练命中磁盘缓存、断网可用：

| 类型 | 代码示例 | 抓数源（带回退） |
|------|----------|------------------|
| 期货主力连续 | `FG0`/`SA0`/`M0`... | `futures_main_sina` → `futures_zh_daily_sina` |
| A股 | `600519`/`000858`... | `stock_zh_a_hist`(东财,前复权) → `stock_zh_a_daily`(新浪) |
| 指数 | `000300`/`000001`/`399006`... | `index_zh_a_hist`(东财) → `stock_zh_index_daily`(新浪) |

### 品种扩展

品种清单集中在 `config.py` 的 `MARKETS` 列表，每项 `(名称, 代码, 类型)`，追加只需加一行：
```python
("名称", "代码", "futures|stock|index"),   # 例：("宁德时代","300750","stock")
```
也可用 `--symbols 代码1,代码2` 临时指定（代码自动推断类型；或用 `名称:代码` 显式指定）。

### 自动扩品种（推荐）

用 `build_markets.py` 一键生成大清单（全部期货主连 + 沪深300成分股），训练时自动优先使用：
```bash
python build_markets.py --futures-all --stock-n 300      # 81 期货 + 300 A股 = 381
python build_markets.py --futures-n 60 --stock-n 100     # 控制数量，避免批量抓数过久/限流
```
生成 `markets_generated.py` 后，`./run.sh train`（不带 `--symbols`）即默认训练这个 381 品种清单。

## 与主项目的关系 / 回嵌

- 主项目 `Trading-model-web` 通过 `app.py` 的 `/iching` 路由调用 `iching_predict.predict_all(demo=False)`。
- 本项目的 `iching_predict.py` 与主项目保持**同样的函数名与签名**（`predict_all`、`_predict_one`、`SYMBOLS`、`NAME_OF`），
  优化完成后可整体复制回主项目替换，或打包成子包供其依赖，从而无缝回嵌。
- 数据抓取已用本项目 `fetch_data.py` 解耦（原 `exp_iching_seq.load_prices` 依赖主项目 `LSTM_prediction.fetch_data`，
  此处已改为独立版）。
