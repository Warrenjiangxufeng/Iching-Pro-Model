#!/bin/bash
# 易经独立模型 便捷运行入口（复用主项目 venv，无需重复安装依赖）
# 用法： ./run.sh train|predict|eval [额外参数...]
#   示例： ./run.sh train --demo --symbols FG0,SA0,M0
#         ./run.sh predict --model models/iching.pt
set -e

# 复用主项目已装好的 venv（含 numpy/pandas/torch/sklearn/akshare）
MAIN_VENV="/Users/jiangxufeng1/Trading-model-web/venv/bin/python"
LOCAL_VENV="$(dirname "$0")/venv/bin/python"

if [ -x "$LOCAL_VENV" ]; then
    PY="$LOCAL_VENV"
elif [ -x "$MAIN_VENV" ]; then
    PY="$MAIN_VENV"
else
    PY="python3"
fi

cd "$(dirname "$0")"
exec "$PY" main.py "$@"
