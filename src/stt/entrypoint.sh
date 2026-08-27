#!/bin/bash
set -e

VENV_DIR="/app/.venv"
STAMP_FILE="$VENV_DIR/.installed"
CTRANSLATE2_STAMP_FILE="$VENV_DIR/.ctranslate2-cuda-installed"

if [ -d "$VENV_DIR" ] && ! "$VENV_DIR/bin/python3" -c "import pip" >/dev/null 2>&1; then
    echo " [STT] 現有 .venv 與容器 Python 不相容，正在重建..."
    rm -rf "$VENV_DIR"
fi

# 若 .venv 不存在，初始化虛擬環境
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "=========================================="
    echo " [STT] 檢測到尚未建立 .venv，正在初始化..."
    echo "=========================================="
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
fi

# 若 requirements.txt 變更過或尚未安裝，自動同步套件
if [ -f "/app/requirements.txt" ]; then
    if [ ! -f "$STAMP_FILE" ] || [ "/app/requirements.txt" -nt "$STAMP_FILE" ]; then
        echo "=========================================="
        echo " [STT] 偵測到 requirements.txt 更新，正在同步套件..."
        echo "=========================================="
        "$VENV_DIR/bin/pip" install -r /app/requirements.txt
        touch "$STAMP_FILE"
        echo "=========================================="
        echo " [STT] 套件同步完成！"
        echo "=========================================="
    fi
fi

CTRANSLATE2_WHEEL=$(find /opt/ctranslate2-wheels -name 'ctranslate2-*.whl' -print -quit)
if [ -n "$CTRANSLATE2_WHEEL" ] && { [ ! -f "$CTRANSLATE2_STAMP_FILE" ] || [ "$CTRANSLATE2_WHEEL" -nt "$CTRANSLATE2_STAMP_FILE" ]; }; then
    echo "=========================================="
    echo " [STT] 正在安裝 CUDA 版 CTranslate2..."
    echo "=========================================="
    "$VENV_DIR/bin/pip" install --force-reinstall --no-deps "$CTRANSLATE2_WHEEL"
    touch "$CTRANSLATE2_STAMP_FILE"
fi

# 執行容器後續指令
exec "$@"
