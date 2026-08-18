#!/bin/bash
set -e

VENV_DIR="/app/.venv"

# 若 .venv 不存在或未初始化，自動建立虛擬環境並安裝 requirements.txt
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "=========================================="
    echo " [STT] 檢測到尚未建立 .venv，正在初始化..."
    echo "=========================================="
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
    
    if [ -f "/app/requirements.txt" ]; then
        echo "=========================================="
        echo " [STT] 正在安裝 requirements.txt 套件..."
        echo "=========================================="
        "$VENV_DIR/bin/pip" install -r /app/requirements.txt
    fi
    echo "=========================================="
    echo " [STT] 虛擬環境建立完成！已保存於 host。"
    echo "=========================================="
fi

# 執行容器後續指令
exec "$@"
