#!/bin/bash
set -e

VENV_DIR="/app/.venv"
STAMP_FILE="$VENV_DIR/.installed"

if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "=========================================="
    echo " [RAG] 正在初始化 .venv..."
    echo "=========================================="
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
fi

if [ -f "/app/requirements.txt" ]; then
    if [ ! -f "$STAMP_FILE" ] || [ "/app/requirements.txt" -nt "$STAMP_FILE" ]; then
        echo "=========================================="
        echo " [RAG] 正在同步套件..."
        echo "=========================================="
        "$VENV_DIR/bin/pip" install -r /app/requirements.txt
        touch "$STAMP_FILE"
        echo "=========================================="
        echo " [RAG] 套件同步完成！"
        echo "=========================================="
    fi
fi

exec "$@"
