#!/bin/bash
set -e

VENV_DIR="/app/.venv"
STAMP_FILE="$VENV_DIR/.installed"
PROVIDER_STAMP_FILE="/root/.openharness/.ollama-provider-initialized"

if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "=========================================="
    echo " [OpenHarness] 正在初始化 .venv..."
    echo "=========================================="
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
fi

if [ -f "/app/requirements.txt" ]; then
    if [ ! -f "$STAMP_FILE" ] || [ "/app/requirements.txt" -nt "$STAMP_FILE" ]; then
        echo "=========================================="
        echo " [OpenHarness] 正在同步套件..."
        echo "=========================================="
        "$VENV_DIR/bin/pip" install -r /app/requirements.txt
        touch "$STAMP_FILE"
        echo "=========================================="
        echo " [OpenHarness] 套件同步完成！"
        echo "=========================================="
    fi
fi

if [ "${OPENHARNESS_AUTO_CONFIGURE_OLLAMA:-true}" = "true" ] && [ ! -f "$PROVIDER_STAMP_FILE" ]; then
    echo "=========================================="
    echo " [OpenHarness] 正在設定 Ollama provider..."
    echo "=========================================="
    mkdir -p /root/.openharness
    oh provider add "${OPENHARNESS_OLLAMA_PROFILE:-ollama}" \
        --label "${OPENHARNESS_OLLAMA_LABEL:-Ollama}" \
        --provider Ollama \
        --api-format openai \
        --auth-source openai_api_key \
        --model "${OPENHARNESS_OLLAMA_MODEL:-qwen3.5:4b}" \
        --base-url "${OPENHARNESS_OLLAMA_BASE_URL:-http://host.docker.internal:11434/v1}"
    oh provider use "${OPENHARNESS_OLLAMA_PROFILE:-ollama}"
    touch "$PROVIDER_STAMP_FILE"
    echo " [OpenHarness] Ollama provider 設定完成！"
fi

exec "$@"