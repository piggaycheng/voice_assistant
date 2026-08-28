#!/bin/bash
set -e

VENV_DIR="/app/.venv"
STAMP_FILE="$VENV_DIR/.installed"
KWS_MODEL_NAME="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KWS_MODEL_DIR="${KWS_MODEL_DIR:-/app/models/$KWS_MODEL_NAME}"
KWS_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$KWS_MODEL_NAME.tar.bz2"
WAKE_WORD="${WAKE_WORD:-嘿小奧}"

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

if [ ! -f "$KWS_MODEL_DIR/tokens.txt" ]; then
    echo "=========================================="
    echo " [STT] 正在下載 sherpa-onnx KWS 模型..."
    echo "=========================================="
    KWS_ARCHIVE="/tmp/$KWS_MODEL_NAME.tar.bz2"
    mkdir -p "$(dirname "$KWS_MODEL_DIR")"
    curl -fL --retry 3 -o "$KWS_ARCHIVE" "$KWS_MODEL_URL"
    tar -xjf "$KWS_ARCHIVE" -C "$(dirname "$KWS_MODEL_DIR")"
    rm -f "$KWS_ARCHIVE"
fi

KEYWORDS_RAW_FILE="$KWS_MODEL_DIR/keywords_raw.txt"
KEYWORDS_FILE="${KWS_KEYWORDS_FILE:-$KWS_MODEL_DIR/keywords.txt}"
EXPECTED_KEYWORD="$WAKE_WORD :${KWS_KEYWORDS_SCORE:-1.5} #${KWS_KEYWORDS_THRESHOLD:-0.25} @$WAKE_WORD"
if [ ! -f "$KEYWORDS_RAW_FILE" ] || [ "$(cat "$KEYWORDS_RAW_FILE")" != "$EXPECTED_KEYWORD" ] || [ ! -f "$KEYWORDS_FILE" ]; then
    printf '%s\n' "$EXPECTED_KEYWORD" > "$KEYWORDS_RAW_FILE"
    "$VENV_DIR/bin/sherpa-onnx-cli" text2token \
        --tokens "$KWS_MODEL_DIR/tokens.txt" \
        --tokens-type phone+ppinyin \
        --lexicon "$KWS_MODEL_DIR/en.phone" \
        "$KEYWORDS_RAW_FILE" "$KEYWORDS_FILE"
    echo " [STT] 喚醒詞已設定為：$WAKE_WORD"
fi

# 執行容器後續指令
exec "$@"
