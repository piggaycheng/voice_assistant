# Voice Assistant

即時語音對話助理系統（STT + LLM + TTS + Client）。

## 服務架構與端口

- **LLM (Ollama)**: `11434`
- **TTS (Kokoro-82M)**: `8001`
- **STT (Faster-Whisper + VAD)**: `8000` (WebSocket: `ws://localhost:8000/ws`)
- **Client**: 本地 Python 音訊客戶端（麥克風錄音 / 喇叭播放）

---

## 1. 啟動後端服務

```bash
# 啟動全部 Docker 容器 (LLM / TTS / STT)
docker compose -f src/llm/docker-compose.yml up -d && \
docker compose -f src/tts/docker-compose.yml up -d && \
docker compose -f src/stt/docker-compose.yml up -d

# (首次使用) 下載預設 LLM 模型
docker exec -it voice-assistant-llm ollama pull qwen3.5:4b
```

---

## 2. 啟動語音客戶端

```bash
cd src/clients

# 啟用虛擬環境（若尚未建立可執行 python3 -m venv .venv && pip install -r requirements.txt）
source .venv/bin/activate

# 啟動語音對話
python3 client.py
```

---

## 3. 常用管理指令

### 查看運行狀態
```bash
docker ps --filter "name=voice-assistant"
```

### 查看日誌
```bash
docker logs -f voice-assistant-stt   # STT 日誌
docker logs -f voice-assistant-tts   # TTS 日誌
docker logs -f voice-assistant-llm   # LLM 日誌
```

### 停止全部服務
```bash
docker compose -f src/stt/docker-compose.yml down && \
docker compose -f src/tts/docker-compose.yml down && \
docker compose -f src/llm/docker-compose.yml down
```
