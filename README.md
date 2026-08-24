# Voice Assistant

即時語音對話助理系統（STT + LLM + TTS + Client）。

## 服務架構與端口

- **LLM (Ollama)**: `11434`
- **OpenHarness Bridge**: `8010`
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

### OpenHarness Bridge

首次啟動時，容器會自動建立並啟用 Ollama provider。若 Ollama 執行於主機或本專案的 LLM 容器並對外映射 `11434`，預設 provider base URL 為：

```text
http://host.docker.internal:11434/v1
```

可透過 `OLLAMA_MODEL` 與 `OLLAMA_BASE_URL` 覆寫模型和位址。自動設定結果會保存在 `src/openharness/config`，後續啟動不會重複執行；如需改用其他 provider，將 `OPENHARNESS_AUTO_CONFIGURE_OLLAMA` 設為 `false`，再於容器內執行 `oh setup`。

啟動 Bridge：

```bash
export OPENHARNESS_BRIDGE_KEY='請替換為長隨機字串'
docker compose -f src/openharness/docker-compose.yml up -d
```

測試查詢：

```bash
curl http://localhost:8010/query \
	-H "Authorization: Bearer ${OPENHARNESS_BRIDGE_KEY}" \
	-H "Content-Type: application/json" \
	-d '{"prompt":"搜尋今天的重要 AI 新聞，並以繁體中文簡短回答"}'
```

Bridge 程式、`.venv` 與 OpenHarness 設定皆由 `src/openharness` 掛載進容器；provider 設定會保存在 `src/openharness/config`，且不會納入 Git。

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
