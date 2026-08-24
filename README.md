# Voice Assistant

即時語音對話助理系統（STT + LLM + TTS + Client）。

## 服務架構與端口

- **LLM (Ollama)**: `11434`
- **OpenHarness Bridge**: `8010`
- **Hermes Agent API**: `8642`
- **TTS (Kokoro-82M)**: `8001`
- **STT (Faster-Whisper + VAD)**: `8000` (WebSocket: `ws://localhost:8000/ws`)
- **Client**: 本地 Python 音訊客戶端（麥克風錄音 / 喇叭播放）

---

## 1. 啟動後端服務

首次啟動先建立 Hermes 與 STT 共用的 API key 設定：

```bash
cp src/hermes/.env.example src/hermes/.env
# 編輯 src/hermes/.env，將 HERMES_API_KEY 換成至少 16 字元的長隨機字串
```

```bash
# 啟動全部 Docker 容器 (LLM / Hermes / TTS / STT)
docker compose -f src/llm/docker-compose.yml up -d && \
docker compose --env-file src/hermes/.env -f src/hermes/docker-compose.yml up -d && \
docker compose -f src/tts/docker-compose.yml up -d && \
docker compose --env-file src/hermes/.env -f src/stt/docker-compose.yml up -d

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

### Hermes Agent

Hermes 使用官方 `nousresearch/hermes-agent:latest` image，並直接提供 OpenAI 相容 API，不需要額外 Bridge。若尚未建立本機 API key 設定：

```bash
cp src/hermes/.env.example src/hermes/.env
# 編輯 src/hermes/.env，將 HERMES_API_KEY 換成至少 16 字元的長隨機字串
docker compose --env-file src/hermes/.env -f src/hermes/docker-compose.yml up -d
```

健康檢查與模型查詢：

```bash
curl http://localhost:8642/health

set -a
source src/hermes/.env
set +a
curl http://localhost:8642/v1/models \
	-H "Authorization: Bearer ${HERMES_API_KEY}"
```

測試對話：

```bash
curl http://localhost:8642/v1/chat/completions \
	-H "Authorization: Bearer ${HERMES_API_KEY}" \
	-H "Content-Type: application/json" \
	-d '{"model":"hermes-agent","messages":[{"role":"user","content":"只回答：HERMES-OK"}]}'
```

`src/llm/docker-compose.yml` 將 Ollama context 設為 `65536`，並與 `src/hermes/config.yaml` 的 `model.context_length` 保持一致，以符合 Hermes 最低 `64000` 的要求。首次啟動會將此模板複製到 `hermes_data` named volume 內的 `/opt/data/config.yaml`；既有執行期設定不會被模板覆蓋。Hermes API 可執行終端與檔案工具，請勿將 `8642` 暴露到不受信任的網路。

#### 切換 Hermes 預設模型

先確認目標模型已存在於 Ollama：

```bash
docker exec voice-assistant-llm ollama list
```

使用 Hermes CLI 修改 `hermes_data` volume 內的執行期設定，再重啟服務：

```bash
docker exec voice-assistant-hermes \
	hermes config set model.default MODEL_NAME
docker restart voice-assistant-hermes
docker exec voice-assistant-hermes \
	hermes config get model.default
```

同時將 `src/hermes/config.yaml` 的 `model.default` 改成相同名稱，確保日後建立全新 volume 時仍使用該模型。只修改專案內的模板不會影響已存在的 volume。`GET /v1/models` 固定顯示 Hermes API alias `hermes-agent`，應使用 `hermes config get model.default` 檢查底層預設模型。

目前 temperature `0.1` 的自訂模型可使用：

```bash
docker exec voice-assistant-hermes \
	hermes config set model.default qwen3.5-4b-temp-0.1
docker restart voice-assistant-hermes
```

STT 會將辨識文字送至 Hermes 的 `/v1/chat/completions`，並把 Hermes 的串流回答轉送給 Client/TTS。啟動 STT 時必須透過 `--env-file src/hermes/.env` 傳入相同的 `HERMES_API_KEY`。

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
docker logs -f voice-assistant-hermes # Hermes 日誌
```

### 停止全部服務
```bash
docker compose -f src/stt/docker-compose.yml down && \
docker compose -f src/tts/docker-compose.yml down && \
docker compose -f src/llm/docker-compose.yml down && \
docker compose --env-file src/hermes/.env -f src/hermes/docker-compose.yml down
```
