# Voice Assistant

即時語音對話助理系統（STT + LLM + TTS + Client）。

## 服務架構與端口

- **LLM (Ollama，另提供 llama.cpp / vLLM Compose)**: `11434`
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

`src/llm/docker-compose.yml` 預設使用 Ollama；另可使用 `src/llm/docker-compose.llamacpp.yml` 啟動 llama.cpp CUDA server。兩者均將 context 設為 `65536`，並與 `src/hermes/config.yaml` 的 `model.context_length` 保持一致，以符合 Hermes 最低 `64000` 的要求。首次啟動會將此模板複製到 `hermes_data` named volume 內的 `/opt/data/config.yaml`；既有執行期設定不會被模板覆蓋。Hermes API 可執行終端與檔案工具，請勿將 `8642` 暴露到不受信任的網路。

#### 切換 Hermes 預設模型

llama.cpp compose 預設從 Hugging Face 下載 `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`，並將模型保存在 host 的 `src/llm/hg_models/`。若要改用其他 llama.cpp 相容模型，可覆寫 repository 與量化標籤：

```bash
LLAMA_HF_REPO=owner/model-GGUF:QUANT \
	docker compose -f src/llm/docker-compose.llamacpp.yml up -d --force-recreate
```

確認 llama.cpp 已載入模型：

```bash
curl http://localhost:11434/health
curl http://localhost:11434/v1/models
```

llama.cpp 對外的模型 alias 預設為 `qwen3.5:4b`。若同時修改 `LLAMA_ARG_ALIAS`，請使用 Hermes CLI 將執行期模型名稱改成相同值，再重啟服務：

```bash
docker exec voice-assistant-hermes \
	hermes config set model.default MODEL_NAME
docker restart voice-assistant-hermes
docker exec voice-assistant-hermes \
	hermes config get model.default
```

同時將 `src/hermes/config.yaml` 的 `model.default` 改成相同名稱，確保日後建立全新 volume 時仍使用該模型。只修改專案內的模板不會影響已存在的 volume。`GET /v1/models` 固定顯示 Hermes API alias `hermes-agent`，應使用 `hermes config get model.default` 檢查底層預設模型。

STT 會將辨識文字送至 Hermes 的 `/v1/chat/completions`，並把 Hermes 的串流回答轉送給 Client/TTS。啟動 STT 時必須透過 `--env-file src/hermes/.env` 傳入相同的 `HERMES_API_KEY`。

#### vLLM / AGX Thor 部署

專案另提供 `src/llm/docker-compose.vllm.yml`，保留與 Ollama / llama.cpp 相同的 `11434` port、OpenAI 相容 API 與 `qwen3.5:4b` alias，因此 Hermes 設定不需修改。vLLM 使用 Hugging Face 原生權重而非 GGUF，下載內容同樣保存在 host 的 `src/llm/hg_models/`。

先停止目前的 Ollama，再啟動 vLLM：

```bash
docker compose -f src/llm/docker-compose.yml down
docker compose -f src/llm/docker-compose.vllm.yml up -d
```

可透過環境變數覆寫 image、模型與 context：

```bash
VLLM_IMAGE=vllm/vllm-openai:latest \
VLLM_MODEL=Qwen/Qwen3.5-4B \
VLLM_MAX_MODEL_LEN=65536 \
	docker compose -f src/llm/docker-compose.vllm.yml up -d
```

官方 vLLM image 同時提供 `amd64` 與 `arm64` manifest。AGX Thor 應使用 JetPack 7 與 CUDA 12.8 以上相容 image；若 JetPack 版本與 `latest` 不相容，請以 `VLLM_IMAGE` 指定經該 JetPack 驗證的 ARM64 image。Qwen3.5 4B BF16 加上 64K KV cache 不適合目前的 10GB RTX 3080，此 compose 主要供 AGX Thor 使用；本機仍建議使用 llama.cpp Q4_K_M。

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
