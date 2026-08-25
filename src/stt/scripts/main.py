import os
import json
import asyncio
from collections import deque
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
import uvicorn

app = FastAPI(title="Faster-Whisper STT & Hermes Voice Assistant Server")

# 全域模型實例
model: WhisperModel = None
vad_model = None

SAMPLE_RATE = 16000
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.35"))          # Silero 人聲判定門檻 (0.35 靈敏捕捉清輔音/弱起音)
VAD_NEG_THRESHOLD = float(os.getenv("VAD_NEG_THRESHOLD", "0.20")) # 靜音/非人聲判定門檻 (0.20 避免句中氣音被過早切斷)
SILENCE_DURATION_SEC = float(os.getenv("SILENCE_DURATION", "0.8")) # 停頓多久視為說話結束 (秒)
MIN_SPEECH_DURATION_SEC = float(os.getenv("MIN_SPEECH_DURATION", "0.4")) # 最小發話長度 (小於此長度視為雜音忽略)
MAX_BUFFER_SEC = 20.0  # 單次發話最大上限長度 (秒)
PRE_ROLL_CHUNKS = int(os.getenv("PRE_ROLL_CHUNKS", "14")) # 前置音訊緩衝 (14 chunks = 700ms，完整保留開頭發音與自然聲學上下文)
CONSECUTIVE_FRAMES_TRIGGER = int(os.getenv("CONSECUTIVE_FRAMES", "1")) # 1 幀達標即觸發錄音，零延遲響應

WHISPER_INITIAL_PROMPT = "繁體中文日常語音對話。公司名稱：盟立、盟立自動化、盟立集團、MiRLE。"
COMPANY_ALIASES = {
    "夢立": "盟立",
    "猛力": "盟立",
    "盟力": "盟立",
}

def normalize_company_aliases(text: str) -> str:
    for alias, canonical_name in COMPANY_ALIASES.items():
        text = text.replace(alias, canonical_name)
    return text

class SileroVADStream:
    """即時串流 Silero VAD (基於 ONNX)，維持每個連線獨立的 hidden state 與 context"""
    def __init__(self, session):
        self.session = session
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.remainder = np.array([], dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> list:
        """處理傳入的音訊 chunk，切分為 512-sample (32ms) 區塊並計算人聲機率"""
        if len(self.remainder) > 0:
            audio = np.concatenate([self.remainder, audio_chunk])
        else:
            audio = audio_chunk

        num_samples = 512
        probs = []

        while len(audio) >= num_samples:
            chunk_512 = audio[:num_samples]
            audio = audio[num_samples:]

            # shape (1, 576) = 64 (context) + 512 (samples)
            inp = np.concatenate([self.context, chunk_512.reshape(1, num_samples)], axis=1)
            self.context = chunk_512[-64:].reshape(1, 64)

            out, hn, cn = self.session.run(None, {"input": inp, "h": self.h, "c": self.c})
            self.h = hn
            self.c = cn
            probs.append(float(out[0]))

        self.remainder = audio
        return probs

# Hermes Agent 串接設定
ENABLE_LLM = os.getenv("ENABLE_LLM", "true").lower() in ("true", "1", "yes")
HERMES_BASE_URL = os.getenv("HERMES_BASE_URL", "http://host.docker.internal:8642")
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")
HERMES_MODEL = os.getenv("HERMES_MODEL", "hermes-agent")
HERMES_TIMEOUT = float(os.getenv("HERMES_TIMEOUT", "180"))
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "你是一個親切的繁體中文語音助理。凡是關於盟立、盟立自動化、盟立集團或 MiRLE 的問題，"
    "回答前必須先使用檔案工具搜尋 /wiki_data/mirle_official_wiki，並讀取相關檔案；"
    "不得依模型記憶直接回答。若搜尋後仍找不到資料，請明確回答知識庫中沒有相關資訊。"
    "回答限制在 1 至 2 句、30 字以內，只提供最重要資訊，不要重述問題，使用口語自然流暢的口吻。"
    "不得提及資料來源、檔案名稱、路徑、搜尋過程或工具使用情形，也不要附上引用或參考連結；"
    "只直接回答結果。"
)

@app.on_event("startup")
def load_whisper_model():
    global model, vad_model
    model_size = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT", "/app/models")

    os.makedirs(download_root, exist_ok=True)

    print("=" * 50)
    print(" 🎙️ 正在載入 faster-whisper & Silero VAD 模型...")
    print(f" - Whisper 模型: {model_size} ({device}, {compute_type})")
    print(f" - 儲存路徑: {download_root}")
    print(f" - Silero VAD 門檻: 人聲={VAD_THRESHOLD}, 靜音={VAD_NEG_THRESHOLD}")
    print(f" - LLM 串接: {'開啟' if ENABLE_LLM else '關閉'}")
    if ENABLE_LLM:
        if len(HERMES_API_KEY) < 16:
            raise RuntimeError("ENABLE_LLM=true 時，HERMES_API_KEY 必須至少 16 個字元")
        print(f" - Hermes 位址: {HERMES_BASE_URL}")
        print(f" - Hermes 模型: {HERMES_MODEL}")
    print("=" * 50)

    # 載入 Silero VAD (ONNX)
    vad_model = get_vad_model()
    print("Silero VAD 模型載入完成！")

    # 載入 Whisper 模型
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root
    )
    print("Whisper 模型載入完成！伺服器已就緒。")

@app.get("/")
def index():
    return {
        "status": "running",
        "service": "faster-whisper-stt-websocket",
        "llm_enabled": ENABLE_LLM,
        "llm_backend": "hermes",
        "llm_model": HERMES_MODEL
    }

async def stream_hermes_chat(websocket: WebSocket, history: list, user_text: str):
    """將使用者文字送往 Hermes Agent 並串流回傳回答"""
    if not ENABLE_LLM:
        return

    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}] + list(history)[-8:]

    payload = {
        "model": HERMES_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 8192
    }
    headers = {"Authorization": f"Bearer {HERMES_API_KEY}"}

    print(f"[LLM] 正在請求 Hermes Agent ({HERMES_MODEL}) 生成回覆...")
    await websocket.send_json({"type": "llm_start"})
    full_response = []

    try:
        async with httpx.AsyncClient(timeout=HERMES_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{HERMES_BASE_URL.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    err_msg = f"Hermes 回應狀態碼異常: {response.status_code}"
                    print(f"[LLM 錯誤] {err_msg}")
                    await websocket.send_json({"type": "llm_error", "error": err_msg})
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        choices = chunk_data.get("choices", [])
                        msg_chunk = choices[0].get("delta", {}).get("content", "") if choices else ""
                        if msg_chunk:
                            full_response.append(msg_chunk)
                            await websocket.send_json({"type": "llm_chunk", "chunk": msg_chunk})
                    except (json.JSONDecodeError, AttributeError, IndexError):
                        continue

        complete_reply = "".join(full_response).strip()
        if complete_reply:
            history.append({"role": "assistant", "content": complete_reply})
            print(f"[LLM 回覆完成] {complete_reply}")
        await websocket.send_json({"type": "llm_end", "text": complete_reply})

    except Exception as e:
        print(f"[LLM 連線失敗] {type(e).__name__}: {e}")
        await websocket.send_json({"type": "llm_error", "error": f"{type(e).__name__}: {e}"})

async def process_transcription(full_audio: np.ndarray, websocket: WebSocket, history: list):
    """執行 STT 語音辨識並串接 LLM"""
    max_val = float(np.max(np.abs(full_audio)))
    
    # 若最大音量極小（接近純靜音），直接忽略避免除以零
    if max_val < 0.01:
        print(f"[STT 忽略] 音量極小 (Max={max_val:.4f})，視為無效音訊")
        await websocket.send_json({"type": "status", "status": "empty"})
        return

    # 音量適度正規化（避免爆音與過度放大）
    full_audio = full_audio / max_val * 0.90

    audio_duration = round(len(full_audio) / SAMPLE_RATE, 2)
    print(f"[STT] 說話結束，開始辨識 ({audio_duration} 秒音訊)...")
    await websocket.send_json({"type": "status", "status": "transcribing"})

    loop = asyncio.get_event_loop()
    segments, info = await loop.run_in_executor(
        None,
        lambda: model.transcribe(
            full_audio,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            language="zh",
            initial_prompt=WHISPER_INITIAL_PROMPT,
            vad_filter=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )
    )

    valid_segments = []
    for s in segments:
        # 過濾高機率非人聲或低置信度片段（從源頭自動防範各類字幕與雜音幻覺）
        if s.no_speech_prob > 0.6:
            print(f"[STT 忽略] 判定為非人聲 (no_speech_prob={s.no_speech_prob:.2f}): '{s.text}'")
            continue
        if s.avg_logprob < -1.0:
            print(f"[STT 忽略] 置信度過低 (avg_logprob={s.avg_logprob:.2f}): '{s.text}'")
            continue
        valid_segments.append(s.text.strip())

    raw_text = "".join(valid_segments).strip()
    text = normalize_company_aliases(raw_text)

    if text:
        if text != raw_text:
            print(f"[STT 名稱修正] {raw_text} -> {text}")
        print(f"[STT 結果] {text} (語言: {info.language}, 音訊長度: {audio_duration}s)")
        await websocket.send_json({
            "type": "result",
            "text": text,
            "language": info.language,
            "duration": audio_duration
        })

        # 串接 Ollama 生成回答
        await stream_hermes_chat(websocket, history, text)
    else:
        print("[STT] 未辨識出有效文字（已過濾雜音/幻覺）")
        await websocket.send_json({"type": "status", "status": "empty"})

@app.websocket("/ws")
async def websocket_stt_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[WebSocket] 客戶端連線成功: {client_info}")

    conversation_history = deque(maxlen=10)
    audio_buffer = []
    pre_roll_buffer = deque(maxlen=PRE_ROLL_CHUNKS)
    is_speaking = False
    consecutive_voice_frames = 0
    silence_samples = 0

    # 建立該連線專屬的 Silero VAD 串流實例（維持獨立 hidden states）
    vad_stream = SileroVADStream(vad_model.session)

    silence_samples_limit = int(SAMPLE_RATE * SILENCE_DURATION_SEC)
    min_speech_samples = int(SAMPLE_RATE * MIN_SPEECH_DURATION_SEC)
    max_buffer_samples = int(SAMPLE_RATE * MAX_BUFFER_SEC)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                chunk_samples = len(chunk)
                if chunk_samples == 0:
                    continue

                # 透過 Silero VAD 計算此 chunk 中的人聲機率
                probs = vad_stream.process_chunk(chunk)
                speech_prob = max(probs) if probs else 0.0

                if speech_prob >= VAD_THRESHOLD:
                    consecutive_voice_frames += 1

                    if not is_speaking:
                        if consecutive_voice_frames >= CONSECUTIVE_FRAMES_TRIGGER:
                            is_speaking = True
                            print(f"[Silero VAD] 偵測到人聲 (機率={speech_prob:.2f})，開始錄音...")
                            await websocket.send_json({"type": "status", "status": "listening"})
                            audio_buffer.extend(list(pre_roll_buffer))
                            pre_roll_buffer.clear()
                            audio_buffer.append(chunk)
                        else:
                            pre_roll_buffer.append(chunk)
                    else:
                        audio_buffer.append(chunk)
                        silence_samples = 0
                elif speech_prob < VAD_NEG_THRESHOLD:
                    consecutive_voice_frames = 0

                    if not is_speaking:
                        pre_roll_buffer.append(chunk)
                    else:
                        audio_buffer.append(chunk)
                        silence_samples += chunk_samples

                        # 靜音超時 -> 處理語音
                        if silence_samples >= silence_samples_limit:
                            full_audio = np.concatenate(audio_buffer)

                            if len(full_audio) >= min_speech_samples:
                                await process_transcription(full_audio, websocket, conversation_history)
                            else:
                                print(f"[Silero VAD] 聲音過短 ({len(full_audio)/SAMPLE_RATE:.2f}s)，判定為雜音已忽略")

                            audio_buffer = []
                            pre_roll_buffer.clear()
                            vad_stream.reset()
                            is_speaking = False
                            silence_samples = 0
                            await websocket.send_json({"type": "status", "status": "ready"})
                else:
                    # 介於 neg_threshold 與 threshold 之間（微弱發音或過渡）
                    if not is_speaking:
                        pre_roll_buffer.append(chunk)
                    else:
                        audio_buffer.append(chunk)
                        # 處於過渡地帶不重設也不增加靜音計數，避免微弱音被過早截斷

                # 超過最大長度限制 -> 強制處理語音
                if is_speaking and sum(len(c) for c in audio_buffer) >= max_buffer_samples:
                    full_audio = np.concatenate(audio_buffer)
                    await process_transcription(full_audio, websocket, conversation_history)

                    audio_buffer = []
                    pre_roll_buffer.clear()
                    vad_stream.reset()
                    is_speaking = False
                    silence_samples = 0
                    await websocket.send_json({"type": "status", "status": "ready"})

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("action") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif payload.get("action") == "clear_history":
                        conversation_history.clear()
                        await websocket.send_json({"type": "status", "status": "history_cleared"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[WebSocket] 客戶端斷開連線: {client_info}")
    except Exception as e:
        print(f"[WebSocket] 錯誤發生: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
