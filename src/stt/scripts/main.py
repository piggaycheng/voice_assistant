import os
import json
import asyncio
from collections import deque
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI(title="Faster-Whisper STT & Ollama Voice Assistant Server")

# 全域 Whisper 模型實例
model: WhisperModel = None

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.035"))  # 基礎聲音能量門檻 (預設提高至 0.035 避免雜音誤觸)
SILENCE_DURATION_SEC = float(os.getenv("SILENCE_DURATION", "0.8"))  # 停頓多久視為說話結束 (秒)
MIN_SPEECH_DURATION_SEC = float(os.getenv("MIN_SPEECH_DURATION", "0.5")) # 最小發話長度 (小於此長度視為雜音忽略)
MAX_BUFFER_SEC = 20.0  # 單次發話最大上限長度 (秒)
PRE_ROLL_CHUNKS = 6    # 前置音訊緩衝 (約 300ms，保留開頭發音)
CONSECUTIVE_FRAMES_TRIGGER = int(os.getenv("CONSECUTIVE_FRAMES", "3")) # 需連續 3 幀 (150ms) 達標才觸發

# Ollama 串接設定
ENABLE_LLM = os.getenv("ENABLE_LLM", "true").lower() in ("true", "1", "yes")
LLM_THINK = os.getenv("LLM_THINK", "false").lower() in ("true", "1", "yes")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "你是一個親切的繁體中文語音助理，回答請簡明扼要，使用口語自然流暢的口吻。"
)

@app.on_event("startup")
def load_whisper_model():
    global model
    model_size = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT", "/app/models")

    os.makedirs(download_root, exist_ok=True)

    print("=" * 50)
    print(" 🎙️ 正在載入 faster-whisper 模型...")
    print(f" - Whisper 模型: {model_size} ({device}, {compute_type})")
    print(f" - 儲存路徑: {download_root}")
    print(f" - LLM 串接: {'開啟' if ENABLE_LLM else '關閉'}")
    if ENABLE_LLM:
        print(f" - Ollama 位址: {OLLAMA_BASE_URL}")
        print(f" - Ollama 模型: {OLLAMA_MODEL}")
        print(f" - 思考模式 (Think): {'開啟' if LLM_THINK else '關閉 (No Think 模式，極速秒回)'}")
    print("=" * 50)

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
        "llm_think": LLM_THINK,
        "llm_model": OLLAMA_MODEL
    }

async def stream_ollama_chat(websocket: WebSocket, history: list, user_text: str):
    """將使用者文字送往 Ollama 並串流回傳回答"""
    if not ENABLE_LLM:
        return

    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}] + list(history)[-8:]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "think": LLM_THINK,
        "stream": True
    }

    print(f"[LLM] 正在請求 {OLLAMA_MODEL} 生成回覆...")
    await websocket.send_json({"type": "llm_start"})
    full_response = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    err_msg = f"Ollama 回應狀態碼異常: {response.status_code}"
                    print(f"[LLM 錯誤] {err_msg}")
                    await websocket.send_json({"type": "llm_error", "error": err_msg})
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line)
                        msg_chunk = chunk_data.get("message", {}).get("content", "")
                        if msg_chunk:
                            full_response.append(msg_chunk)
                            await websocket.send_json({"type": "llm_chunk", "chunk": msg_chunk})
                        if chunk_data.get("done", False):
                            break
                    except Exception:
                        pass

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
    max_val = np.max(np.abs(full_audio))
    if max_val > 0.01:
        full_audio = full_audio / max_val * 0.95

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
            initial_prompt="繁體中文，日常語音助理對話。",
            vad_filter=False
        )
    )
    text = "".join([s.text for s in segments]).strip()

    if text:
        print(f"[STT 結果] {text} (語言: {info.language}, 音訊長度: {audio_duration}s)")
        await websocket.send_json({
            "type": "result",
            "text": text,
            "language": info.language,
            "duration": audio_duration
        })

        # 串接 Ollama 生成回答
        await stream_ollama_chat(websocket, history, text)
    else:
        print("[STT] 未辨識出有效文字")
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
    noise_floor = 0.01

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

                rms = float(np.sqrt(np.mean(chunk**2)))

                if not is_speaking:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms

                active_threshold = max(SILENCE_THRESHOLD, noise_floor * 2.2)

                if rms > active_threshold:
                    consecutive_voice_frames += 1

                    if not is_speaking:
                        if consecutive_voice_frames >= CONSECUTIVE_FRAMES_TRIGGER:
                            is_speaking = True
                            print(f"[VAD] 偵測到人聲 (RMS={rms:.4f}, 底噪={noise_floor:.4f})，開始錄音...")
                            await websocket.send_json({"type": "status", "status": "listening"})
                            audio_buffer.extend(list(pre_roll_buffer))
                            audio_buffer.append(chunk)
                        else:
                            pre_roll_buffer.append(chunk)
                    else:
                        audio_buffer.append(chunk)
                        silence_samples = 0
                else:
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
                                print(f"[VAD] 聲音過短 ({len(full_audio)/SAMPLE_RATE:.2f}s)，判定為雜音已忽略")

                            audio_buffer = []
                            pre_roll_buffer.clear()
                            is_speaking = False
                            silence_samples = 0
                            await websocket.send_json({"type": "status", "status": "ready"})

                # 超過最大長度限制 -> 強制處理語音
                if is_speaking and sum(len(c) for c in audio_buffer) >= max_buffer_samples:
                    full_audio = np.concatenate(audio_buffer)
                    await process_transcription(full_audio, websocket, conversation_history)

                    audio_buffer = []
                    pre_roll_buffer.clear()
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
