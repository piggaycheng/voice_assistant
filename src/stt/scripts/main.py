import os
import json
import asyncio
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI(title="Faster-Whisper STT WebSocket Server")

# 全域 Whisper 模型實例
model: WhisperModel = None

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.015"))  # 聲音能量 (RMS) 門檻
SILENCE_DURATION_SEC = float(os.getenv("SILENCE_DURATION", "0.8"))  # 停頓多久視為說話結束 (秒)
MIN_SPEECH_DURATION_SEC = float(os.getenv("MIN_SPEECH_DURATION", "0.3")) # 最小發話長度 (秒)
MAX_BUFFER_SEC = 20.0  # 單次發話最大上限長度 (秒)
PRE_ROLL_CHUNKS = 6    # 前置音訊緩衝 (約 300ms，避免吃掉發音開頭的輕聲/子音)

@app.on_event("startup")
def load_whisper_model():
    global model
    model_size = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT", "/app/models")

    os.makedirs(download_root, exist_ok=True)

    print("=" * 50)
    print(" 正在載入 faster-whisper 模型...")
    print(f" - 模型大小: {model_size}")
    print(f" - 運算設備: {device}")
    print(f" - 數值精度: {compute_type}")
    print(f" - 儲存路徑: {download_root}")
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
    return {"status": "running", "service": "faster-whisper-stt-websocket"}

@app.websocket("/ws")
async def websocket_stt_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[WebSocket] 客戶端連線成功: {client_info}")

    audio_buffer = []
    pre_roll_buffer = deque(maxlen=PRE_ROLL_CHUNKS)
    is_speaking = False
    silence_samples = 0
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

                # 計算能量 RMS
                rms = np.sqrt(np.mean(chunk**2))

                if rms > SILENCE_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        print("[VAD] 偵測到使用者開始說話...")
                        await websocket.send_json({"type": "status", "status": "listening"})
                        # 將偵測到說話前 300ms 的音訊補入開頭，避免吃字
                        audio_buffer.extend(list(pre_roll_buffer))
                    
                    audio_buffer.append(chunk)
                    silence_samples = 0
                else:
                    if not is_speaking:
                        # 未說話時，維護前置環狀緩衝區
                        pre_roll_buffer.append(chunk)
                    else:
                        # 正在說話但遇到短暫停頓
                        audio_buffer.append(chunk)
                        silence_samples += chunk_samples

                        # 靜音超過設定秒數，判定為一句話結束，進行辨識
                        if silence_samples >= silence_samples_limit:
                            full_audio = np.concatenate(audio_buffer)
                            
                            if len(full_audio) >= min_speech_samples:
                                # 音量正規化 (避免聲音過小辨識失真)
                                max_val = np.max(np.abs(full_audio))
                                if max_val > 0.01:
                                    full_audio = full_audio / max_val * 0.95

                                print(f"[STT] 說話結束，開始辨識 ({len(full_audio)/SAMPLE_RATE:.2f} 秒音訊)...")
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
                                        vad_filter=True,
                                        vad_parameters=dict(min_silence_duration_ms=500)
                                    )
                                )
                                text = "".join([s.text for s in segments]).strip()

                                if text:
                                    print(f"[STT 結果] {text} (語言: {info.language}, 耗時音訊: {len(full_audio)/SAMPLE_RATE:.2f}s)")
                                    await websocket.send_json({
                                        "type": "result",
                                        "text": text,
                                        "language": info.language,
                                        "duration": round(len(full_audio) / SAMPLE_RATE, 2)
                                    })
                                else:
                                    print("[STT] 無法辨識出文字")
                                    await websocket.send_json({"type": "status", "status": "empty"})

                            # 重置狀態
                            audio_buffer = []
                            pre_roll_buffer.clear()
                            is_speaking = False
                            silence_samples = 0
                            await websocket.send_json({"type": "status", "status": "ready"})

                # 若單次發話長度超過上限，強制辨識
                if is_speaking and sum(len(c) for c in audio_buffer) >= max_buffer_samples:
                    full_audio = np.concatenate(audio_buffer)
                    max_val = np.max(np.abs(full_audio))
                    if max_val > 0.01:
                        full_audio = full_audio / max_val * 0.95

                    print(f"[STT] 發話長度達到上限，強制辨識...")
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
                            vad_filter=True
                        )
                    )
                    text = "".join([s.text for s in segments]).strip()
                    if text:
                        await websocket.send_json({
                            "type": "result",
                            "text": text,
                            "language": info.language,
                            "duration": round(len(full_audio) / SAMPLE_RATE, 2)
                        })
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
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[WebSocket] 客戶端斷開連線: {client_info}")
    except Exception as e:
        print(f"[WebSocket] 錯誤發生: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
