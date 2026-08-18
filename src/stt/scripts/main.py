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
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.035"))  # 基礎聲音能量門檻 (預設提高至 0.035 避免雜音誤觸)
SILENCE_DURATION_SEC = float(os.getenv("SILENCE_DURATION", "0.8"))  # 停頓多久視為說話結束 (秒)
MIN_SPEECH_DURATION_SEC = float(os.getenv("MIN_SPEECH_DURATION", "0.5")) # 最小發話長度 (小於此長度視為雜音忽略)
MAX_BUFFER_SEC = 20.0  # 單次發話最大上限長度 (秒)
PRE_ROLL_CHUNKS = 6    # 前置音訊緩衝 (約 300ms，保留開頭發音)
CONSECUTIVE_FRAMES_TRIGGER = int(os.getenv("CONSECUTIVE_FRAMES", "3")) # 需連續 3 幀 (150ms) 達標才觸發，過濾敲鍵盤/點滑鼠等瞬間雜音

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
    consecutive_voice_frames = 0
    silence_samples = 0
    noise_floor = 0.01  # 動態環境底噪估計值

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

                # 計算當前 chunk 的 RMS 能量
                rms = float(np.sqrt(np.mean(chunk**2)))

                # 動態調整環境底噪（未說話時平滑更新）
                if not is_speaking:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms

                # 動態發話門檻：取固定門檻與環境底噪加成之最大值
                active_threshold = max(SILENCE_THRESHOLD, noise_floor * 2.2)

                if rms > active_threshold:
                    consecutive_voice_frames += 1
                    
                    if not is_speaking:
                        # 需連續 N 幀超過門檻才視為真正開始說話（過濾短暫爆音、鍵盤聲）
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

                        # 靜音超過設定秒數，判定一句話結束
                        if silence_samples >= silence_samples_limit:
                            full_audio = np.concatenate(audio_buffer)
                            
                            # 若說話時間太短（如 < 0.5s），視為誤觸雜音直接忽略
                            if len(full_audio) >= min_speech_samples:
                                # 音量正規化
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
                                    print(f"[STT 結果] {text} (語言: {info.language}, 音訊長度: {len(full_audio)/SAMPLE_RATE:.2f}s)")
                                    await websocket.send_json({
                                        "type": "result",
                                        "text": text,
                                        "language": info.language,
                                        "duration": round(len(full_audio) / SAMPLE_RATE, 2)
                                    })
                                else:
                                    print("[STT] 未辨識出有效文字")
                                    await websocket.send_json({"type": "status", "status": "empty"})
                            else:
                                print(f"[VAD] 聲音長度過短 ({len(full_audio)/SAMPLE_RATE:.2f}s)，判定為雜音已忽略")

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
