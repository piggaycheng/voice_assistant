import os
import json
import asyncio
from collections import deque
from datetime import date

import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
import uvicorn

app = FastAPI(title="Faster-Whisper STT & RAG Voice Assistant Server")

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
ENABLE_NOISE_SUPPRESSION = os.getenv("ENABLE_NOISE_SUPPRESSION", "true").lower() in ("true", "1", "yes")
NOISE_REDUCTION_STRENGTH = float(os.getenv("NOISE_REDUCTION_STRENGTH", "1.25"))
MAX_NORMALIZATION_GAIN = float(os.getenv("MAX_NORMALIZATION_GAIN", "3.0"))
MAX_COMPRESSION_RATIO = float(os.getenv("MAX_COMPRESSION_RATIO", "2.4"))
MIN_REPEATED_TEXT_LENGTH = 4
MIN_REPEATED_TEXT_COUNT = 3

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

def has_repeated_text_loop(text: str) -> bool:
    normalized_text = "".join(character for character in text if character.isalnum())
    max_unit_length = min(40, len(normalized_text) // MIN_REPEATED_TEXT_COUNT)

    for start_index in range(len(normalized_text)):
        for unit_length in range(MIN_REPEATED_TEXT_LENGTH, max_unit_length + 1):
            unit = normalized_text[start_index:start_index + unit_length]
            if len(unit) < unit_length:
                break

            repeated_text = unit * MIN_REPEATED_TEXT_COUNT
            if normalized_text.startswith(repeated_text, start_index):
                return True

    return False

def suppress_stationary_noise(audio: np.ndarray) -> np.ndarray:
    """以較安靜音框估計固定背景噪音，使用頻譜閘門降低空調、風扇及底噪。"""
    frame_size = 512
    hop_size = frame_size // 2
    if len(audio) < frame_size:
        return audio

    original_length = len(audio)
    edge_padding = frame_size // 2
    padded_audio = np.pad(audio, (edge_padding, edge_padding), mode="reflect")
    padded_length = frame_size + int(np.ceil((len(padded_audio) - frame_size) / hop_size)) * hop_size
    padded_audio = np.pad(padded_audio, (0, padded_length - len(padded_audio)))
    window = np.hanning(frame_size).astype(np.float32)
    frame_starts = range(0, padded_length - frame_size + 1, hop_size)
    spectra = np.stack([
        np.fft.rfft(padded_audio[start:start + frame_size] * window)
        for start in frame_starts
    ])

    magnitudes = np.abs(spectra)
    noise_floor = np.percentile(magnitudes, 20, axis=0)
    retained_magnitude = np.maximum(
        magnitudes - NOISE_REDUCTION_STRENGTH * noise_floor,
        magnitudes * 0.12,
    )
    filtered_spectra = spectra * retained_magnitude / np.maximum(magnitudes, 1e-8)

    output = np.zeros(padded_length, dtype=np.float32)
    window_sum = np.zeros(padded_length, dtype=np.float32)
    for frame_index, start in enumerate(frame_starts):
        filtered_frame = np.fft.irfft(filtered_spectra[frame_index], n=frame_size).astype(np.float32)
        output[start:start + frame_size] += filtered_frame * window
        window_sum[start:start + frame_size] += window * window

    valid = window_sum > 1e-6
    output[valid] /= window_sum[valid]
    return np.clip(output[edge_padding:edge_padding + original_length], -1.0, 1.0)

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

# LLM 與 RAG 串接設定
ENABLE_LLM = os.getenv("ENABLE_LLM", "true").lower() in ("true", "1", "yes")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy-key")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://host.docker.internal:8003")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "10"))
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    """你是盟立官方知識庫語音助理。只能依據提供的參考資料回答，不得使用模型記憶補充、推測或虛構資訊。

目前日期：{current_date}

回答前必須依照以下規則判斷參考資料：
1. 判斷問題是否涉及時效性。即使使用者沒有明確說「目前、最新、現任」，只要資訊可能隨時間改變，例如人員職務、組織架構、產品規格、價格、政策、財務數據、營運狀態、合作關係、服務內容或聯絡方式，都應視為查詢目前有效資訊。
2. 若參考資料互相衝突，優先採用有效日期較新的資料；若沒有有效日期，再比較發布日期、更新日期或內容中明確提到的事件日期。
3. 後續資料若明確表示接任、卸任、更新、修訂、取代、停止、退休、改名或變更，應視為已取代較早資料。
4. 歷史新聞、歷年公告及過去事件中的描述只能代表當時狀態，不得直接當作目前狀態。
5. 日期相近或無法判斷時，優先採用與問題最直接且具權威性的官方資料。公司治理、人事與財務優先採用董事會、投資人專區及重大公告；組織與職務優先採用經營團隊；產品與規格優先採用正式產品文件及產品頁面；新聞資料只作為事件與歷史背景，除非明確記載最新變更。
6. 不得因資料排在較前面、文字較長或重複出現次數較多，就判定它較新或較正確。
7. 若最新資料已能明確回答，只回答目前有效結果，不需要描述完整歷史沿革。
8. 若參考資料沒有答案，或資料仍然衝突且不足以確認目前狀態，回答「知識庫資料不足以確認最新資訊」，不得自行選擇或猜測。

回答限制在 1 至 2 句、30 字以內，只提供最重要資訊，不要重述問題，使用自然口語的繁體中文。不得提及參考資料、檔案名稱、路徑、檢索過程、排序分數、工具使用情形或參考連結。"""
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
        print(f" - LLM 位址: {LLM_BASE_URL}")
        print(f" - LLM 模型: {LLM_MODEL}")
        print(f" - RAG 位址: {RAG_BASE_URL}")
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
        "llm_backend": "openai-compatible",
        "llm_model": LLM_MODEL
    }

async def stream_llm_chat(websocket: WebSocket, history: list, user_text: str):
    """檢索 RAG 後將使用者文字送往 LLM，並串流回傳回答"""
    if not ENABLE_LLM:
        return

    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
            rag_response = await client.post(
                f"{RAG_BASE_URL.rstrip('/')}/query",
                json={"query": user_text, "top_k": RAG_TOP_K},
            )
            rag_response.raise_for_status()
            matches = rag_response.json().get("matches", [])
    except (httpx.HTTPError, ValueError) as error:
        error_message = f"RAG 查詢失敗: {type(error).__name__}: {error}"
        print(f"[RAG 錯誤] {error_message}")
        await websocket.send_json({"type": "llm_error", "error": error_message})
        return

    if not matches:
        error_message = "RAG 查無相關資料"
        print(f"[RAG] {error_message}")
        await websocket.send_json({"type": "llm_error", "error": error_message})
        return

    references = "\n\n---\n\n".join(
        f"【來源：{match.get('metadata', {}).get('source', '未知')}】\n{match['content']}"
        for match in matches
        if match.get("content")
    )
    system_prompt = LLM_SYSTEM_PROMPT.replace("{current_date}", date.today().isoformat())
    history.append({"role": "user", "content": user_text})
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n【參考資料】\n{references}"}
    ] + list(history)[-8:]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 256,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}

    print(f"[RAG] 已取得 {len(matches)} 筆參考資料")
    print(f"[LLM] 正在請求 {LLM_MODEL} 生成回覆...")
    await websocket.send_json({"type": "llm_start"})
    full_response = []

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    err_msg = f"LLM 回應狀態碼異常: {response.status_code}"
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

    except WebSocketDisconnect:
        raise
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

    if ENABLE_NOISE_SUPPRESSION:
        full_audio = suppress_stationary_noise(full_audio)

    # 限制增益，避免將現場底噪或遠處談話強制放大
    filtered_max = float(np.max(np.abs(full_audio)))
    normalization_gain = min(0.90 / max(filtered_max, 1e-8), MAX_NORMALIZATION_GAIN)
    full_audio = full_audio * normalization_gain

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
        if s.compression_ratio > MAX_COMPRESSION_RATIO:
            print(f"[STT 忽略] 文字重複率過高 (compression_ratio={s.compression_ratio:.2f}): '{s.text}'")
            continue
        valid_segments.append(s.text.strip())

    raw_text = "".join(valid_segments).strip()
    if has_repeated_text_loop(raw_text):
        print(f"[STT 忽略] 偵測到重複文字迴圈: '{raw_text}'")
        raw_text = ""
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
        await stream_llm_chat(websocket, history, text)
    else:
        print("[STT] 未辨識出有效文字（已過濾雜音/幻覺）")
        await websocket.send_json({"type": "status", "status": "empty"})

async def process_turn(full_audio: np.ndarray, websocket: WebSocket, history: list):
    try:
        await process_transcription(full_audio, websocket, history)
    finally:
        await websocket.send_json({"type": "status", "status": "ready"})

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
    processing_task = None

    # 建立該連線專屬的 Silero VAD 串流實例（維持獨立 hidden states）
    vad_stream = SileroVADStream(vad_model.session)

    silence_samples_limit = int(SAMPLE_RATE * SILENCE_DURATION_SEC)
    min_speech_samples = int(SAMPLE_RATE * MIN_SPEECH_DURATION_SEC)
    max_buffer_samples = int(SAMPLE_RATE * MAX_BUFFER_SEC)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            if processing_task is not None and processing_task.done():
                await processing_task
                processing_task = None

            if "bytes" in message and message["bytes"]:
                if processing_task is not None:
                    continue

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
                                processing_task = asyncio.create_task(
                                    process_turn(full_audio, websocket, conversation_history)
                                )
                            else:
                                print(f"[Silero VAD] 聲音過短 ({len(full_audio)/SAMPLE_RATE:.2f}s)，判定為雜音已忽略")

                            audio_buffer = []
                            pre_roll_buffer.clear()
                            vad_stream.reset()
                            is_speaking = False
                            silence_samples = 0
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
                    processing_task = asyncio.create_task(
                        process_turn(full_audio, websocket, conversation_history)
                    )

                    audio_buffer = []
                    pre_roll_buffer.clear()
                    vad_stream.reset()
                    is_speaking = False
                    silence_samples = 0

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
    finally:
        if processing_task is not None:
            if not processing_task.done():
                processing_task.cancel()
            await asyncio.gather(processing_task, return_exceptions=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
