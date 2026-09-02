import os
import io
import json
import asyncio
import subprocess
import uuid
from typing import Optional, List, Dict
import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from kokoro import KPipeline

app = FastAPI(
    title="Kokoro TTS Voice Assistant Server",
    description="High-quality, local Text-to-Speech service powered by Kokoro-82M",
    version="1.0.0"
)

# 環境變數與預設設定
TTS_PORT = int(os.getenv("TTS_PORT", "8001"))
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")
DEFAULT_LANG = os.getenv("TTS_DEFAULT_LANG", "z")       # 預設繁中/簡中 ('z')
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "zf_xiaoxiao")
DEFAULT_SPEED = float(os.getenv("TTS_DEFAULT_SPEED", "1.0"))
SAMPLE_RATE = 24000
OUTPUT_FOLDER = os.getenv("TTS_OUTPUT_FOLDER", "/app/output")
TTS_PUBLIC_BASE_URL = os.getenv("TTS_PUBLIC_BASE_URL", "http://localhost:8001").rstrip("/")
EXTERNAL_DEVICE_API_URL = os.getenv("EXTERNAL_DEVICE_API_URL", "")
EXTERNAL_DEVICE_API_TIMEOUT = float(os.getenv("EXTERNAL_DEVICE_API_TIMEOUT", "10"))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_FOLDER), name="output")

# 支援的語音清單與說明
SUPPORTED_VOICES: Dict[str, Dict[str, str]] = {
    # 國語 / 中文 (Mandarin Chinese) - lang_code: 'z'
    "zf_xiaoxiao": {"lang": "z", "gender": "female", "description": "Mandarin Chinese - Female (親切自然)"},
    "zf_xiaobei": {"lang": "z", "gender": "female", "description": "Mandarin Chinese - Female (溫柔沉穩)"},
    "zf_xiaoni": {"lang": "z", "gender": "female", "description": "Mandarin Chinese - Female (活力清晰)"},
    "zf_xiaoyi": {"lang": "z", "gender": "female", "description": "Mandarin Chinese - Female (知性優雅)"},
    "zm_yunjian": {"lang": "z", "gender": "male", "description": "Mandarin Chinese - Male (沉穩男聲)"},
    "zm_yunxi": {"lang": "z", "gender": "male", "description": "Mandarin Chinese - Male (陽光男聲)"},
    "zm_yunxia": {"lang": "z", "gender": "male", "description": "Mandarin Chinese - Male (溫暖男聲)"},
    "zm_yunyang": {"lang": "z", "gender": "male", "description": "Mandarin Chinese - Male (專業播音)"},

    # 美式英語 (American English) - lang_code: 'a'
    "af_heart": {"lang": "a", "gender": "female", "description": "American English - Female (旗艦極佳聲音)"},
    "af_bella": {"lang": "a", "gender": "female", "description": "American English - Female (甜美自然)"},
    "af_nicole": {"lang": "a", "gender": "female", "description": "American English - Female (清晰流暢)"},
    "af_sarah": {"lang": "a", "gender": "female", "description": "American English - Female (專業穩重)"},
    "af_sky": {"lang": "a", "gender": "female", "description": "American English - Female (輕快活潑)"},
    "af_alloy": {"lang": "a", "gender": "female", "description": "American English - Female (俐落知性)"},
    "am_adam": {"lang": "a", "gender": "male", "description": "American English - Male (沉穩成熟)"},
    "am_michael": {"lang": "a", "gender": "male", "description": "American English - Male (自然對話)"},
    "am_onyx": {"lang": "a", "gender": "male", "description": "American English - Male (厚重低沉)"},

    # 英式英語 (British English) - lang_code: 'b'
    "bf_emma": {"lang": "b", "gender": "female", "description": "British English - Female (英倫典雅)"},
    "bf_isabella": {"lang": "b", "gender": "female", "description": "British English - Female (溫和自然)"},
    "bf_alice": {"lang": "b", "gender": "female", "description": "British English - Female (清脆悅耳)"},
    "bm_george": {"lang": "b", "gender": "male", "description": "British English - Male (紳士男聲)"},
    "bm_lewis": {"lang": "b", "gender": "male", "description": "British English - Male (沉著磁性)"},

    # 日語 (Japanese) - lang_code: 'j'
    "jf_alpha": {"lang": "j", "gender": "female", "description": "Japanese - Female (日系甜美女聲)"},
    "jf_gongitsune": {"lang": "j", "gender": "female", "description": "Japanese - Female (故事旁白女聲)"},
    "jf_nezumi": {"lang": "j", "gender": "female", "description": "Japanese - Female (可愛輕快)"},
    "jf_tebukuro": {"lang": "j", "gender": "female", "description": "Japanese - Female (溫柔沉著)"},
    "jm_kumo": {"lang": "j", "gender": "male", "description": "Japanese - Male (自然男聲)"}
}

# 全域 pipeline 快取
pipeline_cache: Dict[str, KPipeline] = {}
pipeline_lock = asyncio.Lock()

def infer_lang_code(voice: str, explicit_lang: Optional[str] = None) -> str:
    """根據 voice 名稱或參數推導語言代碼"""
    if explicit_lang:
        return explicit_lang
    if voice in SUPPORTED_VOICES:
        return SUPPORTED_VOICES[voice]["lang"]
    if voice and len(voice) >= 2 and voice[1] in ('f', 'm') and (len(voice) == 2 or voice[2] == '_'):
        return voice[0]
    return DEFAULT_LANG

def get_or_create_pipeline(lang_code: str) -> KPipeline:
    """取得或初始化對應語言代碼的 KPipeline"""
    if lang_code not in pipeline_cache:
        print(f"[Kokoro] 正在載入語言 [{lang_code}] 的 TTS 運算管線...")
        pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
        print(f"[Kokoro] 語言 [{lang_code}] 載入完成！")
    return pipeline_cache[lang_code]

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print(" 🔊 正在初始化 Kokoro TTS 語音合成伺服器...")
    print(f" - 預設語言 (Lang): {DEFAULT_LANG}")
    print(f" - 預設語音 (Voice): {DEFAULT_VOICE}")
    print(f" - 預設語速 (Speed): {DEFAULT_SPEED}")
    print(f" - 運算裝置 (Device): {TTS_DEVICE}")
    print(f" - 服務連接埠 (Port): {TTS_PORT}")
    print("=" * 60)

    # 啟動時預熱預設語言管線，避免首次請求時延遲過長
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, get_or_create_pipeline, DEFAULT_LANG)
    print("✅ Kokoro TTS 服務已就緒！")

class TTSRequest(BaseModel):
    text: Optional[str] = None
    input: Optional[str] = None  # OpenAI 相容參數
    voice: Optional[str] = None
    lang_code: Optional[str] = None
    speed: Optional[float] = None
    response_format: Optional[str] = "wav"
    delivery: Optional[str] = "client"

async def post_audio_to_external_device(audio_url: str):
    """通知外部設備下載並播放已產生的 MP3。"""
    if not EXTERNAL_DEVICE_API_URL:
        print("[外部設備] 未設定 EXTERNAL_DEVICE_API_URL，略過音檔通知")
        return

    # TODO: 取得設備 API 的正式規格後，在此調整 JSON 欄位。
    payload = {
        "type": "audio",
        "url": audio_url,
    }

    try:
        async with httpx.AsyncClient(timeout=EXTERNAL_DEVICE_API_TIMEOUT) as client:
            response = await client.post(EXTERNAL_DEVICE_API_URL, json=payload)
            response.raise_for_status()
        print(f"[外部設備] 已送出音檔通知: {audio_url}")
    except httpx.HTTPError as error:
        print(f"[外部設備錯誤] 音檔通知失敗: {type(error).__name__}: {error}")

def create_mp3_destination() -> tuple[str, str]:
    filename = f"speech-{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    public_url = f"{TTS_PUBLIC_BASE_URL}/output/{filename}"
    return output_path, public_url

def synthesize_audio_sync(
    text: str,
    voice: str,
    lang_code: str,
    speed: float,
    mp3_output_path: Optional[str] = None,
) -> bytes:
    """同步執行 Kokoro TTS 語音生成並轉為 WAV 二進位資料"""
    pipeline = get_or_create_pipeline(lang_code)
    generator = pipeline(text, voice=voice, speed=speed)

    audio_chunks = []
    for _, _, audio in generator:
        if audio is not None and len(audio) > 0:
            audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("未能生成任何語音資料")

    full_audio = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, SAMPLE_RATE, format="WAV")

    if mp3_output_path:
        temporary_wav_path = f"{mp3_output_path}.tmp.wav"
        try:
            sf.write(temporary_wav_path, full_audio, SAMPLE_RATE, format="WAV")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", temporary_wav_path,
                    "-codec:a", "libmp3lame", "-b:a", "128k",
                    mp3_output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            if os.path.exists(temporary_wav_path):
                os.remove(temporary_wav_path)

    return buffer.getvalue()

@app.get("/")
def index():
    return {
        "status": "running",
        "service": "kokoro-tts-server",
        "default_voice": DEFAULT_VOICE,
        "default_lang": DEFAULT_LANG,
        "default_speed": DEFAULT_SPEED,
        "sample_rate": SAMPLE_RATE,
        "loaded_languages": list(pipeline_cache.keys()),
        "endpoints": {
            "get_tts": "/tts?text=你好世界&voice=zf_xiaoxiao",
            "post_tts": "/tts",
            "openai_compatible": "/v1/audio/speech",
            "list_voices": "/voices",
            "websocket": "/ws"
        }
    }

@app.get("/voices")
def list_voices():
    """回傳所有支援的語音與說明"""
    return {
        "count": len(SUPPORTED_VOICES),
        "default_voice": DEFAULT_VOICE,
        "voices": SUPPORTED_VOICES
    }

@app.get("/tts")
async def get_tts(
    text: str = Query(..., description="要合成的文字內容"),
    voice: Optional[str] = Query(None, description="語音名稱，如 zf_xiaoxiao, af_heart"),
    lang: Optional[str] = Query(None, description="語言代號，如 z, a, b, j"),
    speed: Optional[float] = Query(None, description="語速倍率 (預設 1.0)"),
    delivery: str = Query("client", description="client 或 external_audio"),
):
    """GET 介面：方便透過瀏覽器網址或 curl 直接播放與測試"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text 參數不得為空")

    target_voice = voice or DEFAULT_VOICE
    target_lang = infer_lang_code(target_voice, lang)
    target_speed = speed if speed is not None else DEFAULT_SPEED
    mp3_output_path = None
    audio_url = None
    if delivery == "external_audio":
        mp3_output_path, audio_url = create_mp3_destination()

    try:
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(
            None,
            synthesize_audio_sync,
            text,
            target_voice,
            target_lang,
            target_speed,
            mp3_output_path,
        )
        if audio_url:
            await post_audio_to_external_device(audio_url)
        headers = {"X-External-Audio-URL": audio_url} if audio_url else None
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers=headers,
        )
    except Exception as e:
        print(f"[TTS 錯誤] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
@app.post("/v1/audio/speech")
async def post_tts(req: TTSRequest):
    """POST 介面：支援標準 JSON 呼叫與 OpenAI 相容介面"""
    raw_text = req.text or req.input
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="請提供 'text' 或 'input' 內容")

    target_voice = req.voice or DEFAULT_VOICE
    target_lang = infer_lang_code(target_voice, req.lang_code)
    target_speed = req.speed if req.speed is not None else DEFAULT_SPEED
    mp3_output_path = None
    audio_url = None
    if req.delivery == "external_audio":
        mp3_output_path, audio_url = create_mp3_destination()

    try:
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(
            None,
            synthesize_audio_sync,
            raw_text,
            target_voice,
            target_lang,
            target_speed,
            mp3_output_path,
        )
        if audio_url:
            await post_audio_to_external_device(audio_url)
        headers = {"Content-Disposition": "inline; filename=speech.wav"}
        if audio_url:
            headers["X-External-Audio-URL"] = audio_url
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers=headers,
        )
    except Exception as e:
        print(f"[TTS 錯誤] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_tts_endpoint(websocket: WebSocket):
    """WebSocket 介面：接收文字生成請求並串流回傳狀態與語音資料"""
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[WebSocket-TTS] 客戶端連線成功: {client_info}")

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await websocket.send_json({"type": "error", "error": "無效的 JSON 格式"})
                continue

            action = data.get("action", "synthesize")

            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if action == "synthesize":
                raw_text = data.get("text", "")
                if not raw_text.strip():
                    await websocket.send_json({"type": "error", "error": "文字內容不得為空"})
                    continue

                target_voice = data.get("voice") or DEFAULT_VOICE
                target_lang = infer_lang_code(target_voice, data.get("lang_code"))
                target_speed = float(data.get("speed", DEFAULT_SPEED))
                mp3_output_path = None
                audio_url = None
                if data.get("delivery") == "external_audio":
                    mp3_output_path, audio_url = create_mp3_destination()

                await websocket.send_json({
                    "type": "status",
                    "status": "synthesizing",
                    "voice": target_voice,
                    "lang": target_lang
                })

                try:
                    loop = asyncio.get_running_loop()
                    wav_bytes = await loop.run_in_executor(
                        None,
                        synthesize_audio_sync,
                        raw_text,
                        target_voice,
                        target_lang,
                        target_speed,
                        mp3_output_path,
                    )
                    if audio_url:
                        await post_audio_to_external_device(audio_url)
                    # 先發送二進位音訊資料
                    await websocket.send_bytes(wav_bytes)
                    # 再發送完成通知
                    await websocket.send_json({
                        "type": "done",
                        "sample_rate": SAMPLE_RATE,
                        "audio_url": audio_url,
                    })
                except Exception as e:
                    print(f"[WebSocket-TTS 錯誤] {e}")
                    await websocket.send_json({"type": "error", "error": str(e)})

    except WebSocketDisconnect:
        print(f"[WebSocket-TTS] 客戶端斷開連線: {client_info}")
    except Exception as e:
        print(f"[WebSocket-TTS 異常] {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=TTS_PORT)
