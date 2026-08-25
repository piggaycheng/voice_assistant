import os
import sys
import json
import asyncio
from urllib.parse import urlsplit, urlunsplit
import numpy as np
import sounddevice as sd
import websockets

# 連線與音訊參數
SERVER_URI = os.getenv("STT_SERVER_URI", "ws://localhost:8002/ws")
TTS_URL = os.getenv("TTS_SERVER_URL", "http://localhost:8001/tts")
ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes")
TTS_VOICE = os.getenv("TTS_VOICE", "zf_xiaoxiao")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_MIN_SEGMENT_CHARS = int(os.getenv("TTS_MIN_SEGMENT_CHARS", "6"))
TTS_SEGMENT_BOUNDARIES = frozenset("，,。！？!?；;：:\n")

def build_tts_ws_url(tts_url: str) -> str:
    parsed = urlsplit(tts_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/tts"):
        base_path = base_path[:-4]
    return urlunsplit((scheme, parsed.netloc, f"{base_path}/ws", "", ""))

TTS_WS_URL = os.getenv("TTS_WS_URL", build_tts_ws_url(TTS_URL))

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 800  # 800 samples @ 16000Hz = 50ms 一包

def resolve_audio_device(device_setting: str, kind: str = "input"):
    """
    解析指定音訊設備 ID 或名稱，預設指名使用 PipeWire (若無則依序嘗試 pulse / default)。
    kind: 'input' 或 'output'
    """
    if device_setting is not None and str(device_setting).strip().isdigit():
        return int(device_setting)

    target = (device_setting or "pipewire").strip().lower()
    search_list = [target]
    # 若指名 pipewire 但找不到，提供相容回退
    if target == "pipewire":
        search_list.extend(["pulse", "default"])

    try:
        devices = sd.query_devices()
        ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
        for candidate in search_list:
            for idx, dev in enumerate(devices):
                if dev.get(ch_key, 0) > 0 and candidate in dev.get("name", "").lower():
                    return idx
    except Exception as e:
        print(f"⚠️ [音訊設備查詢警告] {e}", file=sys.stderr)

    return None

def get_device_label(device_id, kind: str = "input") -> str:
    """取得設備顯示名稱"""
    if device_id is None:
        return "系統預設 (Default)"
    try:
        dev = sd.query_devices(device_id)
        return f"[ID: {device_id}] {dev.get('name', 'Unknown')}"
    except Exception:
        return f"[ID: {device_id}]"

# 設定指名使用 PipeWire
INPUT_DEVICE_ID = resolve_audio_device(os.getenv("AUDIO_INPUT_DEVICE", "pipewire"), kind="input")
OUTPUT_DEVICE_ID = resolve_audio_device(os.getenv("AUDIO_OUTPUT_DEVICE", "pipewire"), kind="output")

# 控制播放狀態（播放時靜音麥克風，避免助理聽到自己講話形成迴音）
is_playing_audio = False

async def play_tts_audio(text: str, loop: asyncio.AbstractEventLoop):
    """透過 TTS WebSocket 接收 PCM 串流並即時播放。"""
    if not ENABLE_TTS or not text.strip():
        return

    output_stream = None
    try:
        print("\n🔊 [Kokoro TTS 串流合成中...]          ", end="\r", flush=True)
        async with websockets.connect(TTS_WS_URL, max_size=None) as tts_ws:
            await tts_ws.send(json.dumps({
                "action": "synthesize",
                "text": text,
                "voice": TTS_VOICE,
                "speed": TTS_SPEED
            }))

            async for message in tts_ws:
                if isinstance(message, bytes):
                    if output_stream is None:
                        raise RuntimeError("TTS 未先傳送 audio_start")
                    audio_data = np.frombuffer(message, dtype="<f4").reshape(-1, 1)
                    await loop.run_in_executor(None, output_stream.write, audio_data)
                    continue

                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "audio_start":
                    output_stream = sd.OutputStream(
                        device=OUTPUT_DEVICE_ID,
                        samplerate=int(event["sample_rate"]),
                        channels=int(event.get("channels", 1)),
                        dtype="float32"
                    )
                    output_stream.start()
                    print("🔊 [AI PCM 串流播放中...]          ", end="\r", flush=True)
                elif event_type == "done":
                    break
                elif event_type == "error":
                    raise RuntimeError(event.get("error", "未知 TTS 錯誤"))

    except (OSError, websockets.exceptions.WebSocketException):
        print(f"\n⚠️  [TTS 連線失敗] 無法連線至 {TTS_WS_URL}，請確認 TTS 容器是否已啟動。")
    except Exception as e:
        print(f"\n⚠️  [TTS 播放異常] {e}")
    finally:
        if output_stream is not None:
            await loop.run_in_executor(None, output_stream.stop)
            output_stream.close()

def extract_tts_segments(buffer: str, flush: bool = False) -> tuple[list[str], str]:
    """從 LLM 串流文字中取出適合立即朗讀的完整短句。"""
    segments = []
    segment_start = 0

    for index, character in enumerate(buffer):
        candidate = buffer[segment_start:index + 1].strip()
        if character in TTS_SEGMENT_BOUNDARIES and len(candidate) >= TTS_MIN_SEGMENT_CHARS:
            segments.append(candidate)
            segment_start = index + 1

    remainder = buffer[segment_start:]
    if flush and remainder.strip():
        segments.append(remainder.strip())
        remainder = ""
    return segments, remainder

async def tts_stream_worker(tts_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """依序合成並播放短句，避免多段音訊互相重疊。"""
    global is_playing_audio
    response_active = False

    while True:
        event, text = await tts_queue.get()
        try:
            if event == "segment":
                if not response_active:
                    response_active = True
                    is_playing_audio = True
                await play_tts_audio(text, loop)
            elif event == "end":
                if response_active:
                    await asyncio.sleep(0.2)
                response_active = False
                is_playing_audio = False
                print("🟢 [準備就緒，請說話...]               ", end="\r", flush=True)
        finally:
            tts_queue.task_done()

async def audio_stream_client():
    global is_playing_audio
    audio_queue = asyncio.Queue()
    tts_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"[音訊警告] {status}", file=sys.stderr)
        
        # 若正在播放 AI 語音，發送靜音 (避免麥克風錄到喇叭聲音)
        if is_playing_audio:
            silent_chunk = np.zeros_like(indata)
            loop.call_soon_threadsafe(audio_queue.put_nowait, silent_chunk.tobytes())
        else:
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.tobytes())

    print("=" * 65)
    print(" 🎙️  Voice Assistant 語音對話客戶端")
    print(f" 🔗 STT 伺服器: {SERVER_URI}")
    print(f" 🔊 TTS 伺服器: {TTS_WS_URL} (語音: {TTS_VOICE}, 語速: {TTS_SPEED})")
    print(f" ⚙️  TTS 語音播放: {'開啟' if ENABLE_TTS else '關閉'}")
    print(f" 🎤 麥克風 (Input):  {get_device_label(INPUT_DEVICE_ID, kind='input')}")
    print(f" 🔈 喇叭 (Output):   {get_device_label(OUTPUT_DEVICE_ID, kind='output')}")
    print("=" * 65)

    while True:
        try:
            async with websockets.connect(SERVER_URI) as ws:
                print("✅ 成功連線至語音助理伺服器！請對著麥克風說話（按 Ctrl+C 結束）...\n")

                # 發送音訊 Task
                async def sender():
                    while True:
                        data = await audio_queue.get()
                        await ws.send(data)

                # 接收辨識結果 Task
                async def receiver():
                    tts_buffer = ""
                    async for message in ws:
                        try:
                            res = json.loads(message)
                            msg_type = res.get("type")

                            if msg_type == "status":
                                status = res.get("status")
                                if status == "listening" and not is_playing_audio:
                                    print("🔴 [正在說話...]          ", end="\r", flush=True)
                                elif status == "transcribing":
                                    print("⏳ [語音辨識中...]        ", end="\r", flush=True)
                                elif status == "empty":
                                    print("⚠️  [未辨識出文字，請靠近麥克風再說一次]   ", end="\r", flush=True)
                                elif status == "ready" and not is_playing_audio:
                                    print("🟢 [準備就緒，請說話...]  ", end="\r", flush=True)

                            elif msg_type == "result":
                                text = res.get("text", "")
                                duration = res.get("duration", 0)
                                print(f"\n🗣️  [你說 ({duration}s)]: {text}")

                            elif msg_type == "llm_start":
                                tts_buffer = ""
                                print("🤖 [AI 回覆]: ", end="", flush=True)

                            elif msg_type == "llm_chunk":
                                chunk = res.get("chunk", "")
                                print(chunk, end="", flush=True)
                                if ENABLE_TTS:
                                    tts_buffer += chunk
                                    segments, tts_buffer = extract_tts_segments(tts_buffer)
                                    for segment in segments:
                                        tts_queue.put_nowait(("segment", segment))
                                    if segments:
                                        await asyncio.sleep(0)

                            elif msg_type == "llm_end":
                                print("\n")
                                if ENABLE_TTS:
                                    segments, tts_buffer = extract_tts_segments(tts_buffer, flush=True)
                                    for segment in segments:
                                        tts_queue.put_nowait(("segment", segment))
                                    tts_queue.put_nowait(("end", ""))
                                    await asyncio.sleep(0)
                                else:
                                    print("🟢 [準備就緒，請說話...]", end="\r", flush=True)

                            elif msg_type == "llm_error":
                                err = res.get("error", "")
                                print(f"\n❌ [AI 回應異常]: {err}\n")
                                tts_buffer = ""
                                if ENABLE_TTS:
                                    await tts_queue.put(("end", ""))
                                else:
                                    print("🟢 [準備就緒，請說話...]", end="\r", flush=True)

                        except Exception as e:
                            print(f"\n[解析回應錯誤] {e}")

                # 啟動麥克風錄音串流 (指名 Input Device)
                with sd.InputStream(
                    device=INPUT_DEVICE_ID,
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    callback=audio_callback,
                ):
                    sender_task = asyncio.create_task(sender())
                    receiver_task = asyncio.create_task(receiver())
                    tts_task = asyncio.create_task(tts_stream_worker(tts_queue, loop))
                    await asyncio.gather(sender_task, receiver_task, tts_task)

        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError):
            print(f"⚠️  無法連線至 {SERVER_URI}，5 秒後嘗試重連...")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except KeyboardInterrupt:
            print("\n👋 程式已終止。")
            break

def main():
    try:
        asyncio.run(audio_stream_client())
    except KeyboardInterrupt:
        print("\n👋 程式已退出。")

if __name__ == "__main__":
    main()
