import os
import io
import sys
import json
import asyncio
import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx
import websockets

# 連線與音訊參數
SERVER_URI = os.getenv("STT_SERVER_URI", "ws://localhost:8000/ws")
TTS_URL = os.getenv("TTS_SERVER_URL", "http://localhost:8001/tts")
ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes")
TTS_VOICE = os.getenv("TTS_VOICE", "zf_xiaoxiao")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 800  # 800 samples @ 16000Hz = 50ms 一包

# 控制播放狀態（播放時靜音麥克風，避免助理聽到自己講話形成迴音）
is_playing_audio = False

async def play_tts_audio(text: str, loop: asyncio.AbstractEventLoop):
    """向 TTS 伺服器請求語音並透過喇叭播放"""
    global is_playing_audio
    if not ENABLE_TTS or not text.strip():
        return

    try:
        print("\n🔊 [Kokoro TTS 合成中...]          ", end="\r", flush=True)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                TTS_URL,
                params={
                    "text": text,
                    "voice": TTS_VOICE,
                    "speed": TTS_SPEED
                }
            )

        if resp.status_code != 200:
            print(f"\n⚠️  [TTS 伺服器錯誤] 狀態碼: {resp.status_code}")
            return

        # 解碼 WAV 音訊
        audio_data, sample_rate = sf.read(io.BytesIO(resp.content), dtype="float32")

        # 設定播放中旗標（麥克風將暫停傳送人聲以防止回授）
        is_playing_audio = True
        print(f"🔊 [AI 語音播放中 ({len(audio_data)/sample_rate:.1f}s)...]          ", end="\r", flush=True)

        # 播放音訊並等待播放完成
        sd.play(audio_data, sample_rate)
        await loop.run_in_executor(None, sd.wait)

    except httpx.ConnectError:
        print(f"\n⚠️  [TTS 連線失敗] 無法連線至 {TTS_URL}，請確認 TTS 容器是否已啟動。")
    except Exception as e:
        print(f"\n⚠️  [TTS 播放異常] {e}")
    finally:
        # 播放完畢，恢復麥克風接收
        await asyncio.sleep(0.2)  # 額外留 200ms 緩衝清空尾音
        is_playing_audio = False
        print("🟢 [準備就緒，請說話...]               ", end="\r", flush=True)

async def audio_stream_client():
    global is_playing_audio
    audio_queue = asyncio.Queue()
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
    print(f" 🔊 TTS 伺服器: {TTS_URL} (語音: {TTS_VOICE}, 語速: {TTS_SPEED})")
    print(f" ⚙️  TTS 語音播放: {'開啟' if ENABLE_TTS else '關閉'}")
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
                                print("🤖 [AI 回覆]: ", end="", flush=True)

                            elif msg_type == "llm_chunk":
                                chunk = res.get("chunk", "")
                                print(chunk, end="", flush=True)

                            elif msg_type == "llm_end":
                                full_reply = res.get("text", "")
                                print("\n")
                                # 觸發 Kokoro TTS 語音播放
                                if ENABLE_TTS and full_reply:
                                    await play_tts_audio(full_reply, loop)
                                else:
                                    print("🟢 [準備就緒，請說話...]", end="\r", flush=True)

                            elif msg_type == "llm_error":
                                err = res.get("error", "")
                                print(f"\n❌ [AI 回應異常]: {err}\n")
                                print("🟢 [準備就緒，請說話...]", end="\r", flush=True)

                        except Exception as e:
                            print(f"\n[解析回應錯誤] {e}")

                # 啟動麥克風錄音串流
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    callback=audio_callback,
                ):
                    sender_task = asyncio.create_task(sender())
                    receiver_task = asyncio.create_task(receiver())
                    await asyncio.gather(sender_task, receiver_task)

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
