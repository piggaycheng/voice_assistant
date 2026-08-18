import asyncio
import json
import sys
import numpy as np
import sounddevice as sd
import websockets

SERVER_URI = "ws://localhost:8000/ws"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 800  # 800 samples @ 16000Hz = 50ms 一包

async def audio_stream_client():
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"[音訊警告] {status}", file=sys.stderr)
        # 將音訊 byte 資料丟進 async queue
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.tobytes())

    print("=" * 60)
    print(" 🎙️  Voice Assistant STT 客戶端")
    print(f" 🔗 連線至伺服器: {SERVER_URI}")
    print("=" * 60)

    while True:
        try:
            async with websockets.connect(SERVER_URI) as ws:
                print("✅ 成功連線至 STT 伺服器！請對著麥克風說話（按 Ctrl+C 結束）...\n")

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
                                if status == "listening":
                                    print("🔴 [正在說話...]", end="\r", flush=True)
                                elif status == "transcribing":
                                    print("⏳ [辨識中...]    ", end="\r", flush=True)
                                elif status == "ready":
                                    print("🟢 [準備就緒，請說話...]", end="\r", flush=True)

                            elif msg_type == "result":
                                text = res.get("text", "")
                                duration = res.get("duration", 0)
                                print(f"\n🗣️  [辨識結果 ({duration}s)]: {text}\n")
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
