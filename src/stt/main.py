import os
from faster_whisper import WhisperModel

def main():
    model_size = os.getenv("WHISPER_MODEL", "tiny")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT", "/app/models")

    # 確保 models 目錄存在
    os.makedirs(download_root, exist_ok=True)

    print(f"Loading faster-whisper model...")
    print(f" - Model size:    {model_size}")
    print(f" - Device:        {device}")
    print(f" - Compute type:  {compute_type}")
    print(f" - Model storage: {download_root}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root
    )
    print("Model loaded successfully!")

if __name__ == "__main__":
    main()
