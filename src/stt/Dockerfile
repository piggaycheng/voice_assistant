FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG CTRANSLATE2_VERSION=4.8.1

# 避免安裝過程中的互動式提示
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH"

# 安裝 Python 3、venv、pip、ffmpeg (音訊處理必需) 及其他基礎工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    ffmpeg \
    libopenblas-dev \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "v${CTRANSLATE2_VERSION}" --depth 1 --recursive \
        https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 \
    && cmake -S /tmp/ctranslate2 -B /tmp/ctranslate2/build \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF \
        -DOPENMP_RUNTIME=COMP \
        -DCUDA_ARCH_LIST=9.0+PTX \
        -DBUILD_CLI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/ctranslate2/build --parallel "$(nproc)" \
    && cmake --install /tmp/ctranslate2/build \
    && ldconfig \
    && python3 -m pip wheel --no-deps --wheel-dir /opt/ctranslate2-wheels \
        /tmp/ctranslate2/python \
    && rm -rf /tmp/ctranslate2

# 設定工作目錄
WORKDIR /app

# 複製入口啟動腳本
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# 預設執行指令
CMD ["python3", "scripts/main.py"]
