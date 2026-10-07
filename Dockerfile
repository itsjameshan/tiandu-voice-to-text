# 云端演示用（只能放虚构材料！）。课堂和内网请用 Windows 便携包或直接 python app.py。
# 构建：docker build -t tiandu .
# 运行：docker run -p 7860:7860 -e DEMO_USERNAME=xxx -e DEMO_PASSWORD=xxx tiandu
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# 构建时下载必需模型
RUN python models/download_models.py

EXPOSE 7860
# 云端演示必须设置 DEMO_USERNAME 和 DEMO_PASSWORD（启用登录）
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
