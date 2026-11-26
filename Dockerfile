# Koyeb / Docker 배포용 (Render는 이 파일을 쓰지 않습니다)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server

# Koyeb는 PORT 환경변수를 넣어 줍니다. 없으면 8000 사용.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT}"]
