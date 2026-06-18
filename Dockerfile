FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY regagent ./regagent
COPY service ./service
COPY data ./data 2>/dev/null || true
ENV PORT=8080
EXPOSE 8080
CMD ["sh","-c","uvicorn service.api:app --host 0.0.0.0 --port ${PORT}"]
