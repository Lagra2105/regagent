FROM python:3.12-slim
WORKDIR /app
# git is needed to pip-install agentcost from its GitHub repo
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY regagent ./regagent
COPY service ./service
COPY data ./data
ENV PORT=8080 \
    AGENTCOST_DB=/tmp/regagent.db \
    AGENTCOST_BASE_PATH=/dashboard
EXPOSE 8080
CMD ["sh","-c","uvicorn service.api:app --host 0.0.0.0 --port ${PORT}"]
