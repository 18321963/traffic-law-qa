
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

COPY pyproject.toml ./
COPY traffic_law_qa/ ./traffic_law_qa/
RUN pip install --no-cache-dir ".[api,agent]" \
 && rm -rf /app/build /app/*.egg-info

COPY 法规知识库/ ./法规知识库/
COPY data/ ./data/

RUN useradd --create-home --uid 1000 app \
 && mkdir -p "/app/法规知识库/chunks" "/app/法规知识库/index" \
 && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "traffic_law_qa.main:app", "--host", "0.0.0.0", "--port", "8000"]
