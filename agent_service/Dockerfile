FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

COPY pyproject.toml ./
COPY rag_contracts/ ./rag_contracts/
COPY api_contracts/ ./api_contracts/
COPY agent_service/ ./agent_service/

ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --index-url ${PYPI_INDEX} ".[agent,api,langfuse]" \
 && rm -rf /app/build /app/*.egg-info

RUN mkdir -p /app/data \
 && useradd --create-home --uid 1000 app \
 && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agent_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
