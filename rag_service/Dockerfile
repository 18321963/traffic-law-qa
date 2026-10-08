
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

# 嵌入与重排要 CUDA 版 torch：PyPI 上 linux 的 torch 轮子本身就是 cu124 构建（requires
# nvidia-*-cu12==12.4.127 / cudnn 9.1 / triton 3.2），清华 PyPI 直装即可。download.pytorch.org 的
# 轮子实际由 R2 CDN 分发（实测 768MB 下到 338MB 读超时、pip 不续传）、aliyun 的 pytorch-wheels 被
# 限速到 ~250KB/s，这两条都别换回来。torch 单独一层：装法不读 pyproject、只随 ARG/基础镜像变，
# 改源码重建时这一层缓存直接命中；pip 缓存挂载让失败重试不必重下已下完的依赖。
ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${PYPI_INDEX} --retries 5 --timeout 60 \
      "torch==2.6.0"

COPY pyproject.toml ./
COPY rag_contracts/ ./rag_contracts/
COPY rag_service/ ./rag_service/
COPY eval/ ./eval/
COPY mcp_server/ ./mcp_server/
COPY api_contracts/ ./api_contracts/
RUN pip install --no-cache-dir --index-url ${PYPI_INDEX} ".[api,local,milvus,pdf]" \
 && rm -rf /app/build /app/*.egg-info

COPY 法规知识库/ ./法规知识库/
COPY data/ ./data/

RUN useradd --create-home --uid 1000 app \
 && mkdir -p "/app/法规知识库/chunks" "/app/法规知识库/index" \
 && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "rag_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
