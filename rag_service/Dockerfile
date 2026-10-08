
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

# 嵌入与重排要 CUDA 版 torch：PyPI 上的 linux 轮子是 CPU-only，得另找 cu124 轮子；
# 直连 download.pytorch.org 在国内不稳，默认走阿里云的 flat 轮子目录（pip 的 --find-links 认它），
# 依赖（nvidia-* 等）走清华 PyPI 镜像。换回官方源：
#   --build-arg TORCH_WHEELS=https://download.pytorch.org/whl/cu124/torch/ --build-arg PYPI_INDEX=https://pypi.org/simple
# torch 单独一层：装法不读 pyproject、只随 ARG/基础镜像变，改源码重建时这一层缓存直接命中。
ARG TORCH_WHEELS=https://mirrors.aliyun.com/pytorch-wheels/cu124/
ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --index-url ${PYPI_INDEX} --find-links ${TORCH_WHEELS} \
      "torch==2.6.0+cu124"

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
