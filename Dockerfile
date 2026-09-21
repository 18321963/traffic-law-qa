# 应用镜像：把 server 跑起来，连 compose 里的 Milvus。
#
#   docker compose up -d --build        # 起 Milvus + 应用
#   docker compose logs -f app          # 看装配日志
#   curl localhost:8000/health
#
# 与 docker-compose.yml 的分工：那里起的是**有状态**的 Milvus（etcd + MinIO + Milvus），
# 这里打的是**无状态**的应用。所以本镜像不碰 volumes/，数据全在 Milvus 那边。

# 与本地 venv 同版本（3.11），避免「本地跑得通、容器里语法不对」这类只在部署时暴露的问题
FROM python:3.11-slim

# PYTHONUNBUFFERED：不加的话 print 会被缓冲，「装配完成」那行日志在 docker logs 里看不到
# LANG=C.UTF-8：知识库目录名是中文（法规知识库/），非 UTF-8 locale 下路径会出问题
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先拷依赖清单 + 源码再装。
COPY pyproject.toml README.md ./
COPY traffic_law_qa/ ./traffic_law_qa/
# 只装 [api]：agent 组（LangGraph）服务层用不到，dev 组（pytest/ruff）更不该进生产镜像。
# 不是 editable 安装 —— 装的是依赖，顺带把包本体也复制了一份进 site-packages。
#
# 那个 site-packages 副本是**不生效的**，实测确认过：`python -m` 会把 CWD（=WORKDIR=/app）
# 放在 sys.path[0]，于是 /app/traffic_law_qa/ 盖过 site-packages，
# 真正加载的是这里 COPY 进来的源码。留一份死副本不优雅，但删掉源码目录反而是错的方向 ——
# 你 exec 进容器改 /app/traffic_law_qa/*.py 会立即生效，便于现场排查。
# 唯一的硬约束：**改了代码必须重新 build**，镜像里没有指向宿主机的链路。
RUN pip install --no-cache-dir ".[api]" \
 && rm -rf /app/build /app/*.egg-info

# 知识库：docx 是唯一真源，parsed/ 与 text/ 是已入库的结构层产物（parsed/ 在版本管理里，
# 所以新 clone 也有它）。chunks/ 与 index/ 被 .gitignore 忽略，构建机上有就一并带上
# （首次启动免重建、免重新向量化）；没有也行 —— **已实测**：把这两个目录清空后首次启动会自己重建，
# 10.0 秒走完「检索层产物缺失 → chunk 619 个 → 向量化 → 建出 619 行稠密+BM25 集合」。
# （619 是那次 4 部法语料下的数字。现在是 6 部 / 508 条 / 812 块，同一条路径会建出 812 行，
#   耗时按比例长一点，量级不变。）
# 其中 parse 一步因 parsed/ 的 sha1 未变而跳过；真删掉 parsed/ 才会从 docx 重解析。
# 「向量化」那一步要花一次 embedding 的钱 —— 那次验证是接桩端点跑完的（桩收到 63 批 / 620 条），零成本。
COPY 法规知识库/ ./法规知识库/
COPY data/ ./data/

# 非 root 运行。这两个派生目录必须**先建好并改属主**：
# compose 的命名卷在首次创建时会继承镜像里同路径目录的属主，
# 不建的话卷是 root 属主，app 用户写不进去，首次建库直接失败。
RUN useradd --create-home --uid 1000 app \
 && mkdir -p "/app/法规知识库/chunks" "/app/法规知识库/index" \
 && chown -R app:app /app
USER app

EXPOSE 8000

# --host 0.0.0.0 是必须的：server 的默认值是 127.0.0.1（对本地裸跑是安全的默认），
# 容器里保持默认的话，宿主机映射过来的端口连不上 —— 这是容器化最经典的坑。
CMD ["python", "-m", "uvicorn", "traffic_law_qa.server:app", "--host", "0.0.0.0", "--port", "8000"]
