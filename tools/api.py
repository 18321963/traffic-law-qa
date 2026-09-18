"""对外唯一入口：`qa()` —— 一次调用拿到答案（或检索结果）。

    from tools import qa

    answer = qa("醉驾怎么处罚")                       # 检索 + 生成
    result = qa("深圳 行人 在机动车道", mode="search")  # 只检索，不花 LLM 的钱
    print(qa("醉驾怎么处罚", debug=True).render())     # 附双通道排名

内部七层管道原样保留，这一层只做三件事：

1. **确保索引就绪**：逐一比对 docx 的 sha1、本地产物、Milvus 集合行数与快照，
   全部一致就复用，只有真过期才重建 —— 不会无谓地重跑向量化花钱。
2. **装配检索器与生成器**，把「先 build 再 ask」的两步收敛成一次调用。
3. **把失败翻译成一行中文提示**（含该执行的命令），而不是 pymilvus 的堆栈。

失败一律抛 `QaError`：库里不该替调用方打印，但消息本身就是给用户看的那一行。
命令行入口（`python -m tools`）与 demo.py 会把它接住并打印。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .contracts import Answer, Question, RetrievalResult

__all__ = ["qa", "QaError", "ReadyState", "ensure_ready", "render", "MODE_ASK", "MODE_SEARCH"]

MODE_ASK = "ask"
MODE_SEARCH = "search"


class QaError(RuntimeError):
    """对外可读的失败：`str(exc)` 就是给用户看的一行中文提示，不带堆栈。"""


@dataclass(frozen=True)
class ReadyState:
    """索引就绪检查的结果。"""

    action: str          # "reuse" 复用已有索引 | "rebuild" 重建过
    reason: str          # 复用 / 重建的原因
    rows: int            # 集合行数（= 子块数）
    laws: int            # 法规部数
    articles: int        # 条文数
    parents: int         # 父块数（应等于条文数）
    dense: bool          # 集合是否带稠密向量
    milvus: str          # Milvus 版本号

    def describe(self) -> str:
        verb = "复用已有索引" if self.action == "reuse" else "已重建索引"
        channels = "稠密+BM25" if self.dense else "纯 BM25"
        return (
            f"{verb}：{self.laws} 部法规 / {self.articles} 条 / {self.parents} 父块，"
            f"集合 {self.rows} 行（{channels}）｜{self.reason}｜Milvus {self.milvus}"
        )


# ================================================================== 就绪检查
def ensure_ready(*, with_vector: bool = True, rebuild: bool = False) -> ReadyState:
    """确保 Milvus 里有一套与 docx 当前内容一致的索引；只在必要时重建。"""
    from .indexer import Indexer
    from .milvus_store import MilvusStore

    docx_files = sorted(config.DOCX_DIR.glob("*.docx"))
    if not docx_files:
        raise QaError(f"知识库为空：{config.DOCX_DIR} 下没有 docx 文件")

    store = MilvusStore(verbose=False)
    version = _ping(store)
    stats = Indexer(verbose=False).load_stats()

    # 重建按「能建就建」定档：绝不比现有索引更弱。查询走不走稠密由 with_vector 决定，
    # 但它不该决定索引被建成什么样 —— 否则一次 --no-vector 就会把已有（花过钱的）稠密索引
    # drop 掉，之后所有默认调用都复用这套纯 BM25 集合，稠密通道静默消失。
    build_dense = config.embed_config().ready and (
        with_vector or bool(stats is not None and stats.vector_enabled)
    )

    reason = "指定了 rebuild=True" if rebuild else _stale_reason(store, stats, want_dense=build_dense)
    if reason is not None:
        print(f"[qa] 索引需要重建（{reason}），开始建库；首次约 30~60 秒…")
        from .pipeline import RagPipeline

        # force=rebuild：rebuild 的语义是「无视 sha1 强制重解析」，不传就只能靠切块层重算，
        # 改了 law_parser 的解析逻辑时新规则永不生效。
        RagPipeline(verbose=True).build(force=rebuild, with_vector=build_dense)
        stats = Indexer(verbose=False).load_stats()
        if stats is None:  # 建完仍读不到快照，说明 index 阶段没真正落盘
            raise QaError(f"建库未写出索引快照：{config.INDEX_META_PATH}")

    laws, articles = _manifest_counts()
    return ReadyState(
        action="rebuild" if reason is not None else "reuse",
        reason=reason or "docx、本地产物与集合三者一致",
        rows=stats.rows,
        laws=laws,
        articles=articles,
        parents=articles,
        dense=stats.vector_enabled,
        milvus=version,
    )


def _ping(store) -> str:
    """探活；连不上时把 MilvusError 换成一行中文提示。"""
    from .milvus_store import MilvusError

    try:
        return store.ping()
    except MilvusError:
        raise QaError(
            f"Milvus 连不上（{store.uri}）→ 先执行：docker compose up -d --wait"
        ) from None


def _stale_reason(store, stats, *, want_dense: bool) -> str | None:
    """比对 docx / 本地产物 / Milvus 集合；返回需要重建的原因，None 表示可直接复用。

    want_dense：本次是否打算建稠密向量（由 ensure_ready 的 build_dense 传入）。
    只在「该有稠密却没有」时算过期；反过来（索引有稠密、本次只查 BM25）不算 ——
    查询通道可以少用，不该为此把索引重建弱。
    """
    from .law_parser import LawLibrary, sha1_of

    library = LawLibrary()
    entries = library.manifest().get("laws", [])
    if not entries:
        return "结构层产物缺失（parsed/manifest.json 为空）"

    for entry in entries:
        if not library.path_of(entry["law_id"]).exists():
            return f"缺少结构层产物 {entry['law_id']}.json"

    cached = {entry["file"]: entry for entry in entries}
    docx_files = sorted(config.DOCX_DIR.glob("*.docx"))
    if len(docx_files) != len(entries):
        return f"docx 数量（{len(docx_files)}）与清单（{len(entries)}）不一致"
    for path in docx_files:
        entry = cached.get(path.name)
        if entry is None:
            return f"新法规未入库：{path.name}"
        if entry.get("sha1") != sha1_of(path):
            return f"{path.name} 已改动（sha1 与清单不一致）"

    if not config.CHUNKS_PATH.exists() or not config.PARENTS_PATH.exists():
        return "检索层产物缺失（chunks/chunks.jsonl 或 parents.jsonl）"
    articles = sum(int(item.get("articles", 0)) for item in entries)
    parents_on_disk = _count_lines(config.PARENTS_PATH)
    if parents_on_disk != articles:
        return f"父块数（{parents_on_disk}）与条文数（{articles}）不一致，切块层产物已过期"

    if stats is None:
        return "索引快照缺失（index/index_meta.json）"
    if not store.has_collection():
        return f"Milvus 集合 {store.collection} 不存在"
    actual = store.count()
    if actual != stats.rows:
        return f"集合行数（{actual}）与索引快照（{stats.rows}）不一致"

    if want_dense and not stats.vector_enabled:
        from .indexer import EMBED_FAILED_NOTE_PREFIX, Indexer

        notes = Indexer(verbose=False).load_notes()
        if any(note.startswith(EMBED_FAILED_NOTE_PREFIX) for note in notes):
            # 上次已试过稠密、向量端点挂了。此刻重建只会再失败一次并白 drop 集合，
            # 所以不自动重试；端点恢复后用 --rebuild 显式补上。
            return None
        return "索引快照为纯 BM25，本次需要向量通道，重建以补上稠密向量字段"

    return None


def _manifest_counts() -> tuple[int, int]:
    """清单里的（法规部数, 条文数）。"""
    from .law_parser import LawLibrary

    entries = LawLibrary().manifest().get("laws", [])
    return len(entries), sum(int(item.get("articles", 0)) for item in entries)


def _count_lines(path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


# ================================================================== 唯一入口
def qa(
    question: str,
    *,
    mode: str = MODE_ASK,
    top_k: int | None = None,
    debug: bool = False,
    with_vector: bool = True,
    rebuild: bool = False,
) -> Answer | RetrievalResult:
    """一次调用拿到结果：确保索引就绪 → 混合检索 →（ask 模式）生成答案。

    mode="ask"     返回 Answer（带 [依据N] 与参考文献）
    mode="search"  返回 RetrievalResult（只检索，不花 LLM 的钱）
    top_k          覆盖默认召回条数（默认取 RAG_TOP_K，6）
    debug=True     额外跑单通道检索，给出每条法条被稠密向量 / BM25 各自排到第几
    with_vector    关闭则只用 BM25（未配 EMBED_API_KEY 时也走得通）
    rebuild=True   无视一致性检查强制重建索引（改了切块/解析逻辑后用这个）
    """
    if mode not in (MODE_ASK, MODE_SEARCH):
        raise QaError(f"未知 mode：{mode!r}，可选 {MODE_ASK!r} 或 {MODE_SEARCH!r}")
    if not question or not question.strip():
        raise QaError("问题为空")

    state = ensure_ready(with_vector=with_vector, rebuild=rebuild)
    _announce(state, debug)

    from .retriever import HybridRetriever

    retriever = HybridRetriever.load(with_vector=with_vector)
    retrieval = retriever.search(question, top_k=top_k, channel_debug=debug)
    if mode == MODE_SEARCH:
        return retrieval

    # 不走 LegalRAG.ask：那条路径不把 channel_debug 透给检索层，
    # debug 模式下拿不到双通道排名。这里只是同样的两步编排。
    from .generator import AnswerGenerator

    return AnswerGenerator().generate(Question(text=question, top_k=top_k), retrieval)


_announced = False


def _announce(state: ReadyState, debug: bool) -> None:
    """debug 模式下播报一次索引状况（同一进程内只播一次，免得循环调用时刷屏）。"""
    global _announced
    if debug and not _announced:
        print(f"[qa] {state.describe()}")
    _announced = True


# ================================================================== 渲染
def render(result: Answer | RetrievalResult, *, debug: bool = False) -> str:
    """把 qa() 的返回值渲染成人读文本（命令行入口与 demo 共用）。"""
    if isinstance(result, RetrievalResult):
        return result.render()
    text = result.render()
    if debug and result.retrieval is not None:
        text += "\n\n" + result.retrieval.render()
    return text
