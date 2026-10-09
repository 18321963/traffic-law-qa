from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_contracts.domain.answer import Answer, Evidence, Question
from rag_contracts.domain.disk import ParentChunk
from rag_contracts.domain.laws import laws_of
from rag_contracts.domain.retrieval import MaterialPassage, RetrievalResult, RetrievedArticle

ROAD = "中华人民共和国道路交通安全法"
PENALTY = "深圳经济特区道路交通安全违法行为处罚条例"

ARTICLE_TEXT = (
    "醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；"
    "五年内不得重新取得机动车驾驶证。饮酒后驾驶营运机动车的，处十五日拘留，并处五千元罚款，"
    "吊销机动车驾驶证，五年内不得重新取得机动车驾驶证。"
    "因饮酒后驾驶机动车被处罚，再次饮酒后驾驶机动车的，处十日以下拘留，并处一千元以上二千元以下罚款。"
)

ARTICLE = ParentChunk(
    parent_id=f"{ROAD}#第九十一条",
    law_id="road_traffic_safety",
    law_name=ROAD,
    version="2021",
    citation=f"《{ROAD}》",
    article_no="第九十一条",
    article_index=91,
    chapter="第七章 法律责任",
    section=None,
    text=ARTICLE_TEXT,
    refs=("第九十条",),
)
OTHER = ParentChunk(
    parent_id=f"{PENALTY}#第九十条",
    law_id="shenzhen_penalty",
    law_name=PENALTY,
    version="2022",
    citation=f"《{PENALTY}》",
    article_no="第九十条",
    article_index=90,
    chapter=None,
    section=None,
    text="驾驶电动自行车未佩戴安全头盔的，处警告或者五十元罚款。",
)
PARENTS = {ARTICLE.parent_id: ARTICLE, OTHER.parent_id: OTHER}

MATERIALS = (
    MaterialPassage(
        label="[材料1]",
        doc_id="d0",
        display_name="车辆管理规定.md",
        index=0,
        text="培训费用按每人每年一千二百元包干，超出部分由所在部门承担。",
    ),
    MaterialPassage(
        label="[材料2]",
        doc_id="d0",
        display_name="车辆管理规定.md",
        index=1,
        text="车辆保险由行政部统一办理，驾驶员不得自行指定保险公司。",
    ),
)

NOTES = (
    "口语对齐：醉驾 → 醉酒驾驶",
    "法名线索：道路交通安全法（命中法规的分数 ×1.5）",
    "未重排：关掉了",
)


STREAM_PARTS = ("答案", "正文[依据1]")


def answer_frames(answer: Answer, parts: Sequence[str] = STREAM_PARTS):
    for part in parts:
        yield "delta", part
    yield "answer", answer


class AgentStubService:

    llm_ready = True
    model_name = "stub-model"

    def __init__(self, *, top_k: int = 6) -> None:
        self.top_k = top_k
        self.parents = dict(PARENTS)
        self.searches: list[dict] = []
        self.answers: list[dict] = []
        self.streams: list[dict] = []
        self.lookups: list[tuple[str, str | None]] = []
        self.material_calls: list[tuple[str, tuple[str, ...], int]] = []
        self.tool_timeouts: list[tuple[str, float | None]] = []

    def laws(self):
        return laws_of(self.parents.values())

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        law_filter: tuple[str, ...] = (),
        candidates: int | None = None,
        timeout: float | None = None,
    ) -> RetrievalResult:
        self.tool_timeouts.append(("search", timeout))
        self.searches.append(
            {
                "question": question,
                "top_k": top_k,
                "channel_debug": channel_debug,
                "law_filter": tuple(law_filter),
                "candidates": candidates,
            }
        )
        return RetrievalResult(
            query=question,
            articles=(
                RetrievedArticle(
                    article=ARTICLE,
                    score=0.912345,
                    vector_rank=1,
                    bm25_rank=2,
                    vector_score=0.712345,
                    bm25_score=3.5,
                    hit_chunks=("c1", "c2"),
                    law_hint="道路交通安全法",
                ),
                RetrievedArticle(article=OTHER, score=0.4),
            ),
            used_vector=True,
            used_bm25=True,
            elapsed_ms=12.5,
            notes=NOTES,
            matched_text=f"{question} 醉酒驾驶",
        )

    def get_article(self, article_no: str, law_name: str | None = None, *, timeout=None):
        from rag_service.query.articles import build_article_index, lookup_article

        self.tool_timeouts.append(("get_article", timeout))
        self.lookups.append((article_no, law_name))
        return lookup_article(
            article_no, law_name, parents=self.parents, index=build_article_index(self.parents)
        )

    def materials(self, doc_ids):
        return MATERIALS if doc_ids else ()

    def search_materials(self, query: str, doc_ids, *, top_k: int = 5, timeout=None):
        from rag_service.query.materials import search_materials as score_materials

        self.tool_timeouts.append(("search_materials", timeout))
        self.material_calls.append((query, tuple(doc_ids), top_k))
        return score_materials(query, self.materials(doc_ids), top_k=top_k)

    def ask(self, question, top_k=None, *, channel_debug=False) -> Answer:
        asked = question if isinstance(question, Question) else Question(text=question)
        return self.answer(
            asked, self.search(asked.text, top_k or asked.top_k, channel_debug=channel_debug)
        )

    def answer(self, question, retrieval, *, timeliness=(), materials=()) -> Answer:
        self.answers.append(
            {
                "question": question,
                "retrieval": retrieval,
                "timeliness": tuple(timeliness),
                "materials": tuple(materials),
            }
        )
        return self._answer(question, retrieval, timeliness, materials)

    def answer_stream(
        self, question, retrieval, *, timeliness=(), materials=(), parts=STREAM_PARTS
    ):
        self.streams.append(
            {
                "question": question,
                "retrieval": retrieval,
                "timeliness": tuple(timeliness),
                "materials": tuple(materials),
            }
        )
        return answer_frames(self._answer(question, retrieval, timeliness, materials), parts)

    def stream(self, question, retrieval, *, timeliness=(), materials=()):
        yield from answer_frames(self._answer(question, retrieval, timeliness, materials))

    def _answer(self, question, retrieval, timeliness, materials) -> Answer:
        evidences = tuple(
            Evidence(
                label=f"【依据{index}】",
                citation=hit.citation,
                text=hit.article.text,
                score=hit.score,
                article=hit.article,
            )
            for index, hit in enumerate(retrieval.articles, start=1)
        )
        return Answer(
            question=question.text,
            text="答案正文[依据1]",
            evidences=evidences,
            model="stub-model",
            elapsed_ms=3.5,
            usage={"total_tokens": 9},
            retrieval=retrieval,
            notes=("桩生成器：没花真调用",),
            timeliness=tuple(timeliness),
            materials=tuple(materials),
        )


class AgentStubLLM:

    def __init__(
        self, replies=(), *, model="stub", available=True, finishes=(), tokens=3
    ) -> None:
        self._replies = list(replies)
        self._finishes = list(finishes)
        self.model_name = model
        self.available = available
        self.tokens = tokens
        self.prompts: list[list[dict]] = []
        self.offered: list[set[str]] = []

    def chat(self, messages, *, tools=None, temperature=None, name="llm.chat"):
        if not self.available:
            raise AssertionError("这个模型不可用，不该被调用")
        self.prompts.append(list(messages))
        self.offered.append({tool["function"]["name"] for tool in tools or ()})
        reply = self._replies.pop(0) if self._replies else {"role": "assistant", "content": "够了"}
        finish = self._finishes.pop(0) if self._finishes else "stop"
        return reply, {"total_tokens": self.tokens}, finish


@pytest.fixture
def agent_rag_server():
    pytest.importorskip("fastapi", reason="api extra 没装：agent 侧 HTTP 验收跳过")
    pytest.importorskip("httpx", reason="TestClient 要 httpx（dev extra）")

    from rag_service.api import app as server
    from rag_service.api.runtime import Runtime
    from rag_service.indexing.readiness import ReadyState

    rt = Runtime(
        ready=ReadyState(
            action="reuse",
            reason="",
            rows=812,
            laws=2,
            articles=2,
            parents=2,
            dense=True,
            milvus="v2.6.24",
        ),
        rag=AgentStubService(),
        boot_ms=0.0,
    )
    server.app.state.rt = rt
    server.app.state.boot_error = None
    return server.app, rt


@pytest.fixture
def agent_wire(agent_rag_server):
    from fastapi.testclient import TestClient

    from api_contracts import RagClient

    app, rt = agent_rag_server
    return RagClient("http://testserver", client=TestClient(app)), rt, app


def make_pdf(pages: list[list[str]]) -> bytes:
    import io

    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        ByteStringObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    cmap = DecodedStreamObject()
    cmap.set_data(
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Adobe-Identity-UCS def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        b"1 beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\n"
        b"endcmap\n"
        b"CMapName currentdict /CMap defineresource pop\n"
        b"end\n"
        b"end\n"
    )
    descendant = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/CIDFontType2"),
            NameObject("/BaseFont"): NameObject("/SimSun"),
            NameObject("/CIDSystemInfo"): DictionaryObject(
                {
                    NameObject("/Registry"): ByteStringObject(b"Adobe"),
                    NameObject("/Ordering"): ByteStringObject(b"Identity"),
                    NameObject("/Supplement"): NumberObject(0),
                }
            ),
            NameObject("/DW"): NumberObject(1000),
        }
    )
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type0"),
            NameObject("/BaseFont"): NameObject("/SimSun"),
            NameObject("/Encoding"): NameObject("/Identity-H"),
            NameObject("/DescendantFonts"): ArrayObject([writer._add_object(descendant)]),
            NameObject("/ToUnicode"): writer._add_object(cmap),
        }
    )
    font_ref = writer._add_object(font)

    for lines in pages:
        page = writer.add_blank_page(width=595, height=842)
        ops = ["BT", "/F1 12 Tf", "72 780 Td"]
        for index, line in enumerate(lines):
            if index:
                ops.append("0 -24 Td")
            ops.append(f"<{line.encode('utf-16-be').hex().upper()}> Tj")
        ops.append("ET")
        content = DecodedStreamObject()
        content.set_data("\n".join(ops).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def pdf_maker():
    pytest.importorskip("pypdf", reason="pdf extra 没装：合成 PDF 夹具跳过")
    return make_pdf
