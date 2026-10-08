from __future__ import annotations

from rag_contracts.domain.disk import ParentChunk
from rag_contracts.domain.laws import LawInfo, laws_of, resolve_law

ROAD = "中华人民共和国道路交通安全法"
PENAL = "中华人民共和国刑法"
SHORT = "道路交通安全法"


def _parent(law_id: str, law_name: str, article_no: str) -> ParentChunk:
    return ParentChunk(
        parent_id=f"{law_id}#{article_no}",
        law_id=law_id,
        law_name=law_name,
        version="2021",
        citation=f"《{law_name}》",
        article_no=article_no,
        article_index=1,
        chapter=None,
        section=None,
        text="条文正文。",
    )


def _laws() -> tuple[LawInfo, ...]:
    return laws_of(
        [
            _parent("road", ROAD, "第九十一条"),
            _parent("road", ROAD, "第九十条"),
            _parent("penal", PENAL, "第一百三十三条"),
        ]
    )


def test_laws_of_counts_articles_per_law() -> None:
    assert [(law.law_id, law.articles) for law in _laws()] == [("penal", 1), ("road", 2)]
    assert _laws()[1].to_dict() == {
        "law_id": "road",
        "law_name": ROAD,
        "version": "2021",
        "articles": 2,
    }


def test_a_law_name_resolves_exactly_before_it_loosely() -> None:
    laws = _laws()
    assert resolve_law(ROAD, laws)[0] == "road"
    assert resolve_law(SHORT, laws)[0] == "road"
    assert resolve_law("刑法", laws)[0] == "penal"


def test_an_unknown_or_ambiguous_name_resolves_to_nothing_and_says_why() -> None:
    laws = _laws()
    assert resolve_law("民法典", laws) == (None, sorted([ROAD, PENAL]))
    assert resolve_law("", laws) == (None, sorted([ROAD, PENAL]))
    assert resolve_law("中华人民共和国", laws)[0] is None, "同时套上两部法，不能替用户挑一部"
