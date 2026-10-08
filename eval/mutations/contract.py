from __future__ import annotations

__all__ = ["MUTATIONS"]

TESTS = ["tests/test_tool_parity.py", "tests/test_openapi_contract.py"]

MUTATIONS = [
    (
        "M1 端点载荷不吐法条正文",
        "rag_contracts/domain/retrieval.py",
        '            "article": self.article.to_dict(),\n',
        "",
        TESTS,
    ),
    (
        "M2 载荷不带 matched_text",
        "rag_contracts/domain/retrieval.py",
        '            "matched_text": self.matched_text,\n',
        "",
        TESTS,
    ),
    (
        "M3 from_dict 丢掉 matched_text",
        "rag_contracts/domain/retrieval.py",
        '            matched_text=str(data.get("matched_text") or ""),\n',
        '            matched_text="",\n',
        TESTS,
    ),
    (
        "M4 载荷不带 notes",
        "rag_contracts/domain/retrieval.py",
        '            "notes": list(self.notes),\n',
        "",
        TESTS,
    ),
    (
        "M5 to_dict 不带命中片段 id",
        "rag_contracts/domain/retrieval.py",
        '            "hit_chunks": list(self.hit_chunks),\n',
        "",
        TESTS,
    ),
    (
        "M6 /articles/lookup 查不到就 404",
        "rag_service/api/routes.py",
        '    payload["found"] = not result.is_empty\n',
        "    if result.is_empty:\n        raise HTTPException(status_code=404, detail=note)\n"
        '    payload["found"] = True\n',
        TESTS,
    ),
    (
        "M7 /qa 丢掉 debug",
        "rag_service/api/app.py",
        "        channel_debug=req.debug,\n        law_filter=tuple(req.law_filter),\n",
        "        channel_debug=False,\n        law_filter=tuple(req.law_filter),\n",
        TESTS,
    ),
    (
        "M8 /qa 丢掉 pool",
        "rag_service/api/app.py",
        "        mode=req.mode,\n        top_k=req.pool or req.top_k,\n",
        "        mode=req.mode,\n        top_k=req.top_k,\n",
        TESTS,
    ),
    (
        "M9 materials 端点不吐段文本",
        "rag_service/api/routes.py",
        '        "passages": [hit.to_dict() for hit in hits],\n',
        '        "passages": [],\n',
        TESTS,
    ),
    (
        "M10 答案载荷不吐依据",
        "rag_contracts/domain/answer.py",
        '            "evidences": [e.to_dict() for e in self.evidences],\n',
        "",
        ["tests/test_tool_parity.py"],
    ),
    (
        "M11 /answer 丢掉请求里的检索结果，自己重检索",
        "rag_service/api/app.py",
        "    retrieval = RetrievalResult.from_dict(req.retrieval)\n",
        "    retrieval = rt.rag.search(req.question.text)\n",
        ["tests/test_tool_parity.py"],
    ),
    (
        "M12 Evidence 载荷带回了法条正文",
        "rag_contracts/domain/answer.py",
        '            "score": round(self.score, 6),\n        }\n',
        '            "score": round(self.score, 6),\n'
        '            "article": None if self.article is None else self.article.to_dict(),\n'
        "        }\n",
        ["tests/test_contracts_architecture.py"],
    ),
]
