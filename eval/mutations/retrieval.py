from __future__ import annotations

__all__ = ["MUTATIONS"]

MUTATIONS = [
    (
        "M1 池子召回退化成 hit@6",
        "eval/harness.py",
        "        return sum(1 for r in rows if r.hit)\n",
        "        return sum(1 for r in rows if r.hit_within(max(KS)))\n",
        [
            "tests/test_harness.py::test_pool_recall_counts_the_gold_that_hit_at_6_leaves_out",
            "tests/test_harness.py::test_mrr_is_locked_at_the_top_k_cutoff",
        ],
    ),
    (
        "M2 MRR 吃进池子里 6 名以外的名次",
        "eval/harness.py",
        "        return sum(1.0 / r.rank for r in rows if r.hit_within(cutoff)) / len(rows)\n",
        "        return sum(1.0 / r.rank for r in rows if r.rank is not None) / len(rows)\n",
        [
            "tests/test_harness.py::test_mrr_is_locked_at_the_top_k_cutoff",
            "tests/test_harness.py::test_pool_recall_counts_the_gold_that_hit_at_6_leaves_out",
        ],
    ),
    (
        "M3 未命中清单按池子口径算",
        "eval/harness.py",
        "            missed = [r for r in self.results if not r.hit_within(max(KS))]\n",
        "            missed = [r for r in self.results if not r.hit]\n",
        ["tests/test_harness.py::test_show_misses_counts_pool_only_hits_as_misses"],
    ),
    (
        "M4 通道没开也按没召回计",
        "eval/harness.py",
        "        known = [flag for flag in flags if flag is not None]\n",
        "        known = flags\n",
        [
            "tests/test_harness.py::test_channel_recall_skips_a_channel_that_never_ran",
            "tests/test_harness.py::test_dense_line_says_off_when_the_channel_never_ran",
        ],
    ),
    (
        "M5 通道名次退回「组内最佳融合子块」",
        "rag_service/query/retriever.py",
        "        ranked = [\n            (ranks.get(chunk_id, {}).get(channel), chunk_id)\n"
        "            for chunk_id, _ in group\n"
        "            if ranks.get(chunk_id, {}).get(channel) is not None\n        ]\n",
        "        ranked = (\n            [(ranks.get(best_id, {}).get(channel), best_id)]\n"
        "            if ranks.get(best_id, {}).get(channel) is not None\n            else []\n        )\n",
        ["tests/test_harness.py::test_a_channel_rank_is_the_articles_best_chunk_in_that_channel"],
    ),
    (
        "M6 并列不再按命中子块数排",
        "rag_service/query/retriever.py",
        "        articles.sort(key=lambda item: (item.score, len(item.hit_chunks)), reverse=True)\n",
        "        articles.sort(key=lambda item: item.score, reverse=True)\n",
        ["tests/test_harness.py::test_equal_fusion_scores_are_still_broken_by_the_number_of_hit_chunks"],
    ),
]
