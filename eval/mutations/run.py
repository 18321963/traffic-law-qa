from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from .contract import MUTATIONS as CONTRACT
from .gates import PROBES as GATES
from .mcp import MUTATIONS as MCP
from .retrieval import MUTATIONS as RETRIEVAL

__all__ = ["main", "TABLES", "run_table"]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

USAGE = """变异自检：把一处规则改坏，跑对应测试，看它是否翻红。

    python -m eval.mutations.run all                  # 四张表全跑
    python -m eval.mutations.run retrieval            # 只跑一张（retrieval / contract / gates / mcp）
    python -m eval.mutations.run gates --only 装配    # 按名字片段挑几条

需要开发环境（pytest 在 dev 装组里；服务镜像没有——缺了会把每条都误记成翻红，所以直接停）。

**为什么是「改坏再看红」。** 测试全绿不等于覆盖：接线断、断言打歪、夹具与默认值撞车，
都能让一份没用的测试保持绿（本仓已经栽过五次）。验收标准是「把被测的那条规则改坏，
它必须翻红」，所以每条都写成「原文 → 改坏文」，跑完按字节还原（LF 锚点不中时再试 CRLF）。

    retrieval  指标口径：池子召回 / MRR@k / 未命中清单 / 通道召回 / 名次与并列取子块
    contract   契约与工具面：端点载荷字段、查不到的带内错误、/qa 的 pool 与 debug
    gates      架构门禁自身：扫描根、懒加载 agent、装配点、越界上溯
    mcp        MCP 适配层：工具参数对齐端点、原样透传、薄客户端、不碰 SDK

判读：`翻红` = 这条规则被测到了 ｜ `仍然绿 == 没检出` = 对应测试是空的（白写）
｜ `PATCH MISS` = 锚点没了（源码改过，这张表要跟着更新）。后两种都非零退出。
"""

ROOT = Path(__file__).resolve().parents[2]
TABLES = {"retrieval": RETRIEVAL, "contract": CONTRACT, "gates": GATES, "mcp": MCP}
PYTEST = [sys.executable, "-m", "pytest", "-o", "addopts=--strict-markers", "-q"]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _pytest(tests: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*PYTEST, *tests],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=ENV,
    )


def run_table(name: str, *, only: str | None = None) -> int:
    if importlib.util.find_spec("pytest") is None:
        print("没有 pytest（服务镜像的装组不含 dev）：缺了它每条都会误记成翻红，已停，请到开发环境跑。")
        return 1
    rows = [row for row in TABLES[name] if not only or only in row[0]]
    if not rows:
        print(f"[{name}] 没有匹配的变异（--only {only}）")
        return 1

    applied = green = missed = 0
    for row_name, rel, old, new, tests in rows:
        path = ROOT / rel
        original = path.read_bytes()
        old_bytes = old.encode("utf-8")
        if old_bytes not in original:
            old_bytes = old_bytes.replace(b"\n", b"\r\n")
        hits = original.count(old_bytes)
        if hits != 1:
            print(f"[{name}] {row_name}: PATCH MISS（命中 {hits} 次）")
            missed += 1
            continue

        path.write_bytes(original.replace(old_bytes, new.encode("utf-8"), 1))
        try:
            proc = _pytest(tests)
        finally:
            path.write_bytes(original)

        applied += 1
        red = proc.returncode != 0
        if not red:
            green += 1
        print(f"[{name}] {row_name}: {'翻红' if red else '仍然绿 == 没检出'}")
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.startswith(("FAILED", "ERROR", "E ")):
                print(f"    {line.strip()[:160]}")

    assert applied + missed == len(rows), f"{name} 表跑了 {applied + missed} 条，表里 {len(rows)} 条"
    print(f"[{name}] {len(rows)} 条：翻红 {applied - green} ｜ 没检出 {green} ｜ 锚点没命中 {missed}")
    return 1 if missed or green else 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args or not args:
        print(USAGE)
        return 0

    parser = argparse.ArgumentParser(prog="python -m eval.mutations.run", add_help=False)
    parser.add_argument("tables", nargs="+", choices=[*TABLES, "all"])
    parser.add_argument("--only", default=None, help="按名字片段只挑几条")
    try:
        options = parser.parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    names = list(TABLES) if "all" in options.tables else options.tables
    return 1 if any(run_table(name, only=options.only) for name in names) else 0


if __name__ == "__main__":
    raise SystemExit(main())
