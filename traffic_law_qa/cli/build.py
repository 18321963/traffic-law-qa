from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import config, container
from ..infra.milvus import MilvusError, wait_until_ready

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_law_qa.cli.build",
        description='建库与查状态（问答走 python -m traffic_law_qa "问题"）',
    )
    subs = parser.add_subparsers(
        dest="command", required=True, metavar="{build,status,layers,docx,parse,chunk,index}"
    )

    build = subs.add_parser(
        "build",
        help="整条建库：parse → chunk → index（手工把法规放进 docx/ 后跑这个；"
        "HTTP 上传走 POST /documents + POST /reindex，不用手工跑）",
    )
    build.add_argument("--force", action="store_true", help="无视 sha1 增量门控，全部重新解析")
    build.add_argument("--no-vector", action="store_true", help="只建 BM25 字段，不算向量")

    subs.add_parser("status", help="看库内规模、索引快照与 Milvus 连接")
    subs.add_parser("layers", help="看七层管道各自的输入输出")

    docx = subs.add_parser("docx", help="单步：源文件 → 段落（不给路径就跑 SOURCE_DIR 全部）")
    docx.add_argument("paths", nargs="*", metavar="源文件", help="要预览的源文件路径")

    parse = subs.add_parser("parse", help="单步：段落 → 法→章→节→条")
    parse.add_argument("--force", action="store_true", help="无视 sha1 门控，全部重解析")
    parse.add_argument("--only", metavar="法id", default=None, help="只解析这一部（如 sz_icv_regulation）")

    chunk = subs.add_parser("chunk", help="单步：条文 → 父子块")
    chunk.add_argument("--show", metavar="条号", default=None, help="切完打印该条号的子块（如 第八条）")

    index = subs.add_parser("index", help="单步：块 → Milvus 集合（先等 Milvus 就绪）")
    index.add_argument("--no-vector", action="store_true", help="只建 BM25 字段，不算向量")
    index.add_argument("--query", metavar="词", default=None, help="建完用纯 BM25 试查一次（top-5）")
    return parser


def _cmd_docx(paths: list[str]) -> int:
    targets = [Path(a) for a in paths] if paths else config.source_files()
    builder = container.build_pipeline(verbose=False)

    for target in targets:
        paragraphs = builder.read(target)
        chars = sum(len(p.text) for p in paragraphs)
        print(f"\n=== {target.name} | 段落 {len(paragraphs)} | 字符 {chars}")
        for para in paragraphs[:12]:
            print(f"  [{para.index:>3}] style={para.style!s:<10} {para.text[:60]}")
    return 0


def _cmd_parse(*, force: bool, only: str | None) -> int:
    container.build_pipeline().parse(force=force, only=only)
    return 0


def _cmd_chunk(show: str | None) -> int:
    chunk_set = container.build_pipeline().chunk()

    if show:
        for chunk in chunk_set.chunks:
            if chunk.article_no == show:
                print(f"\n[{chunk.chunk_id}] part {chunk.part_index + 1}/{chunk.part_total}")
                print(f"  embed_text: {chunk.embed_text}")
    return 0


def _cmd_index(*, with_vector: bool, query: str | None) -> int:
    chunk_set = container.build_chunk_set()
    indexer = container.build_indexer()
    print(f"[index] Milvus 版本 {wait_until_ready(indexer.store)}")
    try:
        indexer.build(chunk_set, with_vector=with_vector)
    except MilvusError as exc:
        print(f"[index] 失败：{exc}")
        return 1

    if query:
        store = indexer.store
        print(f"\n仅 BM25 通道：{query}")
        for chunk_id, score in store.sparse_search(query, limit=5):
            print(f"  {score:8.3f}  {chunk_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()

    if not args:
        parser.print_help()
        return 2

    try:
        options = parser.parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if options.command == "layers":
        print(container.build_pipeline().layers())
        return 0

    if options.command == "status":
        print(container.build_pipeline().status())
        return 0

    if options.command == "build":
        report = container.build_pipeline().build(force=options.force, with_vector=not options.no_vector)
        print("\n管道执行结果：")
        print(report.render())
        return 0

    if options.command == "docx":
        return _cmd_docx(options.paths)
    if options.command == "parse":
        return _cmd_parse(force=options.force, only=options.only)
    if options.command == "chunk":
        return _cmd_chunk(options.show)
    return _cmd_index(with_vector=not options.no_vector, query=options.query)


if __name__ == "__main__":
    raise SystemExit(main())
