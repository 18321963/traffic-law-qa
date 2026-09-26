from __future__ import annotations

import argparse
import sys

from ..api.app import app

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="tlq-serve",
        description="把 qa() 包成 HTTP 服务。装配在启动时做一次，请求只做检索 + 生成。",
        epilog="等价写法：python -m traffic_law_qa.cli.serve；"
        "或让 uvicorn 直接指定 app：uvicorn traffic_law_qa.api.app:app --port 8000。"
        '命令行问答那条路是 python -m traffic_law_qa "问题"；端点一览在 /docs',
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认 127.0.0.1；容器内必须显式给 0.0.0.0，否则映射出来的端口连不上）",
    )
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")

    try:
        options = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    uvicorn.run(app, host=options.host, port=options.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
