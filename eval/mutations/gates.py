from __future__ import annotations

__all__ = ["PROBES"]

TESTS = ["tests/test_architecture.py", "tests/test_contracts_architecture.py"]

PROBES = [
    (
        "门禁① 扫描根不存在时自己先炸（不是静默放行）",
        "tests/test_architecture.py",
        "PACKAGE = Path(rag_service.__file__).resolve().parent\n",
        'PACKAGE = Path(rag_service.__file__).resolve().parent / "no_such_dir"\n',
        TESTS,
    ),
    (
        "门禁② rag 侧顶层 import agent_service 必须翻红（反向门）",
        "rag_service/cli/ask.py",
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nfrom agent_service.prompts import AGENT_SYSTEM_PROMPT\n",
        TESTS,
    ),
    (
        "门禁③ indexing/ 里自己 new 一个 MilvusStore 必须翻红（装配门）",
        "rag_service/indexing/indexer.py",
        "        self.store = store\n",
        "        self.store = store or MilvusStore()\n",
        TESTS,
    ),
    (
        "门禁④ 顶层包文件用 `..` 上溯到包外必须翻红（越界门）",
        "eval/harness.py",
        "from rag_contracts import config\n",
        "from .. import config\n",
        TESTS,
    ),
    (
        "门禁⑤ 契约包顶层 import 重依赖必须翻红（轻依赖门）",
        "rag_contracts/config.py",
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nimport torch\n",
        TESTS,
    ),
    (
        "门禁⑥ 契约包反向 import 消费者必须翻红（方向门）",
        "rag_contracts/ports.py",
        "from .domain.answer import Answer, Question\n",
        "from .domain.answer import Answer, Question\nfrom rag_service import container\n",
        TESTS,
    ),
    (
        "门禁⑦ 顶层包没写进 packages.find.include 必须翻红（打包面门）",
        "pyproject.toml",
        'include = ["rag_contracts*", "rag_service*", "agent_service*", "eval*", "mcp_server*", "api_contracts*"]\n',
        'include = ["rag_contracts*", "rag_service*", "eval*", "mcp_server*", "api_contracts*"]\n',
        TESTS,
    ),
    (
        "门禁⑧ rag 镜像的 extras 丢掉 milvus 必须翻红（装机路径门）",
        "rag_service/Dockerfile",
        '".[api,local,milvus,pdf]"',
        '".[api,local,pdf]"',
        TESTS,
    ),
    (
        "门禁⑨ `milvus` 组被清空必须翻红（依赖组门）",
        "pyproject.toml",
        'milvus = [\n    "pymilvus>=2.6,<3",\n]\n',
        "milvus = []\n",
        TESTS,
    ),
    (
        "门禁⑩ eval 侧长出一条没登记的 rag_service 直取边必须翻红（评测边门）",
        "eval/singlehop.py",
        "        from rag_service.api.facade import QaError\n",
        "        from rag_service.api.facade import QaError\n        from rag_service.query.rag import LegalRAG\n",
        TESTS,
    ),
    (
        "门禁⑪ 前端认一个后端从不发的帧名必须翻红（帧词汇门）",
        "frontend/src/api/sse.ts",
        '  if (frame.event === "step") return { type: "step", data: data as { node: string } };\n',
        '  if (frame.event === "progress") return { type: "step", data: data as { node: string } };\n'
        '  if (frame.event === "step") return { type: "step", data: data as { node: string } };\n',
        TESTS,
    ),
    (
        "门禁⑫ 台账漏出 .dockerignore 必须翻红（本机状态不进镜像门）",
        ".dockerignore",
        "data/*.db\n",
        "",
        TESTS,
    ),
    (
        "门禁⑬ 源文件里写一行散文注释必须翻红（零注释零 docstring 门）",
        "rag_contracts/config.py",
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\n# 这里解释一下临时情况\n",
        TESTS,
    ),
    (
        "门禁⑭ 定义 USAGE 却没人用必须翻红（用法文本不许烂在文件里）",
        "mcp_server/main.py",
        "        description=USAGE,\n",
        '        description="",\n',
        TESTS,
    ),
    (
        "门禁⑮ 六包白名单少一个包必须翻红（拓扑冻结门）",
        "tests/test_contracts_architecture.py",
        'FROZEN_PACKAGES = ("agent_service", "api_contracts", "eval", "mcp_server", "rag_contracts", "rag_service")\n',
        'FROZEN_PACKAGES = ("api_contracts", "eval", "mcp_server", "rag_contracts", "rag_service")\n',
        TESTS,
    ),
    (
        "门禁⑯ 自检文案引了 README 里没有的小节必须翻红（硬引门）",
        "eval/harness.py",
        "README「入口」",
        "README「入口一览」",
        TESTS,
    ),
]
