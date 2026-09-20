"""Agentic RAG：在管道之上加一层「自己决定查什么」的循环。

    graph.py    图的装配（6 节点 + 4 路由）与门面 AgentRunner  ← 唯一拉 langgraph 的模块
    state.py    AgentState（单独成文件是硬约束，见其自述）
    intent.py   意图识别：规则直接取条 / 走混合检索
    nodes.py    规划轮、工具执行轮、收尾轮
    reflect.py  审核轮：够不够、补不补得上
    prompts.py  两份提示词
    llm.py      带工具调用能力的客户端
    trace.py    终态 → 人读的决策链（纯函数）
    cli.py      `python -m traffic_law_qa.agent` 的入口
    tools.py    两个工具：search_law / get_article

**这个 `__init__` 刻意什么都不 import。** 只要这里写一句
`from .graph import AgentRunner`，`import traffic_law_qa` 就会把 langgraph 拖进来 ——
有测试钉着那条边界，它立刻就红。要什么就 `from traffic_law_qa.agent.<模块> import ...`。

**只有 `graph.py` 拉 langgraph**，其余模块都不拉，可以顶层导入（`tests/test_multihop.py`
的 AST 断言钉着这条）。
"""
