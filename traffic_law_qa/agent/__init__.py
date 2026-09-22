"""Agentic RAG：在管道之上加一层「自己决定查什么」的循环。

    graph.py    图的装配（4 节点 + 2 路由）与门面 AgentRunner  ← 唯一拉 langgraph 的模块
    state.py    AgentState（单独成文件是硬约束，见其自述）
    region.py   入口：这道题该按哪个地区的规定回答（判出检索作用域）
    nodes.py    规划轮（自己判断够不够 + 调工具）、工具执行轮、收尾轮
    prompts.py  两份提示词
    llm.py      带工具调用能力的客户端
    trace.py    终态 → 人读的决策链（纯函数）
    cli.py      `python -m traffic_law_qa.agent` 的入口
    tools.py    两个工具：search_law / get_article
    langfuse_tracer.py  可选的云端观测：节点状态 / 模型调用 / 检索 → Langfuse

**这个 `__init__` 刻意什么都不 import。** 只要这里写一句
`from .graph import AgentRunner`，`import traffic_law_qa` 就会把 langgraph 拖进来。
要什么就 `from traffic_law_qa.agent.<模块> import ...`。

**只有 `graph.py` 拉 langgraph**，其余模块都不拉，可以顶层导入。
"""
