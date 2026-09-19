"""Agentic RAG：在管道之上加一层「自己决定查什么」的循环。

    graph.py   LangGraph 状态图 + AgentRunner + ToolCallingLLM + render_trace
    tools.py   agent 的两个工具：search_law / get_article

**这个 `__init__` 刻意什么都不 import。** `graph.py` 会拉 langgraph，而
`import traffic_law_rag` 有一条测试钉着「绝不能把 langgraph 拖进来」——
只要这里写一句 `from .graph import AgentRunner`，那条边界立刻就破了。
要什么就 `from traffic_law_rag.agent.graph import ...`。

`tools.py` 反过来是**不依赖 langgraph** 的（有测试钉住），可以单独用。
"""
