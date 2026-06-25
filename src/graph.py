"""
PromptForge 核心工作流 — LangGraph 状态图定义

这是整个项目的心脏。LangGraph 的 StateGraph 将多个 LLM Agent 编排为
有状态、有分支、可中断的有向图。

运行方式：
    from src.graph import PromptForgeGraph

    graph = PromptForgeGraph()
    result = graph.invoke({"user_input": "写一个情感分析提示词"})
    print(result["final_prompt"])
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from .state import PromptForgeState
from .nodes.analyzer import analyzer_node
from .nodes.retriever import retriever_node
from .nodes.generator import generator_node
from .nodes.critic import critic_node
from .nodes.optimizer import optimizer_node
from .nodes.tester import tester_node


# ============================================================
# 条件路由函数
# ============================================================

def route_after_critic(state: PromptForgeState) -> Literal["tester", "optimizer", "human_review"]:
    """
    根据 Critic 的打分决定下一步。

    路由逻辑：
    - quality_score >= 8.0 → 直接进入测试
    - 已迭代 >= 3 次且仍不达标 → 人工介入
    - auto_optimize = True → 进入优化器
    - 否则 → 人工审核
    """
    score = state.get("quality_score", 0)
    iteration = state.get("iteration_count", 0)

    if score >= 8.0:
        return "tester"
    elif iteration >= 3:
        return "human_review"
    elif state.get("auto_optimize", False):
        return "optimizer"
    else:
        return "human_review"


def route_after_optimizer(state: PromptForgeState) -> Literal["critic", "human_review"]:
    """
    优化完成后，决定是重新审查还是人工介入。

    如果迭代次数 >= 3，不再进入优化循环，直接人工介入。
    """
    if state.get("iteration_count", 0) >= 3:
        return "human_review"
    return "critic"


# ============================================================
# 图构建
# ============================================================

def build_promptforge_graph() -> StateGraph:
    """
    构建 PromptForge 的 LangGraph 工作流。

    图结构（详见 ARCHITECTURE.md）:

        START → Analyzer → Retriever → Generator → Critic ─┬─→ Tester → END
                                                           │
                                                           ├─→ Optimizer ─┐
                                                           │               │
                                                           │   (循环)      │
                                                           │               │
                                                           └─→ HumanReview─┘
    """
    # 创建 StateGraph，传入状态类型
    workflow = StateGraph(PromptForgeState)

    # ---- 添加节点 ----
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("optimizer", optimizer_node)
    workflow.add_node("tester", tester_node)

    # Human-in-the-loop 节点：使用 interrupt 暂停执行，等待人工审批
    workflow.add_node("human_review", human_review_node)

    # ---- 添加边 ----
    # 固定边（无分支，永远走这条路）
    workflow.add_edge("analyzer", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "critic")

    # 条件边（根据审查结果路由）
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "tester": "tester",
            "optimizer": "optimizer",
            "human_review": "human_review",
        }
    )

    # 优化器完成后 → 重新审查 或 人工介入
    workflow.add_conditional_edges(
        "optimizer",
        route_after_optimizer,
        {
            "critic": "critic",
            "human_review": "human_review",
        }
    )

    # 人工审核通过后 → 测试
    workflow.add_edge("human_review", "tester")

    # 测试完成后 → 结束
    workflow.add_edge("tester", END)

    # ---- 设置入口 ----
    workflow.set_entry_point("analyzer")

    return workflow


# ============================================================
# Human-in-the-Loop 节点
# ============================================================

def human_review_node(state: PromptForgeState) -> dict:
    """
    人工审核节点。

    使用 LangGraph 的 interrupt() 函数暂停图执行，
    将控制权交还给调用方（通常是 Streamlit 前端）。
    人工做出决策后，调用方通过 Command(resume=...) 恢复执行。

    interrupt 是 LangGraph 的核心人机协作机制：
    - 暂停时图的状态被持久化到 MemorySaver
    - 可以随时恢复，不会丢失上下文
    - 支持修改状态后恢复
    """
    # interrupt 抛出 GraphInterrupt 异常，暂停执行
    # 调用方捕获后展示给人工审核界面
    human_decision = interrupt({
        "message": "提示词需要人工审核",
        "draft_prompt": state.get("draft_prompt", ""),
        "optimized_prompt": state.get("optimized_prompt", ""),
        "critique": state.get("critique", {}),
        "quality_score": state.get("quality_score", 0),
        "iteration_count": state.get("iteration_count", 0),
    })

    # 人工做出决定后，执行从此处恢复
    # human_decision 是调用方通过 Command(resume=...) 传入的值
    action = human_decision.get("action", "approve")

    if action == "approve":
        # 人工批准，使用优化版或草稿版
        chosen = human_decision.get("chosen_prompt",
                                     state.get("optimized_prompt") or state.get("draft_prompt"))
        return {
            "final_prompt": chosen,
            "next_action": "tester",
        }
    elif action == "reject":
        # 人工拒绝，可能需要重新生成
        return {
            "draft_prompt": "",  # 清空，触发重新生成
            "next_action": "generator",
        }
    else:
        # 人工手动修改
        return {
            "final_prompt": human_decision.get("edited_prompt", ""),
            "next_action": "tester",
        }


# ============================================================
# 封装类
# ============================================================

class PromptForgeGraph:
    """
    PromptForge 工作流封装。

    使用方式：
        graph = PromptForgeGraph()

        # 方式1：同步执行（不需要人工审核时）
        result = graph.invoke({
            "user_input": "帮我写一个情感分析提示词",
            "framework": "few-shot",
            "target_model": "gpt-4",
            "auto_optimize": True,
            "iteration_count": 0,
        })

        # 方式2：流式执行（逐步观察结果）
        for event in graph.stream(initial_state):
            print(event)

        # 方式3：带人工审核
        config = {"configurable": {"thread_id": "session-123"}}
        # 第一次调用会停在 human_review 节点
        result = graph.invoke(initial_state, config)
        # 人工审核后恢复
        graph.resume(config, {"action": "approve"})
    """

    def __init__(self, checkpoint_saver=None):
        """
        初始化工作流。

        Args:
            checkpoint_saver: LangGraph 检查点存储器。
                            默认为 MemorySaver（内存），
                            生产环境用 SqliteSaver / PostgresSaver。
        """
        self._graph = build_promptforge_graph()

        # MemorySaver 持久化每个节点的状态快照
        # 支持：断点续跑、时光旅行、人工审核暂停恢复
        self._checkpointer = checkpoint_saver or MemorySaver()

        # compile 将图编译为可执行对象
        # interrupt_before 在指定节点前暂停（此处不需要）
        self._app = self._graph.compile(checkpointer=self._checkpointer)

    def invoke(self, initial_state: dict, config: dict = None):
        """
        同步执行工作流。

        Args:
            initial_state: 初始状态（至少包含 user_input）
            config: LangGraph 配置，包含 thread_id 用于多会话隔离

        Returns:
            最终状态 dict
        """
        return self._app.invoke(initial_state, config)

    def stream(self, initial_state: dict, config: dict = None):
        """
        流式执行工作流，每次节点完成后 yield 其输出。

        用于前端实时展示进度。
        """
        return self._app.stream(initial_state, config)

    def resume(self, config: dict, resume_value: dict):
        """
        从中断点恢复执行（Human-in-the-Loop 场景）。

        Args:
            config: 与 invoke 时相同的 config
            resume_value: 传给 interrupt() 的返回值
        """
        return self._app.invoke(Command(resume=resume_value), config)

    def get_state(self, config: dict):
        """获取当前会话的状态快照"""
        return self._app.get_state(config)

    def get_state_history(self, config: dict):
        """获取状态历史（时光旅行），可回溯到任意历史状态"""
        return list(self._app.get_state_history(config))

    @property
    def graph(self):
        """返回编译后的图，用于可视化"""
        return self._app


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 快速测试（需要设置 OPENAI_API_KEY）
    graph = PromptForgeGraph()

    initial_state: PromptForgeState = {
        "user_input": "帮我写一个分析用户评论情感倾向的提示词，输出JSON格式，包含 sentiment 和 confidence 字段",
        "framework": "few-shot",
        "target_model": "gpt-4",
        "auto_optimize": True,
        "iteration_count": 0,
        # 以下字段由节点填充，初始化为空
        "intent": "",
        "constraints": [],
        "output_format": "",
        "similar_templates": [],
        "draft_prompt": "",
        "generation_metadata": {},
        "critique": {},
        "quality_score": 0.0,
        "optimized_prompt": "",
        "optimization_history": [],
        "test_cases": [],
        "test_results": [],
        "final_prompt": "",
        "messages": [],
        "next_action": "",
        "error": None,
    }

    # 流式执行，观察每个节点的输出
    print("=" * 60)
    print("PromptForge 工作流启动")
    print("=" * 60)

    for event in graph.stream(initial_state):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        print(f"\n>>> 节点 [{node_name}] 完成:")
        # 只打印关键字段变化
        for key, value in node_output.items():
            if value and key not in ("messages",):
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"    {key}: {val_str}")

    print("\n" + "=" * 60)
    print("工作流执行完毕")
    print("=" * 60)
