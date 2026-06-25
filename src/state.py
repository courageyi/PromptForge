"""
PromptForge 状态定义

LangGraph 使用 TypedDict 定义状态 Schema，每个字段可指定 Reducer 函数。
add_messages 是 LangGraph 内置的 Reducer，自动完成消息列表的追加（而非覆盖）。
"""

from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langgraph.graph.message import add_messages


class PromptForgeState(TypedDict):
    """
    PromptForge 工作流的全局状态。

    所有节点共享此状态，节点通过返回 dict 来更新部分字段。
    未在返回值中出现的字段保持不变。
    """

    # ===========================
    # 用户输入
    # ===========================
    user_input: str
    """用户的原始需求描述，例如 '帮我写一个分析评论情感的提示词'"""

    framework: str
    """期望的提示词框架: few-shot | cot | react | tot | custom"""

    target_model: str
    """目标大模型: gpt-4 | gpt-4o | claude-opus-4-8 | claude-sonnet-4-6 | deepseek-v3"""

    auto_optimize: bool
    """是否启用自动迭代优化（不需要每次人工审批）"""

    # ===========================
    # 分析阶段输出
    # ===========================
    intent: str
    """分析出的用户意图分类: summarization|translation|classification|generation|coding|reasoning|chat"""

    constraints: List[str]
    """提取的约束条件列表，如 ['输出JSON', '限制200字内', '使用正式语气']"""

    output_format: str
    """期望的输出格式描述: json|markdown|text|code|table"""

    # ===========================
    # 检索阶段输出
    # ===========================
    similar_templates: List[Dict[str, Any]]
    """从向量数据库检索到的相似高质量提示词模板 Top-K"""

    # ===========================
    # 生成阶段输出
    # ===========================
    draft_prompt: str
    """当前迭代轮次的提示词（初始为初版，优化后覆盖）"""

    generation_metadata: Dict[str, Any]
    """生成元信息: 使用的框架、模板ID、模型、token用量等"""

    # ===========================
    # 审查阶段输出
    # ===========================
    critique: Dict[str, Any]
    """多维度审查结果
    {
        "clarity": {"score": 8, "comment": "..."},
        "completeness": {"score": 7, "comment": "..."},
        "actionability": {"score": 9, "comment": "..."},
        "robustness": {"score": 6, "comment": "..."},
        "safety": {"score": 10, "comment": "..."}
    }
    """

    quality_score: float
    """综合质量评分 0-10，由 Critic 加权计算"""

    # ===========================
    # 优化阶段输出
    # ===========================
    optimized_prompt: str
    """优化后的提示词"""

    optimization_history: List[Dict[str, Any]]
    """优化历史记录，每次迭代追加一条
    [
        {"round": 1, "before_score": 6.5, "after_score": 7.8, "changes": "..."},
        ...
    ]
    """

    # ===========================
    # 测试阶段输出
    # ===========================
    test_cases: List[Dict[str, Any]]
    """测试用例列表
    [
        {"input": "这个产品太烂了", "expected_sentiment": "negative"},
        ...
    ]
    """

    test_results: List[Dict[str, Any]]
    """测试结果
    [
        {"input": "...", "output": "...", "expected": "...", "pass": true, "latency_ms": 1200},
        ...
    ]
    """

    final_prompt: str
    """最终产出的优化后提示词"""

    # ===========================
    # 控制字段
    # ===========================
    messages: Annotated[list, add_messages]
    """消息历史。使用 add_messages Reducer，调用 LLM 时自动追加。
    注意: 此字段由 LangGraph 的 LLM 调用自动管理，一般不需要手动赋值。
    """

    iteration_count: int
    """当前优化迭代次数，初始为 0，每次 Optimizer 执行后 +1"""

    next_action: str
    """由 Router 节点设置，决定下一次路由目标"""

    error: Optional[str]
    """全局错误信息，任何节点可设置此字段触发错误处理"""
