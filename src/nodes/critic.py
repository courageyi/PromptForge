"""
Critic 节点 — 质量审查器

职责：
1. 从 5 个维度审查提示词质量
2. 给出每个维度的评分和改进建议
3. 计算加权综合评分

设计思考：
- Critic 和 Generator 分离是刻意为之
- Critic 使用更强的推理模型（如 o1 / claude-opus），追求审查精度
- Generator 使用更快模型（如 gpt-4o），追求生成速度
- 分离后各自 prompt 更聚焦，不会相互干扰
"""

from ..state import PromptForgeState

# ============================================================
# 审查维度定义
# ============================================================

CRITIQUE_DIMENSIONS = [
    {
        "key": "clarity",
        "name": "清晰度 (Clarity)",
        "weight": 0.25,
        "description": "指令是否明确无歧义？角色设定是否清晰？任务描述是否易于理解？"
    },
    {
        "key": "completeness",
        "name": "完整性 (Completeness)",
        "weight": 0.25,
        "description": "是否覆盖了用户所有的约束条件？边缘情况是否考虑？输出格式是否定义完整？"
    },
    {
        "key": "actionability",
        "name": "可执行性 (Actionability)",
        "weight": 0.20,
        "description": "输出格式是否可被代码解析？判断标准是否具体可操作？"
    },
    {
        "key": "robustness",
        "name": "鲁棒性 (Robustness)",
        "weight": 0.15,
        "description": "处理异常输入的能力？边界情况是否覆盖？是否有兜底策略？"
    },
    {
        "key": "safety",
        "name": "安全性 (Safety)",
        "weight": 0.15,
        "description": "是否存在 prompt injection 风险？是否会引导模型输出有害内容？"
    },
]

CRITIC_SYSTEM_PROMPT = """你是一个严苛的提示词质量评审专家。你会从多个维度审查提示词并提出改进意见。

评分标准：
- 9-10: 优秀，几乎不需要修改
- 7-8:  良好，有小的改进空间
- 5-6:  及格，有明显缺陷
- 3-4:  较差，核心逻辑有问题
- 1-2:  很差，基本不可用

审查原则：
1. 给出具体改进建议，而非泛泛而谈
2. 指出问题时要给出修改示例
3. 正面反馈和批评并重，不只看问题

返回 JSON 格式：
{
    "clarity": {"score": 0-10, "comment": "具体评语", "suggestion": "具体改进建议"},
    "completeness": {...},
    "actionability": {...},
    "robustness": {...},
    "safety": {...},
    "overall_score": 0-10,
    "summary": "一句话总结"
}"""


def calculate_weighted_score(critique: dict, dimensions: list) -> float:
    """根据各维度权重计算综合评分"""
    total = 0.0
    total_weight = 0.0
    for dim in dimensions:
        key = dim["key"]
        if key in critique:
            total += critique[key]["score"] * dim["weight"]
            total_weight += dim["weight"]

    if total_weight > 0:
        return round(total / total_weight, 1)

    return 0.0


def critic_node(state: PromptForgeState) -> dict:
    """
    多维度审查提示词质量。

    审查流程：
    1. 获取当前提示词（优化后 > 草稿）
    2. 逐个维度评分
    3. 计算加权总分
    4. 输出结构化审查报告
    """
    # 优先取优化后的，没有则取草稿
    prompt_to_review = state.get("optimized_prompt") or state.get("draft_prompt", "")
    user_input = state.get("user_input", "")
    constraints = state.get("constraints", [])

    if not prompt_to_review:
        return {
            "critique": {"error": "没有可供审查的提示词"},
            "quality_score": 0.0,
        }

    # ============================================================
    # 实际实现: 调用 LLM 进行多维度审查
    # ============================================================
    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import SystemMessage, HumanMessage
    #
    # review_prompt = f"""
    # 请审查以下提示词：
    #
    # === 提示词 ===
    # {prompt_to_review}
    #
    # === 用户原始需求 ===
    # {user_input}
    #
    # === 约束条件 ===
    # {constraints}
    #
    # 请从以下维度审查并返回 JSON：
    # {format_dimensions(CRITIQUE_DIMENSIONS)}
    # """
    #
    # # Critic 可以用更强的推理模型
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # response = llm.invoke([
    #     SystemMessage(content=CRITIC_SYSTEM_PROMPT),
    #     HumanMessage(content=review_prompt)
    # ])
    #
    # critique = parse_json(response.content)
    # quality_score = critique.get("overall_score",
    #                   calculate_weighted_score(critique, CRITIQUE_DIMENSIONS))
    #
    # return {
    #     "critique": critique,
    #     "quality_score": quality_score,
    # }

    # ============================================================
    # 骨架实现 (Skeleton) — 模拟审查结果
    # ============================================================
    # 模拟审查逻辑: 基于提示词长度和关键词做简单判断
    has_examples = "示例" in prompt_to_review or "example" in prompt_to_review.lower()
    has_format = "json" in prompt_to_review.lower() or "格式" in prompt_to_review
    has_edge_cases = "边缘" in prompt_to_review or "特殊情况" in prompt_to_review
    has_safety = "风险" in prompt_to_review or "注意" in prompt_to_review
    length_ok = len(prompt_to_review) > 200

    critique = {
        "clarity": {
            "score": 8 if length_ok else 6,
            "comment": "角色设定清晰，任务描述明确" if length_ok else "缺少详细的角色设定",
            "suggestion": "可以增加更多输出示例让预期更明确"
        },
        "completeness": {
            "score": 8 if has_format else 6,
            "comment": "覆盖了主要约束条件" if has_format else "缺少明确的输出格式定义",
            "suggestion": "补充输出格式的完整 Schema"
        },
        "actionability": {
            "score": 9 if has_examples else 6,
            "comment": "有具体示例，模型可直接理解执行" if has_examples else "缺少 Few-shot 示例",
            "suggestion": "增加至少 3 个示例覆盖正/负/中性"
        },
        "robustness": {
            "score": 8 if has_edge_cases else 5,
            "comment": "边缘情况处理得当" if has_edge_cases else "未覆盖边缘情况",
            "suggestion": "增加反讽、混合情感、空输入等特殊情况的说明"
        },
        "safety": {
            "score": 9 if has_safety else 7,
            "comment": "无明显安全隐患" if has_safety else "建议增加输入长度限制",
            "suggestion": "添加 max_tokens 限制防止输出过长"
        },
    }

    quality_score = calculate_weighted_score(critique, CRITIQUE_DIMENSIONS)

    return {
        "critique": critique,
        "quality_score": quality_score,
    }
