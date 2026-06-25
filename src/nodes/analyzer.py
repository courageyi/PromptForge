"""
Analyzer 节点 — 需求分析器

职责：
1. 解析用户自然语言需求，提取结构化信息
2. 意图分类：判断用户想完成什么任务
3. 约束提取：识别用户提出的限制条件
4. 框架推荐：根据意图推荐最佳提示词框架
"""

from ..state import PromptForgeState

# 提示词框架描述（实际项目中放配置文件或 DB）
FRAMEWORK_MAP = {
    "few-shot": "为模型提供若干示例，适合格式明确的分类/生成任务",
    "cot": "Chain-of-Thought，引导模型逐步推理，适合复杂推理任务",
    "react": "ReAct 模式，模型交替进行推理和行动，适合需要工具调用的任务",
    "tot": "Tree-of-Thought，多路径探索合并，适合需要创造力的任务",
    "emotion": "情绪化提示，为模型注入角色情感，适合写作和对话类任务",
}

INTENT_CATEGORIES = [
    "summarization",    # 摘要
    "translation",      # 翻译
    "classification",   # 分类
    "generation",       # 生成
    "coding",           # 编程
    "reasoning",        # 推理
    "chat",             # 对话
]

SYSTEM_PROMPT = """你是一个提示词需求分析专家。分析用户的描述，提取以下信息并以 JSON 格式返回：

{
    "intent": "分类: summarization|translation|classification|generation|coding|reasoning|chat",
    "constraints": ["约束1", "约束2", ...],
    "output_format": "期望的输出格式: json|markdown|text|code|table|csv",
    "recommended_framework": "推荐的提示词框架: few-shot|cot|react|tot",
    "analysis_reason": "分析理由简述"
}

注意：
- 约束包括：字数限制、输出格式、角色设定、风格要求、领域限制等
- 如果用户没有明确指定格式，根据任务类型推断
- 复杂推理任务优先推荐 cot，需要工具调用的推荐 react"""


def analyzer_node(state: PromptForgeState) -> dict:
    """
    分析用户需求，提取意图、约束和推荐框架。

    Args:
        state: 当前状态，至少包含 user_input

    Returns:
        更新后的状态字段: intent, constraints, output_format, framework
    """
    user_input = state["user_input"]

    # ============================================================
    # 实际实现: 调用 LLM 进行分析
    # ============================================================
    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import SystemMessage, HumanMessage
    #
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # response = llm.invoke([
    #     SystemMessage(content=SYSTEM_PROMPT),
    #     HumanMessage(content=f"用户需求：{user_input}")
    # ])
    # result = parse_json(response.content)  # 解析 JSON 输出
    #
    # return {
    #     "intent": result["intent"],
    #     "constraints": result["constraints"],
    #     "output_format": result["output_format"],
    #     "framework": state.get("framework") or result["recommended_framework"],
    # }

    # ============================================================
    # 骨架实现 (Skeleton)
    # ============================================================
    # 这里是示意逻辑，实际项目替换为上面的 LLM 调用

    # 简单关键词意图分类
    intent = "generation"
    constraints = []
    output_format = "text"

    if "情感" in user_input or "sentiment" in user_input:
        intent = "classification"
    if "翻译" in user_input or "translate" in user_input:
        intent = "translation"
    if "代码" in user_input or "code" in user_input:
        intent = "coding"
    if "摘要" in user_input or "总结" in user_input:
        intent = "summarization"

    if "JSON" in user_input or "json" in user_input:
        output_format = "json"
        constraints.append("输出JSON格式")
    if "markdown" in user_input.lower():
        output_format = "markdown"
        constraints.append("输出Markdown格式")

    if "200字" in user_input or "200个字" in user_input:
        constraints.append("限制200字内")
    if "正式" in user_input or "专业" in user_input:
        constraints.append("使用正式语气")

    framework = state.get("framework") or "few-shot"

    return {
        "intent": intent,
        "constraints": constraints,
        "output_format": output_format,
        "framework": framework,
    }
