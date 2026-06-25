"""
Optimizer 节点 — 提示词迭代优化器

职责：
1. 根据 Critic 的审查意见逐条修改提示词
2. 保留原有优点，只修改有问题部分
3. 确保优化不引入新问题
4. 记录每次迭代的变更历史

优化策略：
- 先处理评分最低的维度（最需要改进的地方）
- 小步修改，每次聚焦 1-2 个维度
- 保持原有结构和风格，避免"重写"导致新的缺陷
"""

from ..state import PromptForgeState


OPTIMIZER_SYSTEM_PROMPT = """你是一个提示词优化专家。你会根据审查意见优化提示词，但不做不必要的修改。

优化原则：
1. **保留优点**：不要改动审查中评分高的部分
2. **精准修改**：只改动被指出问题的部分
3. **不引入新问题**：修改后自查是否影响了其他维度
4. **增量改进**：每次优化应看到评分提升

如果审查意见过于笼统，先提炼出具体可操作的修改步骤再执行。"""


def format_critique_for_optimization(critique: dict) -> str:
    """
    将审查报告格式化为优化指令。

    按评分从低到高排列需要改进的维度，
    优先处理最严重的问题。
    """
    if not critique:
        return "无审查意见，无需优化"

    # 按得分排序（从低到高）
    dims = ["clarity", "completeness", "actionability", "robustness", "safety"]
    issues = []

    for dim in dims:
        if dim in critique and isinstance(critique[dim], dict):
            score = critique[dim].get("score", 10)
            suggestion = critique[dim].get("suggestion", "")
            if score < 8 and suggestion:  # 只关注有明显问题的维度
                issues.append((dim, score, suggestion))

    issues.sort(key=lambda x: x[1])  # 最低分排前面

    lines = []
    for i, (dim, score, suggestion) in enumerate(issues, 1):
        lines.append(f"{i}. [{dim}] (当前评分: {score}/10) — {suggestion}")

    return "\n".join(lines) if lines else "所有维度评分良好，重点优化可执行性"


def optimizer_node(state: PromptForgeState) -> dict:
    """
    根据审查意见优化提示词。

    流程：
    1. 读取当前提示词 + 审查报告
    2. 提取需要改进的维度
    3. 调用 LLM 执行定向修改
    4. 记录变更历史
    5. 增加迭代计数

    Args:
        state: 包含 draft_prompt, critique 的当前状态

    Returns:
        更新 optimized_prompt, optimization_history, iteration_count
    """
    current_prompt = state.get("optimized_prompt") or state.get("draft_prompt", "")
    critique = state.get("critique", {})
    iteration = state.get("iteration_count", 0)

    if not current_prompt:
        return {"error": "没有可供优化的提示词"}

    # ============================================================
    # 实际实现: 调用 LLM 进行优化
    # ============================================================
    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import SystemMessage, HumanMessage
    #
    # optimization_instructions = format_critique_for_optimization(critique)
    #
    # optimize_prompt = f"""
    # === 当前提示词 ===
    # {current_prompt}
    #
    # === 审查意见（按优先级排列） ===
    # {optimization_instructions}
    #
    # 请对提示词进行优化。只输出优化后的完整提示词，不要包含任何解释。
    # 保持原有结构（## 标题 / 列表 / 代码块），只修改有问题的地方。
    # """
    #
    # # Optimizer 用更快的模型降低成本
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    # response = llm.invoke([
    #     SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT),
    #     HumanMessage(content=optimize_prompt)
    # ])
    #
    # optimized = response.content
    #
    # # 记录历史
    # history_entry = {
    #     "round": iteration + 1,
    #     "before_score": state.get("quality_score", 0),
    #     "changes_summary": optimization_instructions[:200],
    #     "prompt_before_length": len(current_prompt),
    #     "prompt_after_length": len(optimized),
    # }
    #
    # history = state.get("optimization_history", [])
    # history.append(history_entry)
    #
    # return {
    #     "optimized_prompt": optimized,
    #     "optimization_history": history,
    #     "iteration_count": iteration + 1,
    # }

    # ============================================================
    # 骨架实现 (Skeleton) — 模拟优化
    # ============================================================
    instructions = format_critique_for_optimization(critique)

    # 模拟优化过程：在提示词末尾追加改进说明
    improved_sections = []

    # 根据审查意见补充内容
    if "robustness" in str(critique).lower() or "边缘" in str(critique):
        improved_sections.append("""
## 补充边缘情况处理
- 输入为空字符串或纯空格：返回 {{"sentiment": "neutral", "confidence": 0.0, "error": "empty_input"}}
- 输入包含非文本字符（emoji等）：提取文字部分分析，表情作为辅助参考
- 超长文本（>1000字）：取开头300字+结尾200字进行分析，注明"基于部分内容""")

    if "completeness" in str(critique).lower() or "完整" in str(critique):
        improved_sections.append("""
## 输出字段说明
| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| sentiment | string | 是 | positive / negative / neutral |
| confidence | float | 是 | 0.0-1.0，保留两位小数 |
| keywords | array | 是 | 关键情感词，最多5个 |
| reasoning | string | 是 | 判断理由，不超过100字 |
| error | string | 否 | 处理异常时的错误说明 |""")

    optimized = current_prompt
    if improved_sections:
        optimized = current_prompt.rstrip() + "\n\n" + "\n\n".join(improved_sections)

    # 记录历史
    history = list(state.get("optimization_history", []))  # 创建副本
    history.append({
        "round": iteration + 1,
        "before_score": state.get("quality_score", 0),
        "changes_summary": instructions[:200],
        "prompt_before_length": len(current_prompt),
        "prompt_after_length": len(optimized),
    })

    return {
        "optimized_prompt": optimized,
        "optimization_history": history,
        "iteration_count": iteration + 1,
    }
