"""
Generator 节点 — 提示词生成器

职责：
1. 根据分析结果 + 检索到的模板生成初版提示词
2. 适配目标模型的提示词格式要求
3. 注入 Few-shot 示例和约束条件

关键设计：
- 使用结构化输出（LangChain with_structured_output）确保格式可控
- 不同目标模型使用不同的 System/User prompt 策略
"""

from ..state import PromptForgeState

# ============================================================
# 模型格式适配策略
# ============================================================
# 不同模型对提示词格式的偏好不同：
#
# GPT-4/GPT-4o:
#   最佳实践: 清晰的 System Prompt + 明确的 User Prompt
#   分隔符: 用 ### 或 XML 标签分隔各部分
#
# Claude (Opus/Sonnet):
#   最佳实践: Human/Assistant 对话格式
#   特点: 更重视角色设定，偏好自然语言指令
#
# DeepSeek:
#   最佳实践: 类似 GPT，但 System prompt 权重较低
#   特点: 对格式要求更严格，需要明确输出结构

MODEL_FORMATTERS = {
    "gpt-4": {
        "role_tag": "system/user/assistant",
        "separator": "###",
        "emphasis": "markdown_bold",
        "output_instruction_style": "imperative"
    },
    "gpt-4o": {
        "role_tag": "system/user/assistant",
        "separator": "###",
        "emphasis": "markdown_bold",
        "output_instruction_style": "imperative"
    },
    "claude-opus-4-8": {
        "role_tag": "Human/Assistant",
        "separator": "---",
        "emphasis": "natural_language",
        "output_instruction_style": "conversational"
    },
    "claude-sonnet-4-6": {
        "role_tag": "Human/Assistant",
        "separator": "---",
        "emphasis": "natural_language",
        "output_instruction_style": "conversational"
    },
    "deepseek-v3": {
        "role_tag": "system/user/assistant",
        "separator": "###",
        "emphasis": "markdown_bold",
        "output_instruction_style": "strict"
    },
}

GENERATOR_SYSTEM_PROMPT = """你是一位世界级的提示词工程师(prompt engineer)。

你的任务是根据用户需求，生成高质量的结构化提示词。生成的提示词必须：
1. **清晰**：指令明确，没有歧义
2. **完整**：覆盖所有用户约束
3. **可执行**：输出格式可被解析
4. **鲁棒**：考虑边缘情况
5. **适配**：针对目标模型的格式特点优化

当你生成提示词时：
- 先分析需求的核心目标
- 然后设计提示词结构（角色、指令、格式、示例）
- 参考提供的模板，但根据具体需求调整
- 为约束条件逐一设计处理逻辑"""


def generator_node(state: PromptForgeState) -> dict:
    """
    生成初版提示词。

    流程：
    1. 读取用户需求、约束、框架、模板
    2. 选择目标模型的格式策略
    3. 调用 LLM 生成提示词
    4. 返回生成结果

    Args:
        state: 包含分析结果和模板的完整状态

    Returns:
        更新 draft_prompt 和 generation_metadata
    """
    user_input = state["user_input"]
    intent = state.get("intent", "")
    constraints = state.get("constraints", [])
    output_format = state.get("output_format", "text")
    framework = state.get("framework", "few-shot")
    target_model = state.get("target_model", "gpt-4")
    templates = state.get("similar_templates", [])

    # ============================================================
    # 实际实现: 调用 LLM 生成提示词
    # ============================================================
    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import SystemMessage, HumanMessage
    #
    # formatter = MODEL_FORMATTERS.get(target_model, MODEL_FORMATTERS["gpt-4"])
    #
    # # 构建生成 prompt
    # generation_prompt = f"""
    # 用户需求：{user_input}
    #
    # 参考信息：
    # - 意图类型：{intent}
    # - 约束条件：{constraints}
    # - 期望输出格式：{output_format}
    # - 提示词框架：{framework}
    # - 目标模型：{target_model}
    # - 模型格式偏好：{formatter}
    #
    # 参考模板（供参考格式和风格）：
    # {format_templates(templates)}
    #
    # 请生成最终提示词。只输出提示词本身，不要额外解释。
    # """
    #
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    # response = llm.invoke([
    #     SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
    #     HumanMessage(content=generation_prompt)
    # ])
    #
    # draft = response.content
    #
    # return {
    #     "draft_prompt": draft,
    #     "generation_metadata": {
    #         "framework": framework,
    #         "template_ids": [t.get("source") for t in templates],
    #         "target_model": target_model,
    #         "format_style": formatter["output_instruction_style"],
    #         "token_usage": response.response_metadata.get("token_usage", {}),
    #     }
    # }

    # ============================================================
    # 骨架实现 (Skeleton) — 展示生成的提示词结构
    # ============================================================
    constraints_text = "\n".join([f"- {c}" for c in constraints]) if constraints else "- 无特殊约束"

    template_ref = ""
    if templates:
        template_ref = f"\n## 参考风格\n以下为优秀提示词参考（已内化）：\n{templates[0]['template_text'][:200]}..."

    draft = f"""# 情感分析提示词

## 角色
你是一位专业的用户评论情感分析专家，擅长从文本中精准识别用户的情绪和态度。

## 任务
分析以下用户评论的情感倾向，输出结构化的分析结果。

## 约束
{constraints_text}

## 输出格式
```json
{{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0-1.0,
  "keywords": ["关键情感词1", "关键情感词2"],
  "reasoning": "简短的判断理由"
}}
```

## 判断标准
- **positive**: 包含明显的正面词汇或积极的语气，如"很好"、"推荐"、"满意"
- **negative**: 包含明显的负面词汇或批评的语气，如"很差"、"失望"、"不推荐"
- **neutral**: 客观描述、询问类、或无明确情感倾向
- **confidence**: 根据情感词的强度、修饰词（"比较"、"非常"）、句式综合判断

## 边缘情况处理
- 混合情感（"质量好但价格贵"）：以主要情感为准，在 reasoning 中注明次要情感
- 反讽/讽刺：标注为对应表面含义，confidence 降低 0.1-0.2
- 无文本或纯表情：sentiment="neutral", confidence=0.3
- 非中文内容：检查是否为已知语言，是则正常分析，否则返回 neutral

## 示例
| 输入 | sentiment | confidence |
|------|-----------|------------|
| "这个产品太棒了，强烈推荐！" | positive | 0.95 |
| "一般般吧，没什么特别的" | neutral | 0.80 |
| "垃圾，用了三天就坏了" | negative | 0.90 |
| "性价比高但是物流太慢" | positive | 0.65 |

## 待分析内容
{{content}}
"""

    return {
        "draft_prompt": draft,
        "generation_metadata": {
            "framework": framework,
            "template_refs": [t.get("source") for t in templates[:2]],
            "target_model": target_model,
            "format_style": MODEL_FORMATTERS.get(target_model, {}).get("output_instruction_style", "default"),
        }
    }
