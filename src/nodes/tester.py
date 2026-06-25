"""
Tester 节点 — 提示词效果测试器

职责：
1. 用预定义的测试用例运行提示词
2. 验证输出格式是否符合预期
3. 评估输出质量（与参考答案对比）
4. 生成测试报告

测试策略：
- 正向用例：正常输入应得到正确结果
- 负向用例：异常输入应得到合理处理
- 边界用例：极端值、空值、超长输入
"""

import json
from typing import List, Dict, Any
from ..state import PromptForgeState


# ============================================================
# 预设测试用例
# ============================================================

# 实际项目中，测试用例可以来自：
# 1. 用户提供的标注数据
# 2. 平台积累的测试集
# 3. LLM 自动生成的边界用例

DEFAULT_TEST_CASES = [
    {
        "input": "这个产品太棒了，强烈推荐！",
        "expected_sentiment": "positive",
        "category": "正向-明确好评"
    },
    {
        "input": "垃圾，用了三天就坏了，客服态度还特别差",
        "expected_sentiment": "negative",
        "category": "负向-明确差评"
    },
    {
        "input": "还行吧，没什么特别的感觉",
        "expected_sentiment": "neutral",
        "category": "中性-平淡评价"
    },
    {
        "input": "质量确实好，但是价格也太贵了吧",
        "expected_sentiment": "positive",
        "category": "混合情感-正向为主"
    },
    {
        "input": "",
        "expected_sentiment": "neutral",
        "category": "边界-空输入"
    },
    {
        "input": "这个面膜用完后，我的脸" + "非常好" * 50,
        "expected_sentiment": "positive",
        "category": "边界-超长重复输入"
    },
]


def validate_output_format(output: dict) -> tuple[bool, str]:
    """
    校验输出格式是否符合预期。

    Returns:
        (is_valid, error_message)
    """
    # 必须包含的字段
    required_fields = ["sentiment", "confidence"]
    for field in required_fields:
        if field not in output:
            return False, f"缺少必要字段: {field}"

    # sentiment 值校验
    if output["sentiment"] not in ("positive", "negative", "neutral"):
        return False, f"sentiment 值不合法: {output['sentiment']}"

    # confidence 范围和类型
    conf = output.get("confidence")
    if not isinstance(conf, (int, float)):
        return False, f"confidence 应为数字，实际为: {type(conf)}"
    if not (0.0 <= conf <= 1.0):
        return False, f"confidence 超出 [0,1] 范围: {conf}"

    return True, ""


def evaluate_result(expected: str, actual: dict) -> Dict[str, Any]:
    """
    评估测试结果与预期的匹配程度。

    评估维度：
    1. sentiment 是否匹配
    2. confidence 是否合理
    3. 格式是否合法
    """
    format_ok, format_error = validate_output_format(actual)

    sentiment_match = actual.get("sentiment") == expected if format_ok else False

    # 对于匹配的情况，置信度应该较高（>=0.6）
    # 对于不匹配的情况，检查是否是合理的歧义情况
    confidence = actual.get("confidence", 0) if format_ok else 0

    return {
        "sentiment_match": sentiment_match,
        "format_valid": format_ok,
        "format_error": format_error,
        "confidence": confidence,
        "actual_sentiment": actual.get("sentiment"),
    }


def tester_node(state: PromptForgeState) -> dict:
    """
    测试提示词的效果。

    流程：
    1. 获取最终提示词
    2. 获取测试用例（优先用户提供的）
    3. 逐条执行测试
    4. 收集结果并计算统计指标
    5. 生成测试报告

    注意：测试节点是工作流的最后一环，通过后即可输出最终提示词。
    """
    final_prompt = state.get("final_prompt") or \
                   state.get("optimized_prompt") or \
                   state.get("draft_prompt", "")

    test_cases = state.get("test_cases") or DEFAULT_TEST_CASES

    if not final_prompt:
        return {
            "test_results": [{"error": "没有可测试的提示词"}],
        }

    # ============================================================
    # 实际实现: 逐条运行测试
    # ============================================================
    # from langchain_openai import ChatOpenAI
    #
    # llm = ChatOpenAI(model=state["target_model"], temperature=0)
    # test_results = []
    #
    # for case in test_cases:
    #     # 填充提示词模板
    #     filled_prompt = final_prompt.replace("{content}", case["input"])
    #
    #     import time
    #     start = time.time()
    #
    #     try:
    #         response = llm.invoke(filled_prompt)
    #         latency_ms = int((time.time() - start) * 1000)
    #
    #         # 尝试解析 JSON 输出
    #         try:
    #             actual_output = json.loads(response.content)
    #         except json.JSONDecodeError:
    #             actual_output = {"raw": response.content, "parse_error": True}
    #
    #         evaluation = evaluate_result(case["expected_sentiment"], actual_output)
    #         evaluation["input"] = case["input"][:50]
    #         evaluation["category"] = case.get("category", "")
    #         evaluation["latency_ms"] = latency_ms
    #         test_results.append(evaluation)
    #
    #     except Exception as e:
    #         test_results.append({
    #             "input": case["input"][:50],
    #             "error": str(e),
    #             "category": case.get("category", ""),
    #         })
    #
    # # 汇总统计
    # total = len(test_results)
    # passed = sum(1 for r in test_results if r.get("sentiment_match"))
    # format_ok = sum(1 for r in test_results if r.get("format_valid"))
    # avg_latency = sum(r.get("latency_ms", 0) for r in test_results) / max(total, 1)
    #
    # return {
    #     "test_results": test_results,
    #     "final_prompt": final_prompt,
    #     "test_summary": {
    #         "total": total,
    #         "sentiment_accuracy": f"{passed}/{total} ({passed/total*100:.1f}%)" if total else "N/A",
    #         "format_valid_rate": f"{format_ok}/{total} ({format_ok/total*100:.1f}%)" if total else "N/A",
    #         "avg_latency_ms": int(avg_latency),
    #     }
    # }

    # ============================================================
    # 骨架实现 (Skeleton) — 模拟测试结果
    # ============================================================
    test_results: List[Dict[str, Any]] = []

    for case in test_cases[:6]:  # 只模拟前 6 条
        # 简单的模拟逻辑：正向词 → positive，负向词 → negative
        inp = case["input"]
        if not inp or inp.strip() == "":
            actual = {"sentiment": "neutral", "confidence": 0.0}
        elif "差" in inp or "垃圾" in inp or "烂" in inp:
            actual = {"sentiment": "negative", "confidence": 0.92}
        elif "好" in inp or "棒" in inp or "推荐" in inp:
            actual = {"sentiment": "positive", "confidence": 0.95}
        elif "但是" in inp or "不过" in inp:
            actual = {"sentiment": "positive", "confidence": 0.65}
        else:
            actual = {"sentiment": "neutral", "confidence": 0.75}

        evaluation = evaluate_result(case["expected_sentiment"], actual)
        evaluation["input"] = inp[:50]
        evaluation["category"] = case.get("category", "")
        evaluation["latency_ms"] = 800 + hash(inp) % 500  # 模拟延迟
        test_results.append(evaluation)

    total = len(test_results)
    passed = sum(1 for r in test_results if r.get("sentiment_match"))
    format_ok = sum(1 for r in test_results if r.get("format_valid"))
    avg_latency = sum(r.get("latency_ms", 0) for r in test_results) / max(total, 1)

    return {
        "test_results": test_results,
        "final_prompt": final_prompt,
        # 额外的测试汇总方便前端展示
        # 注意: test_summary 不是状态字段，仅作测试节点输出
    }
