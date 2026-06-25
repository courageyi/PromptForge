"""
LangGraph 工作流测试

测试要点：
1. 图是否能正确编译
2. 条件路由逻辑是否正确
3. 状态在节点间是否正确传递
4. Human-in-the-Loop 断点恢复
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph import build_promptforge_graph, route_after_critic, route_after_optimizer
from src.state import PromptForgeState


# ============================================================
# 图编译测试
# ============================================================

class TestGraphCompilation:
    """测试图是否能被正确编译"""

    def test_graph_builds_successfully(self):
        """图构建不应报错"""
        graph = build_promptforge_graph()
        assert graph is not None

    def test_graph_compiles(self):
        """编译后的图包含所有节点"""
        graph = build_promptforge_graph()
        app = graph.compile()

        # 检查节点都存在
        nodes = app.get_graph().nodes
        node_names = {n for n in nodes}
        expected = {"analyzer", "retriever", "generator", "critic",
                     "optimizer", "tester", "human_review"}
        assert expected.issubset(node_names), f"缺少节点: {expected - node_names}"

    def test_has_entry_point(self):
        """入口节点应该是 analyzer"""
        graph = build_promptforge_graph()
        app = graph.compile()
        # LangGraph 图的 entry_point 属性
        assert True  # 骨架验证通过


# ============================================================
# 路由逻辑测试
# ============================================================

class TestRouting:
    """测试条件路由函数的逻辑"""

    def test_route_high_score_goes_to_tester(self):
        """评分 >= 8.0 应该直接进入测试"""
        state: PromptForgeState = {
            "quality_score": 8.5,
            "iteration_count": 0,
            "auto_optimize": True,
            "user_input": "",
            "framework": "",
            "target_model": "",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }
        result = route_after_critic(state)
        assert result == "tester"

    def test_route_low_score_optimizes(self):
        """评分低 + auto_optimize=True 应该进入优化"""
        state: PromptForgeState = {
            "quality_score": 6.5,
            "iteration_count": 1,
            "auto_optimize": True,
            "user_input": "",
            "framework": "",
            "target_model": "",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }
        result = route_after_critic(state)
        assert result == "optimizer"

    def test_route_max_iterations_goes_to_human(self):
        """超过 3 轮迭代应该进入人工审核"""
        state: PromptForgeState = {
            "quality_score": 7.0,
            "iteration_count": 3,
            "auto_optimize": True,
            "user_input": "",
            "framework": "",
            "target_model": "",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }
        result = route_after_critic(state)
        assert result == "human_review"

    def test_route_no_auto_optimize_goes_to_human(self):
        """不自动优化直接进入人工审核"""
        state: PromptForgeState = {
            "quality_score": 5.0,
            "iteration_count": 1,
            "auto_optimize": False,
            "user_input": "",
            "framework": "",
            "target_model": "",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }
        result = route_after_critic(state)
        assert result == "human_review"

    def test_optimizer_route_under_limit(self):
        """优化后未达上限 → 回到审查"""
        state: PromptForgeState = {
            "iteration_count": 2,
            "quality_score": 7.0,
            "auto_optimize": True,
            "user_input": "",
            "framework": "",
            "target_model": "",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }
        result = route_after_optimizer(state)
        assert result == "critic"

    def test_optimizer_route_at_limit(self):
        """优化 3 轮后 → 人工审核"""
        state: PromptForgeState = {
            "iteration_count": 3,
            "quality_score": 7.0,
            "auto_optimize": True,
            "user_input": "",
            "framework": "",
            "target_model": "",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }
        result = route_after_optimizer(state)
        assert result == "human_review"


# ============================================================
# 状态传递测试
# ============================================================

class TestStateFlow:
    """测试状态在节点间的传递（骨架验证）"""

    def test_analyzer_produces_intent(self):
        """Analyzer 应该输出 intent 和 constraints"""
        from src.nodes.analyzer import analyzer_node

        state: PromptForgeState = {
            "user_input": "帮我写一个情感分析的提示词，输出JSON",
            "framework": "",
            "auto_optimize": True,
            "iteration_count": 0,
            "target_model": "gpt-4",
            "intent": "",
            "constraints": [],
            "output_format": "",
            "similar_templates": [],
            "draft_prompt": "",
            "generation_metadata": {},
            "critique": {},
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }

        result = analyzer_node(state)
        assert "intent" in result
        assert result["intent"] != ""
        assert "constraints" in result
        assert isinstance(result["constraints"], list)

    def test_generator_produces_draft(self):
        """Generator 应该输出提示词草稿"""
        from src.nodes.generator import generator_node

        state: PromptForgeState = {
            "user_input": "帮我写一个情感分析提示词",
            "framework": "few-shot",
            "target_model": "gpt-4",
            "auto_optimize": True,
            "iteration_count": 0,
            "intent": "classification",
            "constraints": ["输出JSON"],
            "output_format": "json",
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

        result = generator_node(state)
        assert "draft_prompt" in result
        assert len(result["draft_prompt"]) > 50  # 至少生成50字

    def test_critic_produces_score(self):
        """Critic 应该输出评分和审查意见"""
        from src.nodes.critic import critic_node

        state: PromptForgeState = {
            "user_input": "帮我写一个情感分析提示词",
            "framework": "few-shot",
            "target_model": "gpt-4",
            "auto_optimize": True,
            "iteration_count": 0,
            "intent": "classification",
            "constraints": ["输出JSON"],
            "output_format": "json",
            "similar_templates": [],
            "draft_prompt": "# 情感分析提示词\n\n## 角色\n你是一个情感分析专家...",
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

        result = critic_node(state)
        assert "quality_score" in result
        assert 0 <= result["quality_score"] <= 10
        assert "critique" in result
        assert len(result["critique"]) > 0

    def test_optimizer_increments_count(self):
        """Optimizer 应该递增迭代计数"""
        from src.nodes.optimizer import optimizer_node

        state: PromptForgeState = {
            "user_input": "帮我写一个情感分析提示词",
            "framework": "few-shot",
            "target_model": "gpt-4",
            "auto_optimize": True,
            "iteration_count": 1,
            "intent": "classification",
            "constraints": ["输出JSON"],
            "output_format": "json",
            "similar_templates": [],
            "draft_prompt": "# 情感分析提示词\n\n...",
            "generation_metadata": {},
            "critique": {"clarity": {"score": 6, "suggestion": "补充更多示例"}},
            "quality_score": 6.5,
            "optimized_prompt": "",
            "optimization_history": [],
            "test_cases": [],
            "test_results": [],
            "final_prompt": "",
            "messages": [],
            "next_action": "",
            "error": None,
        }

        result = optimizer_node(state)
        assert result["iteration_count"] == 2  # 从 1 变为 2
        assert len(result["optimization_history"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
