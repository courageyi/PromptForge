"""
PromptForge Streamlit 前端

提供简易的 Web 交互界面，用于：
1. 输入需求并启动优化流程
2. 流式观察每个 Agent 的输出
3. 在人工审核节点做出决策
"""

import streamlit as st
import requests
import json

st.set_page_config(
    page_title="PromptForge - 智能提示词优化",
    page_icon="🔨",
    layout="wide",
)

# ============================================================
# 配置
# ============================================================

API_BASE = "http://localhost:8000"

# ============================================================
# 侧边栏 — 输入区
# ============================================================

with st.sidebar:
    st.title("🔨 PromptForge")
    st.markdown("> 基于 LangGraph 的智能提示词优化平台")
    st.divider()

    st.subheader("📝 输入需求")
    user_input = st.text_area(
        "描述你想要什么提示词",
        placeholder="例如：帮我写一个分析用户评论情感倾向的提示词，输出JSON格式，包含置信度",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        framework = st.selectbox(
            "提示词框架",
            ["few-shot", "cot (思维链)", "react (推理+行动)", "tot (思维树)", "custom"],
        )
    with col2:
        target_model = st.selectbox(
            "目标模型",
            ["gpt-4", "gpt-4o", "claude-opus-4-8", "claude-sonnet-4-6", "deepseek-v3"],
        )

    auto_optimize = st.toggle("自动优化（跳过人工审核）", value=True)

    st.divider()

    if st.button("⚡ 开始优化", type="primary", use_container_width=True):
        st.session_state.run_clicked = True
        st.session_state.user_input = user_input
    else:
        if "run_clicked" not in st.session_state:
            st.session_state.run_clicked = False


# ============================================================
# 主区域 — 结果展示
# ============================================================

st.title("🔨 PromptForge")
st.caption("Analyzr → Generator → Critic → Optimizer → Tester")

# 流程可视化
st.subheader("📊 工作流进度")

placeholder = st.empty()

if st.session_state.get("run_clicked"):
    with placeholder.container():
        # 模拟展示各节点的进度
        tabs = st.tabs(["📋 分析", "🔍 检索", "🎨 生成", "🔎 审查", "⚡ 优化", "🧪 测试"])

        with tabs[0]:
            st.info("正在分析需求...")
            st.json({
                "intent": "classification",
                "constraints": ["输出JSON", "包含confidence字段"],
                "output_format": "json",
                "framework": "few-shot",
            })

        with tabs[1]:
            st.info("正在检索相似模板...")
            st.dataframe([
                {"模板来源": "内部最佳实践", "相似度": "92%", "框架": "few-shot"},
                {"模板来源": "社区贡献", "相似度": "85%", "框架": "basic"},
            ])

        with tabs[2]:
            st.info("正在生成提示词...")
            st.markdown("""
            # 情感分析提示词
            ## 角色
            你是一位专业的用户评论情感分析专家...
            *(完整内容见 API 返回)*
            """)

        with tabs[3]:
            st.info("正在审查质量...")
            st.json({
                "clarity": {"score": 8, "status": "✅"},
                "completeness": {"score": 8, "status": "✅"},
                "actionability": {"score": 9, "status": "✅"},
                "robustness": {"score": 8, "status": "✅"},
                "safety": {"score": 9, "status": "✅"},
                "overall": 8.5,
            })

        with tabs[4]:
            st.info("优化完成（1 轮迭代）")
            st.markdown("已补充：边缘情况处理、输出字段说明表")

        with tabs[5]:
            st.success("测试完成！")
            st.dataframe([
                {"用例": "明确好评", "预期": "positive", "实际": "positive", "通过": "✅", "延迟": "850ms"},
                {"用例": "明确差评", "预期": "negative", "实际": "negative", "通过": "✅", "延迟": "920ms"},
                {"用例": "中性评价", "预期": "neutral", "实际": "neutral", "通过": "✅", "延迟": "780ms"},
                {"用例": "混合情感", "预期": "positive", "实际": "positive", "通过": "✅", "延迟": "1100ms"},
                {"用例": "空输入", "预期": "neutral", "实际": "neutral", "通过": "✅", "延迟": "350ms"},
                {"用例": "超长输入", "预期": "positive", "实际": "negative", "通过": "❌", "延迟": "2300ms"},
            ])

    # ============================================================
    # Human-in-the-Loop 审核区
    # ============================================================
    st.divider()
    st.subheader("👤 人工审核区")

    review_col1, review_col2 = st.columns(2)

    with review_col1:
        st.markdown("**当前提示词**")
        st.text_area("提示词内容", value="# 情感分析提示词\n\n## 角色\n你是一位...", height=200, disabled=True)

    with review_col2:
        st.markdown("**审查报告**")
        st.json({"quality_score": 7.2, "main_issues": ["缺少边缘情况处理", "输出格式 Schema 不完整"]})

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        st.button("✅ 批准（直接通过）", use_container_width=True)
    with btn_col2:
        st.button("✏️ 修改后批准", use_container_width=True)
    with btn_col3:
        st.button("🔄 重新生成", use_container_width=True)

else:
    st.markdown("""
    ### 👈 从左边的侧边栏开始

    1. 输入你的需求（自然语言描述即可）
    2. 选择提示词框架和目标模型
    3. 点击「开始优化」
    4. 观察 5 个 AI Agent 如何协作生成高质量提示词
    5. 在关键节点进行人工审核

    ---

    ### 🧠 工作原理

    PromptForge 使用 **LangGraph** 将多个 LLM Agent 编排为流水线：

    - **Analyzer**：理解你的需求，提取约束和意图
    - **Retriever**：从向量库检索相似高质量模板
    - **Generator**：基于分析结果生成初版提示词
    - **Critic**：从 5 个维度审查质量
    - **Optimizer**：根据审查意见迭代优化
    - **Tester**：用测试用例验证效果

    如果质量不达标，系统会自动进入优化循环（最多 3 轮），
    超过后引入人工审核避免无限循环。
    """)
