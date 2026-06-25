# PromptForge — 基于 LangGraph 的智能提示词工程平台

## 一句话概述

> 使用 LangGraph 多智能体协作技术，将用户的模糊需求自动转化为高质量、可测试的结构化提示词。

## 项目亮点

- 🧠 **多智能体流水线**：分析 → 生成 → 审查 → 优化 → 测试，5 个 Agent 协作产出最优提示词
- 🔀 **条件路由**：根据提示词质量评分自动决定重试、人工介入或直接输出
- 🛠️ **工具增强(Tool Augmentation)**：集成搜索引擎、代码执行器、向量数据库检索
- 👤 **Human-in-the-Loop**：关键节点支持人工审核，LangGraph Checkpoint 实现断点恢复
- 📊 **A/B 测试框架**：自动生成多组提示词变体并量化对比效果
- 🎯 **多框架支持**：Few-shot、Chain-of-Thought、ReAct、Tree-of-Thought 等多种提示词框架

## 技术栈

| 层 | 技术 | 用途 |
|-----|------|------|
| 编排引擎 | **LangGraph** | 有状态多智能体工作流 |
| LLM 抽象 | **LangChain** | 统一调用 GPT-4/Claude/文心一言等模型 |
| API 服务 | **FastAPI** | 提供 RESTful API |
| 前端 | **Streamlit** | 快速搭建交互界面 |
| 向量存储 | **PostgreSQL + pgvector** | 优秀提示词模板检索 |
| 可观测性 | **LangSmith** | 链路追踪 + 性能监控 |
| 缓存 | **Redis** | 热点模板缓存 + 任务队列 |

## 项目结构

```
PromptForge/
├── src/
│   ├── graph.py              # LangGraph 工作流定义（核心）
│   ├── state.py              # 状态类型定义
│   ├── nodes/
│   │   ├── analyzer.py       # 需求分析节点
│   │   ├── generator.py      # 提示词生成节点
│   │   ├── critic.py         # 质量审查节点
│   │   ├── optimizer.py      # 迭代优化节点
│   │   └── tester.py         # 自动测试节点
│   ├── tools/
│   │   ├── search.py         # 联网搜索工具
│   │   ├── retriever.py      # 向量检索工具
│   │   └── evaluator.py      # 效果评估工具
│   ├── utils/
│   │   ├── template.py       # 提示词模板库
│   │   ├── metrics.py        # 评分指标计算
│   │   └── db.py             # 数据库操作
│   └── server.py             # FastAPI 入口
├── frontend/
│   └── app.py                # Streamlit 界面
├── tests/
│   └── test_graph.py         # 工作流单元测试
├── notebooks/
│   └── demo.ipynb            # Jupyter 演示
├── requirements.txt
├── docker-compose.yml
├── ARCHITECTURE.md           # 架构设计文档
└── INTERVIEW_QA.md           # 面试问答
```

## 核心工作流

```
用户输入粗略需求
      │
      ▼
┌─────────────┐
│  Analyzer   │  ← 提取意图、约束、输出格式要求
│  (分析器)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retriever  │  ← 从向量库检索相似的高质量提示词模板
│  (检索器)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐    质量不够
│  Generator  │────────┐
│  (生成器)    │        │
└──────┬──────┘        ▼
       │         ┌─────────────┐
       ▼         │  Optimizer  │
┌─────────────┐  │  (优化器)    │
│   Critic    │  └──────┬──────┘
│  (审查器)    │◄────────┘
└──────┬──────┘
       │ 质量达标
       ▼
┌─────────────┐
│   Tester    │  ← 用测试用例验证提示词效果
│  (测试器)    │
└──────┬──────┘
       │
       ▼
   最终输出：
   - 优化后的提示词
   - 测试报告
   - 使用建议
```

## 快速开始

```bash
# 1. 启动依赖服务
docker-compose up -d

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 API 服务
python -m src.server

# 4. 启动前端（另一个终端）
streamlit run frontend/app.py
```

## 演示截图

```python
# 调用示例
from src.graph import PromptForgeGraph

graph = PromptForgeGraph()

result = graph.invoke({
    "user_input": "帮我写一个分析用户评论情感倾向的提示词，输出 JSON 格式",
    "framework": "few-shot",
    "target_model": "gpt-4",
    "auto_optimize": True
})

print(result["final_prompt"])
print(f"质量评分: {result['quality_score']}/10")
```
