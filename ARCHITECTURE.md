# PromptForge 架构设计文档

## 1. 设计动机

### 痛点
- 写高质量提示词需要反复试错，效率低下
- 不同模型对提示词格式敏感度不同，适配成本高
- 缺乏系统化的提示词测试和评估手段
- 团队内部优秀提示词经验无法沉淀复用

### 解决思路
利用 LangGraph 的状态图能力，将「需求分析 → 模板检索 → 提示词生成 → 质量审查 → 迭代优化 → 效果测试」这一完整链路建模为有向图，每个环节由独立的 LLM Agent 负责，节点间通过结构化状态传递上下文。

## 2. LangGraph 状态图设计

### 2.1 状态定义 (State Schema)

```python
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class PromptForgeState(TypedDict):
    # === 输入 ===
    user_input: str                    # 用户原始需求
    framework: str                     # 期望的提示词框架
    target_model: str                  # 目标模型 (gpt-4 / claude-3 / etc.)
    auto_optimize: bool                # 是否自动迭代优化

    # === 分析阶段 ===
    intent: str                        # 意图分类
    constraints: List[str]             # 约束条件列表
    output_format: str                 # 期望输出格式

    # === 检索阶段 ===
    similar_templates: List[dict]      # 检索到的相似模板

    # === 生成阶段 ===
    draft_prompt: str                  # 生成的第一版提示词
    generation_metadata: dict          # 生成元信息（框架、参数等）

    # === 审查阶段 ===
    critique: dict                     # 审查意见 {clarity, completeness, ...}
    quality_score: float               # 质量评分 0-10

    # === 优化阶段 ===
    optimized_prompt: str              # 优化后的提示词
    optimization_history: List[dict]   # 优化历史记录

    # === 测试阶段 ===
    test_results: List[dict]           # 测试结果
    final_prompt: str                  # 最终输出提示词

    # === 控制 ===
    messages: Annotated[list, add_messages]  # 消息历史（自动追加）
    iteration_count: int              # 优化迭代次数
    next_action: str                  # 下一步动作路由
```

### 2.2 图结构

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ Analyzer │  (分析用户需求)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ Retriever│  (检索相似模板)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
              ┌─────│Generator │  (生成初版提示词)
              │     └────┬─────┘
              │          │
              │     ┌────▼─────┐
              │     │  Critic  │  (多维度审查)
              │     └────┬─────┘
              │          │
              │     ┌────▼─────┐
              │     │  Router  │  (条件路由：通过/优化/人工)
              │     └─┬───┬───┬┘
              │       │   │   │
              │  <通过│   │   │优化>
              │       │   │   │
              │       │   │   └──────────┐
              │       │   │              │
              │       │   │ 人工审核      │
              │       │   ▼              │
              │       │ ┌──────────────┐ │
              │       │ │HumanReview   │ │
              │       │ │(human-in-loop)│ │
              │       │ └──────┬───────┘ │
              │       │        │         │
              │       │   <同意>         ▼
              │       │        │   ┌──────────┐
              │       │        │   │Optimizer │ (优化提示词)
              │       │        │   └────┬─────┘
              │       │        │        │
              │       │        │        │ iteration++
              │       │        │   ┌────▼─────┐
              │       │        │   │   Loop   │ (超过3次则走人工)
              │       │        │   └────┬─────┘
              │       │        │        │<3次
              │       │        │        └──────┐
              │       │        │               │
              │       │        └───────────────┘
              │       │                │
              │       └────────────────┘
              │                │
              │           ┌────▼─────┐
              └──────────►│  Tester  │ (测试验证)
                          └────┬─────┘
                               │
                          ┌────▼─────┐
                          │   END    │
                          └──────────┘
```

### 2.3 条件路由逻辑

```python
def route_after_critic(state: PromptForgeState) -> str:
    """根据审查结果决定下一步"""
    score = state["quality_score"]

    if score >= 8.0:
        return "tester"       # 质量合格，进入测试
    elif state["iteration_count"] >= 3:
        return "human_review" # 超过最大迭代次数，人工介入
    elif state["auto_optimize"]:
        return "optimizer"    # 自动优化后重新审查
    else:
        return "human_review" # 需要人工决策
```

## 3. 各节点职责

### 3.1 Analyzer (分析器)
- **输入**：`user_input`
- **任务**：
  - 意图分类（摘要 / 翻译 / 分类 / 生成 / 代码 / 推理 / 对话）
  - 提取约束条件（字数、格式、风格、角色扮演等）
  - 推断期望输出格式
  - 选择最适合的提示词框架
- **LLM 策略**：单次调用 + Few-shot 分类 prompt
- **输出**：`intent`, `constraints`, `output_format`

### 3.2 Retriever (检索器)
- **输入**：`intent` + `constraints` + `framework`
- **任务**：
  - 将意图+约束向量化
  - 在 pgvector 中检索 Top-K 相似提示词模板
  - 融合多个框架的模板（如果有）
- **工具**：pgvector 向量相似度搜索
- **输出**：`similar_templates`

### 3.3 Generator (生成器)
- **输入**：`user_input` + `constraints` + `output_format` + `similar_templates`
- **任务**：
  - 基于框架和模板生成初版提示词
  - 注入角色设定、格式约束、示例
  - 适配目标模型的格式要求
- **LLM 策略**：
  - gpt-4：系统的 system prompt + 清晰的 user prompt
  - claude：Human/Assistant 对话格式
- **输出**：`draft_prompt`

### 3.4 Critic (审查器)
- **输入**：`draft_prompt` + `user_input` + `constraints`
- **任务**：从以下维度打分和提意见
  - **清晰度** (Clarity)：指令是否明确无歧义
  - **完整性** (Completeness)：是否覆盖所有约束
  - **可执行性** (Actionability)：输出是否可解析
  - **鲁棒性** (Robustness)：边缘情况是否考虑
  - **安全性** (Safety)：是否存在注入风险
- **LLM 策略**：Chain-of-Thought 逐维评分
- **输出**：`critique`, `quality_score`

### 3.5 Optimizer (优化器)
- **输入**：`draft_prompt` + `critique`
- **任务**：
  - 根据审查意见逐条修改
  - 保留原有优点
  - 确保修改不引入新问题
- **LLM 策略**：ReAct 迭代修改（思考→行动→观察）
- **输出**：`optimized_prompt`（更新回 `draft_prompt`）

### 3.6 Tester (测试器)
- **输入**：`final_prompt` + `test_cases`
- **任务**：
  - 用预设测试用例运行提示词
  - 检查输出格式是否符合预期
  - 评估输出质量（与参考答案对比）
  - 生成测试报告
- **工具**：LLM 调用 + 正则/JSON Schema 校验
- **输出**：`test_results`

### 3.7 Human Review (人工审查)
- 使用 LangGraph 的 `interrupt` 机制暂停图执行
- 将当前状态推送到 Streamlit 前端供人工审核
- 人工可以：批准、修改后批准、拒绝并重新生成

## 4. 关键 LangGraph 能力运用

| LangGraph 能力 | 本项目中的运用 |
|----------------|---------------|
| `StateGraph` | 定义 7 个节点+条件路由的状态图 |
| `TypedDict State` | 强类型状态，IDE 有自动补全 |
| `add_messages` Reducer | 自动追加消息历史，不需要手动拼接 |
| `ConditionalEdges` | 根据质量评分动态路由到不同节点 |
| `interrupt` / `Command` | Human-in-the-loop 断点等待人工审批 |
| `MemorySaver` | 持久化检查点，支持断点续跑和时光旅行 |
| `ToolNode` | 封装搜索、检索、评估等外部工具 |
| `Send()` API | 并行生成多个提示词变体后对比 |
| `Subgraph` | 将 Analyzer+Generator 封装为可复用子图 |

## 5. 部署架构

```
┌──────────────────────────────────────────┐
│              Streamlit 前端               │
│   (输入需求 + 审核 + 结果展示)              │
└──────────────────┬───────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────┐
│           FastAPI 后端                     │
│  ┌─────────────────────────────────────┐ │
│  │         LangGraph 工作流              │ │
│  │  Analyzer→Generator→Critic→Optimizer │ │
│  └────────────────┬────────────────────┘ │
│                   │                       │
│  ┌────────────────▼────────────────────┐ │
│  │         LangChain LLM 抽象层         │ │
│  │    OpenAI │ Claude │ 文心一言 │ ...   │ │
│  └─────────────────────────────────────┘ │
└──────────────────┬───────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌───────┐  ┌──────────┐  ┌──────────┐
│Redis  │  │PostgreSQL│  │ RabbitMQ │
│缓存    │  │+ pgvector│  │ 任务队列  │
└───────┘  └──────────┘  └──────────┘
```

## 6. 核心设计决策

### 6.1 为什么选择 LangGraph 而不是直接写 Prompt Chain？
- **状态管理**：LangGraph 提供强类型状态 + 自动持久化，多个 Agent 之间可以安全共享上下文。
- **可控流程**：条件路由 + 循环 + 中断恢复，比 LangChain 的 `SequentialChain` 灵活得多。
- **可观测性**：每个节点执行自动打点，配合 LangSmith 可视化调试。

### 6.2 为什么要有 Retriever 节点？
- 从零生成不如参照优秀范例，Retriever 提供 Few-shot 的 "shot"。
- RAG 方式比纯 Prompt 能承载更多经验（上千个模板 vs 有限的上下文窗口）。

### 6.3 为什么 Critic 和 Optimizer 分离？
- 单一职责：审查和修改是不同的能力，分开后各自 prompt 更聚焦。
- Critic 可以用更强的推理模型（如 o1），Optimizer 用更快的模型降成本。

### 6.4 为什么迭代次数限制为 3？
- 经验表明 3 轮内质量曲线收敛，超过 3 轮提升微乎其微。
- 成本控制：每多一轮 = 多 2 次 LLM 调用。
- 超过后引入人工判断避免死循环。
