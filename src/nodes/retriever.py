"""
Retriever 节点 — 模板检索器

职责：
1. 将分析结果（意图 + 约束）向量化
2. 在 pgvector 中检索相似的高质量提示词模板
3. 返回 Top-K 模板作为生成器的 Few-shot 示例

LangGraph 中的工具调用方式：
- 使用 ToolNode 封装工具
- 在节点内使用 tool.invoke() 同步调用
- 或使用 create_react_agent 创建 ReAct 循环
"""

from typing import List, Dict, Any
from ..state import PromptForgeState


def build_retrieval_query(intent: str, constraints: list, framework: str) -> str:
    """
    构造检索查询字符串。

    将结构化信息拼接为自然语言查询，提升向量检索召回率。
    """
    query_parts = [f"意图: {intent}"]

    if framework:
        query_parts.append(f"框架: {framework}")

    if constraints:
        query_parts.append(f"约束: {', '.join(constraints)}")

    return " | ".join(query_parts)


def retriever_node(state: PromptForgeState) -> dict:
    """
    从向量数据库检索相似提示词模板。

    检索策略：
    1. 语义检索：用 embedding 向量相似度匹配
    2. 关键词过滤：约束条件作为过滤条件
    3. 融合排序：结合相似度 + 好评率 + 使用频次

    返回 Top-5 模板，每个模板包含：
    - template_text: 提示词原文
    - score: 综合评分
    - metadata: 来源、使用的模型、效果评级等
    """
    intent = state.get("intent", "")
    constraints = state.get("constraints", [])
    framework = state.get("framework", "")

    # ============================================================
    # 实际实现: 向量检索
    # ============================================================
    # from langchain_openai import OpenAIEmbeddings
    # from langchain_postgres import PGVector
    #
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # vectorstore = PGVector(
    #     embeddings=embeddings,
    #     collection_name="prompt_templates",
    #     connection_string=os.getenv("DATABASE_URL"),
    # )
    #
    # query = build_retrieval_query(intent, constraints, framework)
    #
    # # 第一步: 语义检索 Top-10
    # candidates = vectorstore.similarity_search_with_score(query, k=10)
    #
    # # 第二步: 重排序（Cohere Rerank / BGE Reranker）
    # from langchain.retrievers import ContextualCompressionRetriever
    # from langchain_cohere import CohereRerank
    #
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    # compressor = CohereRerank(top_n=5)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_retriever=retriever,
    #     base_compressor=compressor,
    # )
    #
    # docs = compression_retriever.invoke(query)
    # similar_templates = [
    #     {
    #         "template_text": doc.page_content,
    #         "score": doc.metadata.get("relevance_score", 0),
    #         "source": doc.metadata.get("source", "community"),
    #         "avg_rating": doc.metadata.get("avg_rating", 0),
    #         "usage_count": doc.metadata.get("usage_count", 0),
    #     }
    #     for doc in docs
    # ]
    #
    # return {"similar_templates": similar_templates}

    # ============================================================
    # 骨架实现 (Skeleton)
    # ============================================================
    # 模拟检索结果，展示数据结构
    mock_templates: List[Dict[str, Any]] = [
        {
            "template_text": (
                "你是一个专业的{domain}分析师。请分析以下{content_type}的情感倾向。\n\n"
                "要求：\n"
                "1. 输出JSON格式：{{\"sentiment\": \"positive|negative|neutral\", \"confidence\": 0.0-1.0}}\n"
                "2. 置信度基于语气词、修辞手法、上下文综合判断\n"
                "3. 如无法判断，标记为 neutral 并说明原因\n\n"
                "示例：\n"
                "输入：'这个产品太棒了，强烈推荐！'\n"
                "输出：{{\"sentiment\": \"positive\", \"confidence\": 0.95}}\n\n"
                "待分析内容：{content}"
            ),
            "score": 0.92,
            "source": "internal_best_practice",
            "avg_rating": 4.8,
            "usage_count": 1503,
            "framework": "few-shot",
            "tags": ["情感分析", "分类", "JSON"]
        },
        {
            "template_text": (
                "Task: Classify the sentiment of the user review.\n"
                "Rules:\n"
                "- Return JSON with 'sentiment' and 'confidence' keys\n"
                "- Confidence must be a float between 0 and 1\n\n"
                "Review: {content}"
            ),
            "score": 0.85,
            "source": "community",
            "avg_rating": 4.5,
            "usage_count": 892,
            "framework": "basic",
            "tags": ["sentiment", "classification"]
        },
    ]

    # 根据 intent 过滤更匹配的模板
    filtered = [t for t in mock_templates
                if any(tag in intent or intent in str(t.get("tags", [])) for tag in ["分类", "classification"])]

    return {
        "similar_templates": filtered or mock_templates[:1]
    }
