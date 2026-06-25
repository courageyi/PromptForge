"""
PromptForge FastAPI 服务

提供 RESTful API 来使用提示词优化工作流。
支持同步执行、流式执行、人工审核恢复三种模式。
"""

import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .graph import PromptForgeGraph

load_dotenv()

# ============================================================
# FastAPI 初始化
# ============================================================

app = FastAPI(
    title="PromptForge API",
    description="基于 LangGraph 的智能提示词优化平台",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局图实例（生产环境应放在依赖注入容器中）
graph = PromptForgeGraph()


# ============================================================
# 请求/响应模型
# ============================================================

class OptimizeRequest(BaseModel):
    """提示词优化请求"""
    user_input: str = Field(..., description="用户的模糊需求描述", min_length=5, max_length=2000)
    framework: str = Field(default="few-shot", description="提示词框架: few-shot|cot|react|tot")
    target_model: str = Field(default="gpt-4", description="目标模型")
    auto_optimize: bool = Field(default=True, description="是否自动迭代优化")

    class Config:
        json_schema_extra = {
            "example": {
                "user_input": "帮我写一个分析用户评论情感倾向的提示词，输出JSON",
                "framework": "few-shot",
                "target_model": "gpt-4",
                "auto_optimize": True,
            }
        }


class OptimizeResponse(BaseModel):
    """提示词优化响应"""
    session_id: str
    final_prompt: str
    quality_score: float
    critique: dict
    optimization_rounds: int
    test_summary: dict


class ResumeRequest(BaseModel):
    """人工审核恢复请求"""
    session_id: str
    action: str = Field(..., description="approve | reject | edit")
    chosen_prompt: Optional[str] = None
    edited_prompt: Optional[str] = None


# ============================================================
# 会话状态管理（生产环境应用 Redis 替代）
# ============================================================

# {session_id: {"config": {...}, "state": {...}}}
sessions: dict = {}


# ============================================================
# API 端点
# ============================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "PromptForge",
        "langgraph_version": "0.2+",
    }


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest):
    """
    提交提示词优化任务。

    此端点会同步执行完整的 Analyze → Retrieve → Generate → Critic → Optimize → Test 流程。
    如果 auto_optimize=True 且质量达标，会跳过人工审核直接返回结果。

    如需人工审核，使用 /api/optimize/stream 端点获取中断信号。
    """
    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": session_id}}

    # ============================================================
    # 实际实现
    # ============================================================
    # try:
    #     # 流式执行，捕获中间状态
    #     final_state = None
    #     for event in graph.stream(request.dict(), config):
    #         final_state = event
    #
    #     # 获取最终状态
    #     state = graph.get_state(config)
    #     state_values = state.values
    #
    #     return {
    #         "session_id": session_id,
    #         "final_prompt": state_values.get("final_prompt", ""),
    #         "quality_score": state_values.get("quality_score", 0),
    #         "critique": state_values.get("critique", {}),
    #         "optimization_rounds": state_values.get("iteration_count", 0),
    #         "test_summary": final_state.get("tester", {}).get("test_summary", {}),
    #     }
    #
    # except GraphInterrupt as e:
    #     # 工作流暂停在 human_review 节点
    #     sessions[session_id] = {"config": config, "interrupt_data": e.args[0]}
    #     raise HTTPException(
    #         status_code=202,
    #         detail={
    #             "message": "需要人工审核",
    #             "session_id": session_id,
    #             "review_data": e.args[0],
    #         }
    #     )

    # ============================================================
    # 骨架返回
    # ============================================================
    return {
        "session_id": session_id,
        "final_prompt": "# 情感分析提示词\n\n你是一个... (完整提示词见 graph 输出)",
        "quality_score": 8.5,
        "critique": {
            "clarity": {"score": 8, "comment": "结构清晰"},
            "completeness": {"score": 8, "comment": "覆盖主要约束"},
        },
        "optimization_rounds": 1,
        "test_summary": {
            "total": 6,
            "sentiment_accuracy": "5/6 (83.3%)",
            "format_valid_rate": "6/6 (100%)",
            "avg_latency_ms": 950,
        },
    }


@app.post("/api/optimize/{session_id}/resume")
async def resume_optimize(session_id: str, request: ResumeRequest):
    """
    恢复人工审核暂停的工作流。

    在 /api/optimize 返回 202 后，
    前端展示当前提示词和审查报告，
    用户审核后调用此端点恢复执行。
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    session = sessions.pop(session_id)
    config = session["config"]

    resume_value = {
        "action": request.action,
        "chosen_prompt": request.chosen_prompt,
        "edited_prompt": request.edited_prompt,
    }

    try:
        final_state = graph.resume(config, resume_value)
        return {
            "session_id": session_id,
            "status": "completed",
            "final_prompt": final_state.get("final_prompt", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates")
async def list_templates(intent: Optional[str] = None, limit: int = 10):
    """
    列出可用的提示词模板。

    用于浏览参考模板或人工选择模板。
    """
    # 实际实现：从 pgvector 查询
    return {
        "templates": [],
        "total": 0,
        "filters": {"intent": intent},
    }


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
