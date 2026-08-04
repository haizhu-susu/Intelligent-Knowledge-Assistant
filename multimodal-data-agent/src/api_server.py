"""
FastAPI 后端服务：提供数据分析 API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
import io
import json
import tempfile

from src.agent import DataAnalysisAgent


# 创建 FastAPI 应用
app = FastAPI(
    title="Multimodal Data Analysis Agent API",
    description="基于多模态内容理解的全自动数据分析可视化 API",
    version="0.1.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储分析会话
sessions: Dict[str, DataAnalysisAgent] = {}


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    session_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """分析结果模型"""
    success: bool
    data_shape: tuple
    columns: List[str]
    summary: Dict[str, Any]
    insights: List[str]
    anomalies: Dict[str, Any]


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用多模态数据分析 Agent API",
        "version": "0.1.0",
        "endpoints": {
            "POST /analyze": "上传并分析数据",
            "POST /query": "自然语言查询",
            "GET /report/{session_id}": "获取分析报告",
            "DELETE /session/{session_id}": "清除会话"
        }
    }


@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    """
    上传并分析数据文件
    
    - **file**: CSV、Excel 或 JSON 格式的数据文件
    """
    try:
        # 读取文件内容
        contents = await file.read()
        
        # 根据文件类型解析
        file_type = file.filename.split('.')[-1].lower()
        
        if file_type == 'csv':
            df = pd.read_csv(io.BytesIO(contents))
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(contents))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式：{file_type}")
        
        # 创建新的分析会话
        session_id = f"session_{len(sessions) + 1}"
        agent = DataAnalysisAgent(verbose=False)
        
        # 执行分析
        result = agent.analyze(df, auto_visualize=False)
        
        # 保存会话
        sessions[session_id] = agent
        
        return {
            "success": True,
            "session_id": session_id,
            "data_shape": result['data_shape'],
            "columns": result['columns'],
            "summary": result['summary'],
            "insights": result['insights'],
            "anomalies": result['anomalies']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_data(request: QueryRequest):
    """
    自然语言查询数据
    
    - **question**: 用户的问题
    - **session_id**: 会话 ID（可选）
    """
    session_id = request.session_id
    
    if session_id and session_id in sessions:
        agent = sessions[session_id]
    else:
        # 如果没有指定会话或会话不存在，返回错误
        raise HTTPException(
            status_code=404, 
            detail="请先上传数据进行分析 (POST /analyze)"
        )
    
    # 执行查询
    result = agent.query(request.question)
    
    return {
        "question": request.question,
        "answer": result['answer'],
        "data": result.get('data'),
        "chart_type": result.get('chart_type')
    }


@app.get("/visualize/{session_id}")
async def get_visualizations(session_id: str):
    """
    获取可视化图表
    
    - **session_id**: 会话 ID
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    agent = sessions[session_id]
    
    if agent.current_data is None:
        raise HTTPException(status_code=400, detail="没有可可视化的数据")
    
    # 生成可视化
    viz_result = agent.visualizer.auto_visualize(agent.current_data)
    
    return {
        "charts": viz_result.get('charts', []),
        "recommendations": viz_result.get('recommendations', [])
    }


@app.get("/report/{session_id}")
async def generate_report(session_id: str, format: str = "html"):
    """
    生成分析报告
    
    - **session_id**: 会话 ID
    - **format**: 输出格式 (html 或 json)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    agent = sessions[session_id]
    
    if agent.current_data is None:
        raise HTTPException(status_code=400, detail="没有可生成报告的数据")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp:
        report_path = agent.generate_report(tmp.name, format=format)
        
        return FileResponse(
            report_path,
            media_type=f"text/{format}",
            filename=f"analysis_report.{format}"
        )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    清除分析会话
    
    - **session_id**: 会话 ID
    """
    if session_id in sessions:
        sessions[session_id].clear()
        del sessions[session_id]
        return {"message": f"会话 {session_id} 已清除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
