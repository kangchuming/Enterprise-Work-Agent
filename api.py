# api.py
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langsmith import traceable, get_current_run_tree
from agent.langraph_core import graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
import sqlite3
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text
from datetime import datetime
from tools.semantic_cache import SemanticCache
from sentence_transformers import SentenceTransformer
import aiosqlite                              # ← 新增
from contextlib import asynccontextmanager  


load_dotenv()

# ====== 1. 初始化 LangGraph —— 改为 lifespan 懒加载 ======
_checkpointer = None
_app_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer, _app_graph
    # Startup: 事件循环已运行，可以正常创建
    conn = await aiosqlite.connect("checkpoints.db")
    _checkpointer = AsyncSqliteSaver(conn)
    _app_graph = graph.compile(checkpointer=_checkpointer)
    yield  # ← App 运行期间保持连接
    # Shutdown: 关闭数据库连接
    await conn.close()

# 创建引擎 —— SQLite 只需要一个文件路径
engine = create_engine("sqlite:///app.db", echo=False)

class STEmbedding:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        self.model=SentenceTransformer(model_name, local_files_only=True)
    def embed_query(self, input: str):
        return self.model.encode(input)

embedding_model = STEmbedding()
cache = SemanticCache(
    embedding_model=embedding_model,
    similarity_threshold=0.72,
    ttl=86400
)
cache_hits = 0
cache_misses = 0

# 1.2 声明基类 —— 所有 Model 都继承它
class Base(DeclarativeBase):
    pass

# 1.5. 定义表
class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[int] = mapped_column(Integer)
    input: Mapped[str] = mapped_column(Text)
    user: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50))
    answer: Mapped[str] = mapped_column(Text)

# 1.8 建表
Base.metadata.create_all(engine)

# ====== 2. FastAPI 应用 ======
app = FastAPI(
    title="fastAPI应用",
    description="fastapi+langraph",
    version="0.1",
    lifespan=lifespan, 
)

# ====== 3. Pydantic 请求模型 ======
class ChatRequest(BaseModel):
    input: str
    thread_id: int
    user: str
    resource: str

class ResumeRequest(BaseModel):
    user: str
    decision:  str
    thread_id: int
# ====== 4. 辅助函数：从 result 里提取最终回答 ======
def extract_answer(result):
    try:
        messages = result.get("messages", [])

        if not messages:
            return "No response generated"

        message = list(reversed(messages))[0]

        if message.get("role", '') == 'assistant' and  message["content"]:
            return message.get("content", "")
        return "No answer found"
    except Exception as e:
            return f"错误{e}"

def build_interrupt_response(interrupt_key: str, result, req):
    if interrupt_key in result:
        return {
            "status": interrupt_key,
            "message": "处于中断阶段",
            "interrupt_info": result[interrupt_key][0].value,
            "thread_id": req.thread_id
        }
    return None

def save_conversation(req, answer: str, status: str):
    try:
        with Session(engine) as session:
            conv = Conversation(
                thread_id = req.thread_id,
                input = getattr(req, "input", req.decision if hasattr(req, "decision") else ''),
                user = getattr(req, 'user', ""),
                answer = answer,
                status = status
            )
            session.add(conv)
            session.commit()
    except Exception as e:
        print(f"操作报错: {e}")

# ====== 5. 接口定义 ======
@app.get("/health")
async def getHealth():
    return {"status": "ok", "service": "ai agent"}

@app.post("/chat")
@traceable(name="chat_endpoint", run_type="chain")
async def chat(req: ChatRequest):
    global cache_hits, cache_misses
    cached = cache.lookup(req.input)
    run = get_current_run_tree()

    if cached:
        cache_hits += 1

        if run:
            run.add_metadata({
                "cached": 'hits',
                "cache_hits_number": cache_hits
        })
        save_conversation(req, cached, "cache_hit")

        return {
            "status": "cache_hit",
            "answer": cached,
            "thread_id": req.thread_id,
            "from_cache": True
        }
        
        
    cache_misses += 1

    if run:
            run.add_metadata({
                "cached": 'miss',
                "cache_hits_number": cache_hits
            })

    config = {
            "configurable": {"thread_id": req.thread_id}, 
            "metadata": {
                "user": req.user,
                "thread_id": req.thread_id,
                "resource": req.resource,
            }
    }
    # result = app_graph.invoke({"input": req.input, "user": req.user, "resource": req.resource}, config)
    # interrupt_resp = build_interrupt_response("__interrupt__", result, req)
    # if interrupt_resp:
    #     answer = extract_answer(result)
    #     status = interrupt_resp.get("status", '')
    #     save_conversation(req, answer, status)

    #     return interrupt_resp
    # else:
    #     answer = extract_answer(result)
    #     status = 'completed'
    #     save_conversation(req, answer, status)

    #     if "调用 LLM 出错" not in answer and "Error" not in answer:
    #         cache.save(req.input, answer)

    #     return {
    #         "status": status,
    #         "answer": answer,
    #         "thread_id": req.thread_id,
    #         "step": result["step"]
    #     }

    # ========== 新增：节点状态映射表 ==========
    NODE_STATUS_MAP = {
        "prepare": "已完成上下文准备",
        "execute": "工具执行完成",
        "finish":   "任务完成",
        # llm 节点动态判断，不写死
    }
    TRACKED_NODES = {"prepare", "llm", "execute", "finish"}  # 只关心这 4 个

    async def event_stream():
        accumulated_content = ""

        # ====== 新增：推送初始状态 ======
        yield f"data: {json.dumps({'node': 'start', 'status': 'Agent 已启动'}, ensure_ascii=False)}\n\n"

        async for event in _app_graph.astream_events(
            {"input": req.input, "user": req.user, "resource": req.resource},
            config,
            version="v2"        # ← 必须 v2，才有 on_chain_start/end
        ):
            kind = event["event"]
            name = event.get("name", "")

            # ================================================================
            # 1. Token 流式输出（你现有的逻辑，保持不变）
            # ================================================================
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    accumulated_content += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            # ================================================================
            # 2. ★ 新增：节点开始 —— 推送 "正在准备..." 之类
            # ================================================================
            elif kind == "on_chain_start" and name in TRACKED_NODES:
                START_STATUS = {
                    "prepare": "正在准备上下文...",
                    "llm":     "模型正在推理...",
                    "execute": "正在执行工具...",
                    "finish":  "正在整理结果...",
                }
                status = START_STATUS.get(name, "处理中...")
                yield f"data: {json.dumps({'node': name, 'status': status, 'action': 'start'}, ensure_ascii=False)}\n\n"

            # ================================================================
            # 3. ★ 新增：节点完成 —— 这就是你要的「每完成一个节点推送」
            # ================================================================
            elif kind == "on_chain_end" and name in TRACKED_NODES:
                output = event["data"].get("output", {})

                if name == "llm":
                    # llm 节点特殊处理：根据是否有工具调用给不同状态
                    tool_calls = output.get("tool_calls", [])
                    if tool_calls:
                        tool_names = [tc["function"]["name"] for tc in tool_calls]
                        status = f"正在调用工具: {', '.join(tool_names)}"
                    else:
                        status = "模型回复完成"
                else:
                    status = NODE_STATUS_MAP.get(name, "处理完成")

                yield f"data: {json.dumps({'node': name, 'status': status, 'action': 'end'}, ensure_ascii=False)}\n\n"

            # ================================================================
            # 4. 工具事件（你现有的逻辑，保持不变）
            # ================================================================
            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_end":
                output = str(event['data'].get('output', ''))[:200]
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output': output}, ensure_ascii=False)}\n\n"

        # ========== 善后逻辑（你现有的逻辑，保持不变）==========
        state = await _app_graph.aget_state(config)
        state_values = state.values if state else {}

        if state_values and "__interrupt__" in state_values:
            interrupt_info = state_values["__interrupt__"][0].value
            save_conversation(req, accumulated_content or '(pending)', '__interrupt__')
            yield f"data: {json.dumps({'type': 'interrupt', 'message': '需要审批', 'interrupt_info': str(interrupt_info), 'thread_id': req.thread_id}, ensure_ascii=False)}\n\n"
        else:
            messages = state_values.get("messages", [])
            answer = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    answer = msg["content"]
                    break
            answer = answer or accumulated_content or "No response generated"
            save_conversation(req, answer, 'completed')
            yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'thread_id': req.thread_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/resume")
async def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await _app_graph.ainvoke(Command(resume=req.decision), config)
    interrupt_resp = build_interrupt_response("__interrupt__", result, req)

    if interrupt_resp:
        answer = extract_answer(result)
        status = interrupt_resp.get("status", '')
        save_conversation(req, answer, status)

        return interrupt_resp
    else:
        answer = extract_answer(result)
        status = 'completed'
        save_conversation(req, answer, status)

        return {
            "status": status,
            "answer": answer,
            "thread_id": req.thread_id,
            "step": result["step"]
        }

@app.get("/cache/status")
async def cache_status():
    global cache_hits, cache_misses
    cache_all = cache_misses + cache_hits
    hits_rate = (cache_hits / cache_all * 100) if cache_all > 0 else 0.0

    return {
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hits_rate": hits_rate,
        "cache_keys": cache.count() 
    }
