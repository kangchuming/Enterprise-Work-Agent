# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.langraph_core import graph
from langgraph.checkpoint.sqlite import SqliteSaver
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


load_dotenv()

# ====== 1. 初始化 LangGraph（和 main_langraph.py 一样）======
checkpointer = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
app_graph = graph.compile(checkpointer=checkpointer)

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
    version="0.1"
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
async def chat(req: ChatRequest):
    global cache_hits, cache_misses
    cached = cache.lookup(req.input)

    if cached:
        cache_hits += 1
        save_conversation(req, cached, "cache_hit")

        return {
            "status": "cache_hit",
            "answer": cached,
            "thread_id": req.thread_id,
            "from_cache": True
        }
    cache_misses += 1
    config = {"configurable": {"thread_id": req.thread_id}}
    result = app_graph.invoke({"input": req.input, "user": req.user, "resource": req.resource}, config)
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

        if "调用 LLM 出错" not in answer and "Error" not in answer:
            cache.save(req.input, answer)

        return {
            "status": status,
            "answer": answer,
            "thread_id": req.thread_id,
            "step": result["step"]
        }

@app.post("/resume")
async def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = app_graph.invoke(Command(resume=req.decision), config)
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
