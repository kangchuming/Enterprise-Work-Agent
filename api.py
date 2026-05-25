# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.langraph_core import graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

# ====== 1. 初始化 LangGraph（和 main_langraph.py 一样）======
checkpointer = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
app_graph = graph.compile(checkpointer=checkpointer)

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
    decision:  str
    thread_id: int
# ====== 4. 辅助函数：从 result 里提取最终回答 ======
def extract_answer(result):
    try:
        message = list(reversed(result.get("messages", [])))[0]

        if message.get("role", '') == 'assistant' and  message["content"]:
            return message.get("content", "")
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
# ====== 5. 接口定义 ======
@app.get("/health")
async def getHealth():
    return {"status": "ok", "service": "ai agent"}

@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = app_graph.invoke({"input": req.input, "user": req.user, "resource": req.resource}, config)
    interrupt_resp = build_interrupt_response("__interrupt__", result, req)
    if interrupt_resp:
        return interrupt_resp
    else:
        return {
            "status": 'completed',
            "answer": extract_answer(result),
            "thread_id": req.thread_id,
            "step": result["step"]
        }

@app.post("/resume")
async def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = app_graph.invoke(Command(resume=req.decision), config)
    interrupt_resp = build_interrupt_response("__interrupt__", result, req)

    if interrupt_resp:
        return interrupt_resp
    else:
        return {
            "status": 'completed',
            "answer": extract_answer(result),
            "thread_id": req.thread_id,
            "step": result["step"]
        }

