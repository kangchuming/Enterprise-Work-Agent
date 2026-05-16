from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import chromadb
import sys
import os
import json
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"    # ← 加这行，全局禁止联网


from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import CrossEncoder
from tavily import TavilyClient

# ======== 0. 全局初始化（只跑一次） ========
embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5", local_files_only=True)
chroma_client = chromadb.PersistentClient("./chroma")
collection = chroma_client.get_or_create_collection("demo-collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
# ======== 1. State 定义 ========
class RAGState(TypedDict):
    question: str
    results: list
    reranked: list
    answer: str
    timeout: int

# ======== 2. Node 函数 ========
def retrieve_node(state: RAGState):
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    results = index.as_retriever(similarity_top_k = 10).retrieve(state["question"])

    clearn_result = [
        {"text": r.text, "score": float(r.score)} for r in results
    ]

    return {"results": clearn_result}

# ======== 0.5 工具函数 ========
def _get_text(item):
    """从结果项提取文本，兼容 dict / 正常 NodeWithScore / 反序列化损坏的对象"""
    if isinstance(item, dict):
        return item.get("text", "")
    if isinstance(item, str):
        return item
    
    # 直接尝试 .text，不先 hasattr（因为 hasattr 会触发 property 的异常）
    try:
        return item.text
    except (ValueError, AttributeError):
        pass
    
    # 兜底：通过 node.get_content()
    if hasattr(item, 'node') and hasattr(item.node, 'get_content'):
        try:
            return item.node.get_content()
        except Exception:
            pass
    return ""


def web_search_node(state: RAGState):
    tavilyClient = TavilyClient(os.getenv("TAVILY_API_KEY"))

    try:
        res = tavilyClient.search(query=state.get("question", ""), max_results=3, timeout = state.get("timeout", 500), include_raw_content="markdown")
        results = res.get('results', '')
        maxCount = 500

        if not results:
            return "没有搜索到结果"

        trip = []
        for i, obj in enumerate(results, 1):
            raw_text = obj.get("raw_content", '')
            title = obj.get("title", "")
            text = raw_text
            
            if len(raw_text) > maxCount:
                text = raw_text[:maxCount] + "...\n"
            trip.append(f"{i} {title} \n {text}")
        return f'搜索结果：{"\n".join(trip)}'
    except Exception as e:
        print(e)
        return f"搜索报错：{e}"

def rerank_node(state: RAGState):
    # 兼容 dict 和 NodeWithScore 对象
    doc_list = [_get_text(item) for item in state.get("results", [])]

    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    reranked = model.rank(state.get("question", ""), doc_list)

    clean = []
    for item in reranked:
        clean.append({
            "score": float(item["score"]),
            "corpus_id": int(item["corpus_id"])
        })
    return {"reranked": clean}


def format_answer_node(state: RAGState):
    reranked = state.get("reranked", [])
    results = state.get("results", [])
    idx = reranked[0].get("corpus_id", -1)
    answer = _get_text(results[idx])
    return {"answer": answer}

def uncertain_answer_node(state: RAGState):
    reranked = state.get("reranked", [])
    results = state.get("results", [])
    candidates = []

    for item in reranked[:3]:
        idx = item.get('corpus_id', -1)
        score = item.get('score', -10)
        
        if 0 <= idx < len(results):
            text = _get_text(results[idx])[:300]
        else:
            text = "（内容缺失）"
        candidates.append(f"【置信度 {score:.0%}】 {text}")  

    answer = "以下结果置信度中等，请自行判断: \n\n" + "\n---\n".join(candidates)
    return {"answer": answer}


def no_result_node(state: RAGState):
    answer = web_search_node(state)
    return {"answer": answer}

# ======== 3. 条件判断函数 ========
def should_answer(state: RAGState):
    if state.get("reranked", []) and state["reranked"][0].get("score", -1) >= 0.7:
        return "format_answer"
    elif state.get("reranked", []) and state["reranked"][0].get("score", - 1) >= 0.4:
        return "uncertain_answer"
    return "no_result_node"

# ======== 4. 组装图 ========
graph = StateGraph(state_schema=RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("format_answer", format_answer_node)
graph.add_node("no_result", no_result_node)
graph.add_node("uncertain_answer", uncertain_answer_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "rerank")

# 分叉！
graph.add_conditional_edges(
    'rerank', 
    should_answer,
    {
        "format_answer": "format_answer",
        "uncertain_answer": "uncertain_answer",
        "no_result": "no_result"
    }
)

graph.add_edge("format_answer", END)
graph.add_edge("no_result", END)
graph.add_edge("uncertain_answer", END)

# checkpointer = InMemorySaver()
checkpointer = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))

app = graph.compile(checkpointer = checkpointer)

# ======== 5. 跑 ========

config = {"configurable": {"thread_id": 1}}
result1 = app.invoke({"question": "如何在阿德莱德游玩"}, config)

print("=" * 60)
print("【第一次运行 - 正常流程】")
print(f"answer: {result1.get('answer', 'NO ANSWER')[:150]}...")
print(f"results 数量: {len(result1.get('results', []))}")
print(f"reranked 数量: {len(result1.get('reranked', []))}")

# --- 回到 retrieve 之后，注入假数据 ---
history = list(app.get_state_history(config))
checkpoint_before_rerank = history[1]

fake_result = {
    "text": "【注入数据】阿德莱德是南澳大利亚州的首府，以葡萄酒产区、海滩和艺术节闻名。",
    "score": 0.6
}

app.update_state(
    checkpoint_before_rerank.config,
    {"results": [fake_result] + checkpoint_before_rerank.values["results"]}
)

# --- 重新从 retrieve 之后运行 ---
result2 = app.invoke(None, checkpoint_before_rerank.config)

print("=" * 60)
print("【第二次运行 - 注入 fake_result 后重新跑】")
print(f"answer: {result2.get('answer', 'NO ANSWER')[:150]}...")
print(f"results 数量: {len(result2.get('results', []))}")
print(f"reranked 数量: {len(result2.get('reranked', []))}")

# --- 对比结果 ---
print("=" * 60)
print("【对比】")
print(f"修改前 answer: {result1.get('answer', '')[:100]}")
print(f"修改后 answer: {result2.get('answer', '')[:100]}")

# --- 查看完整历史时间线 ---
print("=" * 60)
print("【历史步骤时间线】")
for snap in app.get_state_history(config):
    step = snap.metadata.get("step", "?")
    task = snap.metadata.get("source", "?")

    # 截取 answer 前 60 字显示
    ans = snap.values.get("answer", "")
    if isinstance(ans, str) and ans:
        ans_preview = ans[:60].replace("\n", " ")
    else:
        ans_preview = "(无)"

    # 显示 results 数量
    res = snap.values.get("results", [])
    res_count = len(res) if res else 0

    print(f"  Step {step} | node: {task} | results: {res_count}个 | answer: {ans_preview}")