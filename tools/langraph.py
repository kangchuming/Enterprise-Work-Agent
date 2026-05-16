from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
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

    return {"results": results}

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
    doc_list = [item.text for item in state.get("results", [])]

    model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    reranked = model.rank(state.get("question", ""), doc_list)

    return {"reranked": reranked}

def format_answer_node(state: RAGState):
    reranked = state.get("reranked", [])
    idx = reranked[0].get("corpus_id", -1)
    answer = state.get("results", [])[idx].text
    
    return {"answer": answer}

def uncertain_answer_node(state: RAGState):
    reranked = state.get("reranked", [])
    results = state.get("results", [])
    # idx = reranked[0].get("corpus_id", -1)
    # answer = state.get("results", [])[idx].text
    candidates = []

    for item in reranked[:3]:
        idx = item.get('corpus_id', -1)
        score = item.get('score', -10)
        res = results[idx]
        
        if 0 <= idx < len(results):
            text = res.text[:300]
        else:
            text = "（内容缺失）"
        candidates.append(f"【置信度】 {score} {text}")       
    
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

app = graph.compile()

# ======== 5. 跑 ========
result = app.invoke({"question": "如何在阿德莱德游玩"})
print(f"result: {result}")
