from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import CrossEncoder

# ======== 0. 全局初始化（只跑一次） ========
chroma_client = chromadb.PersistentClient("./chroma")
collection = chroma_client.get_or_create_collection("demo-collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# ======== 1. State 定义 ========
class RAGState(TypedDict):
    question: str
    results: list
    reranked: list
    answer: str

# ======== 2. Node 函数 ========
def retrieve_node(state: RAGState):
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    results = index.as_retriever(similarity_top_k = 10).retrieve(state["question"])

    return {"results": results}

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

def no_result_node(state: RAGState):
    answer = "no answer"
    return {"answer": answer}

# ======== 3. 条件判断函数 ========
def should_answer(state: RAGState):
    if state.get("reranked", []) and state["reranked"][0].get("score", -1) >= 0.6:
        return "format_answer"
    return "no_result"

# ======== 4. 组装图 ========
graph = StateGraph(state_schema=RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("format_answer", format_answer_node)
graph.add_node("no_result", no_result_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "rerank")

# 分叉！
graph.add_conditional_edges(
    'rerank', 
    should_answer,
    {
        "format_answer": "format_answer",
        "no_result": "no_result"
    }
)

graph.add_edge("format_answer", END)
graph.add_edge("no_result", END)


app = graph.compile()

# ======== 5. 跑 ========
result = app.invoke({"question": "如何在阿德莱德游玩"})
print(f"result: {result}")
