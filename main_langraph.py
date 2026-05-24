#main_langraph.py
from agent.langraph_core import graph
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
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
from llama_index.core.node_parser import SentenceSplitter

# ======== 0. 全局初始化（只跑一次） ========
embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5", local_files_only=True)
chroma_client = chromadb.PersistentClient("./chroma")
collection = chroma_client.get_or_create_collection("demo-collection")
vector_store = ChromaVectorStore(chroma_collection=collection)

checkpointer = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
app = graph.compile(checkpointer = checkpointer)

# ======== 5. 跑 ========

config = {"configurable": {"thread_id": 6}}
result = app.invoke({"input": "网络搜一下明朝那些事儿这本书，并把书籍简介写到新建的kcm.md",}, config)

if "__interrupt__" in result:
    interrupt_info = result["__interrupt__"][0].value
    print(f"interrupt_info: {interrupt_info}")
    decision = input("是否批准执行？(yes/no): ").strip()   
    result =  app.invoke(Command(resume=decision), config)
    print(f"result, 根据用户输入后执行结果为：{result}")
else:
    print(f"一次性执行result：{result}")