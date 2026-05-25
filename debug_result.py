from agent.langraph_core import graph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3, json

checkpointer = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
app_graph = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "debug_001"}}
result = app_graph.invoke(
    {"input": "网络搜一下明朝那些事儿这本书，并把书籍简介写到新建的kcm.md"},
    config
)

print("=== keys ===")
print(list(result.keys()))

print("=== full result ===")
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
