"""
RAG 三步走：
  1. 建索引（跑一次）   → 加载文件 → 切分 → 嵌入 → 存 ChromaDB
  2. 检索（反复跑）     → 从 ChromaDB 查 → 重排序 → 输出
"""
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import CrossEncoder

# ====== 不变的零件（只初始化一次） ======
chroma_client = chromadb.PersistentClient("./chroma")
collection = chroma_client.get_or_create_collection("demo-collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5") 

# ====== 建索引（只跑一次） ======
def build_index(input_dir="./docs"):
  docs = SimpleDirectoryReader(input_dir).load_data()

  splitter = SentenceSplitter(chunk_size = 500, chunk_overlap = 50)

  nodes = splitter.get_nodes_from_documents(docs)

  vectorStoreIndex = VectorStoreIndex(nodes = nodes, embed_model = embed_model, storage_context = StorageContext.from_defaults(vector_store = vector_store))

  print(f"处理了{collection}")

# ====== 检索（反复跑） ======
def search(question):
  index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

  results = index.as_retriever(similarity_top_k = 10).retrieve(question)
  doc_list = [item.text for item in results]

  model = CrossEncoder("BAAI/bge-reranker-v2-m3")

  reranker = model.rank(question, doc_list)

  for item in reranker[:3]:
    idx = item.get("corpus_id", -1)
    print(f"doc: {results[idx].node.text}")
    print(item['score'])




if __name__ == "__main__":
    # build_index()      # 第一次跑，建完索引后注释掉
    search("介绍一下悉尼旅游")
