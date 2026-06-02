"""
evaluate_ragas.py  —— 对比 Naive RAG vs Hybrid+Reranker 三级路由
放在项目根目录下运行
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ====== 导入你的 RAG 组件（和 tools/langraph.py 共用底层） ======
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from sentence_transformers import CrossEncoder

load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"     # 和 langraph.py 保持一致，禁止联网


# ============================================================
# 1. 评测引擎：Ragas 需要一 LLM + 一 Embedding
# ============================================================
class DeepSeekEvalLLM(ChatOpenAI):
    """
    DeepSeek API 只支持 n=1，Ragas 内部可能传 n>1，
    这里强制覆盖 n=1，避免 400 错误。
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


evaluator_llm = DeepSeekEvalLLM(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)

evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# ============================================================
# 2. 全局初始化你的 RAG 组件（复用 langraph.py 的逻辑）
# ============================================================
embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5", local_files_only=True)
chroma_client = chromadb.PersistentClient("./chroma")
collection = chroma_client.get_or_create_collection("demo-collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")


# ============================================================
# 3. 定义两个 RAG 管道
# ============================================================
class NaiveRAG:
    """
    Naive RAG：只用 Embedding 检索。
    - 不做 Reranker
    - 不做三级路由
    - 直接返回 top_k 文档作为"答案"
    """
    def __init__(self, top_k=5):
        self.top_k = top_k

    def query(self, question: str) -> dict:
        """返回 {"answer": str, "contexts": list[str]}"""
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        results = index.as_retriever(
            similarity_top_k=self.top_k
        ).retrieve(question)

        contexts = [r.text for r in results]
        # Naive：直接拼接文档作为答案
        answer = "\n\n".join(contexts[:3])

        return {"answer": answer, "contexts": contexts}


class HybridRAG:
    """
    Hybrid + Reranker 三级路由（和 langraph.py 逻辑完全一致）：
    1. Embedding 检索 top_k=10
    2. CrossEncoder Reranker 重排序
    3. 三级路由：
       - score ≥ 0.7 → 自信返回 top-1
       - 0.4 ≤ score < 0.7 → 不确定，返回 top-3 带置信度
       - score < 0.4 → 无结果
    """
    def __init__(self, retrieval_top_k=10):
        self.retrieval_top_k = retrieval_top_k

    def query(self, question: str) -> dict:
        """返回 {"answer": str, "contexts": list[str], "route": str}"""
        # Step 1: Embedding 检索
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        raw_results = index.as_retriever(
            similarity_top_k=self.retrieval_top_k
        ).retrieve(question)

        doc_list = [r.text for r in raw_results]

        # Step 2: Reranker 重排序
        ranked = reranker_model.rank(question, doc_list)
        reranked = [
            {"score": float(item["score"]), "corpus_id": int(item["corpus_id"])}
            for item in ranked
        ]

        top_score = reranked[0]["score"]

        # Step 3: 三级路由（和 langraph.py should_answer 一致）
        if top_score >= 0.7:
            route = "confident"
            idx = reranked[0]["corpus_id"]
            answer = doc_list[idx]
            contexts = [doc_list[i["corpus_id"]] for i in reranked[:3]]
        elif top_score >= 0.4:
            route = "uncertain"
            candidates = []
            for item in reranked[:3]:
                idx = item["corpus_id"]
                score = item["score"]
                text = doc_list[idx][:300]
                candidates.append(f"【置信度 {score:.0%}】 {text}")
            answer = "以下结果置信度中等，请自行判断:\n\n" + "\n---\n".join(candidates)
            contexts = [doc_list[i["corpus_id"]] for i in reranked[:3]]
        else:
            route = "no_result"
            answer = "未找到高置信度结果。"
            contexts = [doc_list[i["corpus_id"]] for i in reranked[:3]]

        return {"answer": answer, "contexts": contexts, "route": route}


# ============================================================
# 4. 测试数据（你需要换成自己知识库的真实问题）
# ============================================================
# 格式说明：
#   question     → 用户提的问题
#   ground_truth → 人工写的标准答案（faithfulness 不需要，但 context_recall 需要）
#
# 建议至少准备 10-20 组，太少没有统计意义
test_questions = [
    {
        "question": "如何在阿德莱德游玩？",
        "ground_truth": "阿德莱德可以参观巴罗莎谷葡萄酒产区、阿德莱德中央市场、格莱内尔格海滩，还可以参加艺术节活动。"
    },
    {
        "question": "悉尼有哪些著名的旅游景点？",
        "ground_truth": "悉尼著名景点包括悉尼歌剧院、海港大桥、邦迪海滩、曼利海滩、达令港和塔龙加动物园。"
    },
    {
        "question": "墨尔本有什么特色美食推荐？",
        "ground_truth": "墨尔本以咖啡文化、巷弄美食、意大利菜和亚洲融合菜闻名，推荐去Degraves Street和唐人街。"
    },
    {
        "question": "澳大利亚有哪些必去的自然景观？",
        "ground_truth": "澳大利亚必去自然景观包括大堡礁、乌鲁鲁巨岩、十二门徒岩、蓝山国家公园和卡卡杜国家公园。"
    },
    {
        "question": "布里斯班适合带小孩去玩吗？",
        "ground_truth": "布里斯班非常适合亲子游，有南岸公园人工海滩、龙松考拉动物园和昆士兰博物馆。"
    },
    # ... 继续加你自己的问题
]


# ============================================================
# 5. 跑两个管道 + 收集结果
# ============================================================
def collect_results(rag, questions, name):
    """用指定 RAG 管道跑所有问题，收集 answer 和 contexts"""
    records = []
    for i, item in enumerate(questions):
        result = rag.query(item["question"])
        records.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": item["ground_truth"],
        })
        route_info = result.get("route", "N/A")
        print(f"  [{i+1}/{len(questions)}] {name} → route={route_info}")
    return records


print("🔄 运行 Naive RAG（仅 Embedding）...")
naive_rag = NaiveRAG(top_k=5)
naive_records = collect_results(naive_rag, test_questions, "Naive")

print("\n🔄 运行 Hybrid + Reranker 三级路由...")
hybrid_rag = HybridRAG(retrieval_top_k=10)
hybrid_records = collect_results(hybrid_rag, test_questions, "Hybrid")

# 看看 Hybrid 的路由分布
from collections import Counter
route_dist = Counter(r.get("route", "?") for r in hybrid_records if "route" in r)
print(f"\n📊 Hybrid 路由分布: {dict(route_dist)}")


# ============================================================
# 6. 转成 Ragas Dataset 格式
# ============================================================
def to_ragas_dataset(records):
    """把 records 列表转成 Ragas 需要的 Dataset 对象"""
    return Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

naive_dataset = to_ragas_dataset(naive_records)
hybrid_dataset = to_ragas_dataset(hybrid_records)

print(f"\n📊 Naive 数据集大小: {len(naive_dataset)} 条")
print(f"📊 Hybrid 数据集大小: {len(hybrid_dataset)} 条")


# ============================================================
# 7. 跑 Ragas 评测
# ============================================================
def run_evaluation(dataset, name):
    """跑 Ragas 评测，返回平均分字典"""
    print(f"\n🔄 Ragas 评测 {name} 中...（每条数据会调多次 LLM，请耐心等待）")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    scores = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
    }

    # 保存逐条明细
    df.to_csv(f"ragas_{name}_detail.csv", index=False)
    print(f"   ✅ {name} 完成！明细已保存到 ragas_{name}_detail.csv")
    return scores


naive_scores = run_evaluation(naive_dataset, "naive")
hybrid_scores = run_evaluation(hybrid_dataset, "hybrid")


# ============================================================
# 8. 画对比柱状图
# ============================================================
# ============================================================
# 8. 画对比柱状图
# ============================================================
# 先配置中文字体（必须在 plt.subplots 之前）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

metrics = ["faithfulness", "answer_relevancy"]
metric_labels = ["Faithfulness（忠实度）", "Answer Relevancy（答案相关性）"]

naive_vals = [naive_scores[m] for m in metrics]
hybrid_vals = [hybrid_scores[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, naive_vals, width,
               label="Naive RAG\n(仅 Embedding)", color="#FF6B6B", edgecolor="white")
bars2 = ax.bar(x + width/2, hybrid_vals, width,
               label="Hybrid + Reranker\n三级路由", color="#4ECDC4", edgecolor="white")

# 柱子上标注数值
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
            f"{h:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
            f"{h:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylabel("Score", fontsize=13)
ax.set_title("RAG 管道评测对比：Naive vs Hybrid + Reranker", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("ragas_comparison.png", dpi=150)
plt.show()
print("\n📈 柱状图已保存到: ragas_comparison.png")

# ... 后面不变 ...
plt.tight_layout()
plt.savefig("ragas_comparison.png", dpi=150)
plt.show()
print("\n📈 柱状图已保存到: ragas_comparison.png")


# ============================================================
# 9. 输出最终总结
# ============================================================
faith_improve = (hybrid_scores["faithfulness"] - naive_scores["faithfulness"]) * 100
rel_improve = (hybrid_scores["answer_relevancy"] - naive_scores["answer_relevancy"]) * 100

print("\n" + "=" * 60)
print("📊 最终评测结果")
print("=" * 60)
print(f"Naive RAG     → Faithfulness: {naive_scores['faithfulness']:.1%}  "
      f"| Answer Relevancy: {naive_scores['answer_relevancy']:.1%}")
print(f"Hybrid+Rerank → Faithfulness: {hybrid_scores['faithfulness']:.1%}  "
      f"| Answer Relevancy: {hybrid_scores['answer_relevancy']:.1%}")
print("-" * 60)
print(f"✅ Faithfulness 从 {naive_scores['faithfulness']:.0%} "
      f"提升到 {hybrid_scores['faithfulness']:.0%}"
      f"（+{faith_improve:.1f} 个百分点）")
print(f"✅ Answer Relevancy 从 {naive_scores['answer_relevancy']:.0%} "
      f"提升到 {hybrid_scores['answer_relevancy']:.0%}"
      f"（+{rel_improve:.1f} 个百分点）")
print("=" * 60)
