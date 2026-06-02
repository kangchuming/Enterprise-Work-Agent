
"""
RAG 评测模板 —— 5 步法（单方法版本）
"""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from tools.langraph import app


# ====== Step 1: 明确需求 ======
"""
1. 评测对象：我的 RAG 管道
2. 指标：Faithfulness（防编造）+ Answer Relevancy（防跑题）
3. 数据：自己知识库里的真实问题，至少 10 组
4. 目的：看自己的 RAG 达不达标，不达标就针对性修
"""

# ====== Step 2: 搭评测引擎 ======
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

# TODO: 换成你自己的模型
evaluator_llm = DeepSeekEvalLLM(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)
evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ====== Step 3: 造数据（用你的 RAG 真实跑） ======
class LangGraphRAG:
    """薄包装器，只做格式转换"""
    def __init__(self):
        self._thread_counter = 0

    def query(self, question: str) -> dict:
        self._thread_counter += 1
        config = {"configurable": {"thread_id": self._thread_counter}}
        result = app.invoke({"question": question}, config)
        return {
            "answer": result.get("answer", ""),
            "contexts": [r["text"] for r in result.get("results", [])
                         if isinstance(r, dict) and "text" in r],
        }

def collect_results(rag, questions):
    """用 RAG 管道跑所有问题，收集 answer + contexts"""
    records = []
    for item in questions:
        result = rag.query(item["question"])
        records.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": item["ground_truth"],
        })
    return records


# TODO: 换成你自己的测试问题（至少 10 组）
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
    {
        "question": "去大堡礁潜水有什么推荐的地点？",
        "ground_truth": "大堡礁推荐凯恩斯、圣灵群岛和艾尔利海滩作为潜水出发点，可以看珊瑚礁和海洋生物。"
    },
    {
        "question": "澳大利亚旅游最佳季节是什么时候？",
        "ground_truth": "澳大利亚北部最佳旅游季节是4-9月旱季，南部10-4月夏季最宜人，整体春秋两季最舒适。"
    },
    {
        "question": "从悉尼到墨尔本怎么走比较方便？",
        "ground_truth": "从悉尼到墨尔本可乘飞机约1.5小时，自驾沿大洋路约9小时，也可坐火车约11小时。"
    },
    {
        "question": "珀斯有什么值得去的景点？",
        "ground_truth": "珀斯有国王公园、科茨洛海滩、弗里曼特尔港口和罗特尼斯岛，以短尾矮袋鼠闻名。"
    },
    {
        "question": "乌鲁鲁有什么特别的文化意义和旅游注意事项？",
        "ground_truth": "乌鲁鲁是原住民阿南古人的圣地，游客应尊重不攀爬的规定，最佳观赏时间是日出和日落时分。"
    },
]


# TODO: 换成你自己的 RAG 实例
records = collect_results(LangGraphRAG(), test_questions)


# ====== Step 4: Ragas 打分 ======
def run_eval(records, name):
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    df = result.to_pandas()
    df.to_csv(f"ragas_{name}_detail.csv", index=False)
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
    }


scores = run_eval(records, "my_rag")


# ====== Step 5: 看结果 + 画图 + 诊断 ======
# 设定及格线
THRESHOLD = 0.8

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

metric_names = ["faithfulness", "answer_relevancy"]
labels = ["Faithfulness\n（忠实度）", "Answer Relevancy\n（答案相关性）"]
values = [scores[m] for m in metric_names]
colors = ["#FF6B6B", "#4ECDC4"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

# 在柱子上方标分数
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2., v + 0.01,
            f"{v:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=14)

# 画一条 0.8 及格线
ax.axhline(y=THRESHOLD, color="gray", linestyle="--", linewidth=1.5, label=f"及格线 ({THRESHOLD:.0%})")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("RAG 评测结果", fontsize=16, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("ragas_single_result.png", dpi=150)
plt.show()

# 打印诊断
print(f"📊 Faithfulness:      {scores['faithfulness']:.1%}", end="")
if scores["faithfulness"] >= THRESHOLD:
    print("  ✅ 达标")
else:
    print(f"  ⚠️ 偏低 —— 建议：加强 prompt 约束 / 提高检索精度")

print(f"📊 Answer Relevancy:  {scores['answer_relevancy']:.1%}", end="")
if scores["answer_relevancy"] >= THRESHOLD:
    print("  ✅ 达标")
else:
    print(f"  ⚠️ 偏低 —— 建议：检查检索回来的 chunks 是否跑题")
