"""语义缓存：Embedding → Redis 相似匹配 → 命中返回"""
import json
import numpy as np
import redis
import hashlib
from typing import Optional


class SemanticCache:
    def __init__(
        self,
        embedding_model,
        redis_host="localhost",
        redis_port=6379,
        similarity_threshold=0.92,
        ttl=86400,                    # 24 小时过期
    ):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
        self.ttl = ttl
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.embedding_model.embed_query(text), dtype=np.float32)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _make_key(self, text: str) -> str:
        return f"cache:{hashlib.md5(text.encode()).hexdigest()[:12]}"

    # ─── 核心：查缓存 ───
    def lookup(self, query: str) -> Optional[str]:
        query_vec = self._embed(query)
        cursor, best_sim, best_answer = 0, 0.0, None

        while True:
            cursor, keys = self.redis.scan(cursor, match="cache:*", count=50)
        
            for key in keys:
                try:
                    data = json.loads(self.redis.get(key))
                    sim = self._cosine(query_vec, np.array(data["embedding"], dtype=np.float32))
                    if sim > best_sim:
                        best_sim, best_answer = sim, data["answer"]
                except Exception:
                    continue

            if cursor == 0:
                break
        print(f"best_sim, {best_sim}, best_answer, {best_answer}")
        return best_answer if best_sim >= self.threshold else None

    # ─── 核心：存缓存 ───
    def save(self, query: str, answer: str):
        data = json.dumps({
            "query": query,
            "answer": answer,
            "embedding": self._embed(query).tolist()
        }, ensure_ascii=False)

        self.redis.setex(self._make_key(query), self.ttl, data)

    # ─── 辅助 ───
    def count(self) -> int:
        """当前缓存条数"""
        n, cursor = 0, 0
        while True:
            cursor, keys = self.redis.scan(cursor, match="cache:*")
            n += len(keys)
            if cursor == 0:
                break
        return n
