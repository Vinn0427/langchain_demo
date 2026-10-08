"""
Reciprocal Rank Fusion（RRF）：只看排名，不看原始分数。

    score(d) = Σ_{每一路召回 r}  1 / (k + rank_r(d))

Dense 的 cosine（0~1）和 BM25（0~十几，无上界）尺度完全不同，
直接 0.7 * dense + 0.3 * bm25 加权等于让 BM25 的量纲决定结果。RRF 把两路都换算成"排名"再相加，
在两路中排名都靠前的文档得分最高；只在一路中出现的文档也能得到 1/(k+rank) 的分数。
k 越大，排名靠前与靠后的差距越小（常用 60）。
"""
from langchain_core.documents import Document

import config


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Document]],
    k: int = config.RRF_K,
    top_n: int = config.RRF_TOP_K,
) -> list[Document]:
    scores: dict[str, float] = {}
    merged: dict[str, Document] = {}

    for source, documents in ranked_lists.items():          # source = "dense" / "sparse"
        for rank, doc in enumerate(documents, start=1):
            chunk_id = doc.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in merged:
                merged[chunk_id] = Document(id=doc.id, page_content=doc.page_content, metadata=dict(doc.metadata))
            else:
                merged[chunk_id].metadata.update(doc.metadata)  # 合并另一路的 *_score / *_rank

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    fused = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        doc = merged[chunk_id]
        doc.metadata.update(rrf_score=scores[chunk_id], rrf_rank=rank)
        fused.append(doc)
    return fused
