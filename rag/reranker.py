from __future__ import annotations
 
K = 60
 
def rrf_merge(
    bm25_results: list[dict],
    dense_results: list[dict],
    top_k: int = 6
) -> list[dict]:
    scores: dict[str, float] = {}
    docs_map: dict[str, dict] = {}
 
    for result_list in (bm25_results, dense_results):
        for item in result_list:
            text = item['doc']['text']
            key  = text[:80]   # fingerprint — first 80 chars as dedup key
            scores[key]   = scores.get(key, 0.0) + 1.0 / (K + item['rank'] + 1)
            docs_map[key] = item['doc']
 
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{'doc': docs_map[k], 'rrf_score': s} for k, s in fused]
