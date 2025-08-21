
from __future__ import annotations
def token_contributions(text: str, vec, coef, topk: int = 10) -> dict:
    X = vec.transform([text])
    vals = X.tocoo()
    contrib = {}
    for i, v in zip(vals.col, vals.data):
        contrib[i] = coef[i] * v
    idx_to_tok = {i: t for t,i in vec.vocabulary_.items()}
    items = [(idx_to_tok.get(i, f"feat_{i}"), c) for i,c in contrib.items()]
    pos = sorted([x for x in items if x[1] > 0], key=lambda x: x[1], reverse=True)[:topk]
    neg = sorted([x for x in items if x[1] < 0], key=lambda x: x[1])[:topk]
    return {"top_positive": pos, "top_negative": neg}
