
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import score_one, load_models
from src.explain import token_contributions

app = FastAPI(title="TikTok Viral Content Predictor API")

class Item(BaseModel):
    title: str
    script: str | None = ""
    hashtags: str | None = ""

@app.post("/score")
def score(item: Item):
    return score_one(item.title, item.script or "", item.hashtags or "")

@app.post("/explain")
def explain(item: Item):
    clf, reg, vec = load_models()
    text = " ".join([x for x in [item.title, item.script, item.hashtags] if x])
    coef = clf.coef_.ravel()
    return token_contributions(text, vec, coef, topk=10)
