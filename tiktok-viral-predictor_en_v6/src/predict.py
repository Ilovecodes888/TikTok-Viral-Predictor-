
from __future__ import annotations
import joblib, pathlib, pandas as pd
from .features import TextPrep
from scipy.sparse import hstack

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'

def load_models():
    clf = joblib.load(MODELS/'clf_logreg.joblib')
    reg = joblib.load(MODELS/'reg_sgd.joblib')
    vec = joblib.load(MODELS/'tfidf.joblib')
    return clf, reg, vec

def make_X(df: pd.DataFrame, vec):
    prep = TextPrep(); F = prep.transform(df)
    X_text = vec.transform(F['text']); X_num = F.drop(columns=['text']).to_numpy()
    return hstack([X_text, X_num])

def score_one(title: str, script: str, hashtags: str):
    df = pd.DataFrame([{'title': title, 'script': script, 'hashtags': hashtags}])
    clf, reg, vec = load_models()
    X = make_X(df, vec)
    viral_prob = float(clf.predict_proba(X)[0,1])
    er_pred = float(reg.predict(X)[0])
    er_cap = max(min(er_pred, 0.5), 0.0)
    score_0_100 = int(round(er_cap / 0.5 * 100))
    return {'viral_probability': viral_prob, 'engagement_rate_pred': er_pred, 'score': score_0_100}
