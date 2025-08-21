
from __future__ import annotations
import pathlib, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
import joblib
from .features import TextPrep

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'sample_dataset.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True, parents=True)

CONFIG = {'RANDOM_STATE': 42, 'viral_percentile': 0.75, 'max_features': 3000, 'ngram_range': (1,2), 'min_df': 2}

def load_data():
    df = pd.read_csv(DATA)
    df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']).astype(float) / (df['views'].replace(0, np.nan))
    df['engagement_rate'] = df['engagement_rate'].fillna(0.0).clip(0, 1.0)
    return df

def build(df):
    prep = TextPrep()
    F = prep.transform(df)
    vec = TfidfVectorizer(max_features=CONFIG['max_features'], ngram_range=CONFIG['ngram_range'], min_df=CONFIG['min_df'])
    X_text = vec.fit_transform(F['text'])
    X_num = F.drop(columns=['text']).to_numpy()
    from scipy.sparse import hstack
    X = hstack([X_text, X_num])
    return X, vec

def main():
    df = load_data()
    thr = df['engagement_rate'].quantile(CONFIG['viral_percentile'])
    df['viral'] = (df['engagement_rate'] > thr).astype(int)

    X, vec = build(df)
    y_cls = df['viral'].values
    y_reg = df['engagement_rate'].values

    X_tr, X_te, y_ct, y_ce, y_rt, y_re = train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=CONFIG['RANDOM_STATE'], stratify=y_cls)

    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    clf.fit(X_tr, y_ct)
    auc = roc_auc_score(y_ce, clf.predict_proba(X_te)[:,1])

    reg = SGDRegressor(loss='squared_error', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=CONFIG['RANDOM_STATE'])
    reg.fit(X_tr, y_rt)
    er_pred = reg.predict(X_te)
    mae = mean_absolute_error(y_re, er_pred); r2 = r2_score(y_re, er_pred)

    joblib.dump(clf, MODELS/'clf_logreg.joblib')
    joblib.dump(reg, MODELS/'reg_sgd.joblib')
    joblib.dump(vec, MODELS/'tfidf.joblib')

    report = {'auc': float(auc), 'mae': float(mae), 'r2': float(r2), 'viral_threshold': float(thr)}
    (MODELS/'metrics.json').write_text(json.dumps(report, indent=2))
    print("Metrics:", json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
