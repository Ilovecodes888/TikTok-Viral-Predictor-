#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-click pipeline (CSV or pyktok) -> normalize -> train -> optional start."""
import argparse, pathlib, sys, subprocess, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/'data'; MODELS_DIR = ROOT/'models'
DATA_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)
def load_csv(p): return pd.read_csv(p)
def load_pyktok(user, limit=300):
    import pyktok as pyk
    posts = pyk.scrape_posts(user, count=limit); return pd.DataFrame(posts)
def map_cols(df):
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in lower: return lower[n.lower()]
        return None
    t = pick("title","video_title","caption","description")
    s = pick("script","body","text")
    h = pick("hashtags","tags")
    v = pick("views","play","play_count","view_count","plays")
    lk = pick("likes","digg_count","heart","like_count")
    cm = pick("comments","comment_count")
    sh = pick("shares","share_count","reposts")
    out = pd.DataFrame({
        "title": df[t] if t else "",
        "script": df[s] if s else "",
        "hashtags": df[h] if h else "",
        "views": df[v] if v else 0,
        "likes": df[lk] if lk else 0,
        "comments": df[cm] if cm else 0,
        "shares": df[sh] if sh else 0,
    })
    return out
def train():
    r = subprocess.run([sys.executable, "-m", "src.train"], cwd=ROOT)
    if r.returncode != 0: raise SystemExit(r.returncode)
def start(t):
    if t=="streamlit": subprocess.run(["streamlit","run","app/streamlit_app.py"], cwd=ROOT)
    elif t=="api": subprocess.run(["uvicorn","api.main:app","--reload"], cwd=ROOT)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv","pyktok"], required=True)
    ap.add_argument("--csv_path"); ap.add_argument("--user"); ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--start", choices=["streamlit","api","none"], default="none")
    args = ap.parse_args()
    raw = load_csv((ROOT/args.csv_path).resolve()) if args.source=="csv" else load_pyktok(args.user, args.limit)
    mapped = map_cols(raw); mapped.to_csv(DATA_DIR/'sample_dataset.csv', index=False)
    print("[OK] normalized CSV -> data/sample_dataset.csv", len(mapped))
    train()
    if args.start!="none": start(args.start)
