
# TikTok Viral Content Predictor (Beauty Vertical)

Predict TikTok virality with ML. Input title/script/hashtags; get a 0-100 viral score, probability, and estimated engagement rate (ER) with keyword-level explainability.

Highlights
- Streamlit demo UI + FastAPI scoring API
- One-click data replace & retrain (scripts/one_click.py) for CSV or pyktok
- One-click client report generator (scripts/make_report.py) -> Markdown + PDF + Excel
- Deploy: Render / Cloud Run / Streamlit Cloud
- CI via GitHub Actions; model card; MIT license

Quickstart
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app/streamlit_app.py
# or: uvicorn api.main:app --reload
```
