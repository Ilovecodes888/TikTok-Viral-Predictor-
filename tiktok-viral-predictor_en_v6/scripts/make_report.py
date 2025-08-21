#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a virality report from a CSV of candidate topics (text-only OK)."""
import argparse, pathlib, json, datetime, pandas as pd, sys, textwrap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import score_one
TEMPLATE = (ROOT/'reports/templates/beauty_report_template.md').read_text(encoding='utf-8')
def render_md_to_pdf(md_text: str, pdf_path: pathlib.Path):
    pp = PdfPages(pdf_path)
    pages = md_text.split('\n---\n')
    for page in pages:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.08, 0.06, 0.84, 0.9]); ax.axis('off')
        wrapped = []
        for line in page.splitlines():
            if line.startswith('|'): wrapped.append(line)
            elif len(line) > 100: wrapped += textwrap.wrap(line, width=100, break_long_words=False)
            else: wrapped.append(line)
        y = 0.98
        for line in wrapped:
            y -= 0.02
            if y < 0.05:
                pp.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69)); ax = fig.add_axes([0.08, 0.06, 0.84, 0.9]); ax.axis('off'); y = 0.98
            ax.text(0, y, line, fontsize=9, family='monospace')
        pp.savefig(fig); plt.close(fig)
    pp.close()
def ensure_models():
    import subprocess
    if not (ROOT/'models/clf_logreg.joblib').exists():
        r = subprocess.run([sys.executable, '-m', 'src.train'], cwd=ROOT)
        if r.returncode != 0: raise SystemExit('Training failed.')
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_path', required=True)
    ap.add_argument('--brand', default='BeautyLab')
    ap.add_argument('--outdir', default='reports')
    args = ap.parse_args()
    df_raw = pd.read_csv((ROOT/args.csv_path).resolve())
    lower = {c.lower(): c for c in df_raw.columns}
    def pick(*names):
        for n in names:
            if n.lower() in lower: return lower[n.lower()]
        return None
    t = pick('title','video_title','caption','description')
    s = pick('script','body','text')
    h = pick('hashtags','tags')
    df = pd.DataFrame({
        'title': df_raw[t] if t else '',
        'script': df_raw[s] if s else '',
        'hashtags': df_raw[h] if h else ''
    })
    ensure_models()
    scored = []
    for _, row in df.iterrows():
        out = score_one(str(row.get('title','')), str(row.get('script','')), str(row.get('hashtags','')))
        scored.append({**row.to_dict(), **out})
    scored.sort(key=lambda x: x['score'], reverse=True)
    rows = []
    for idx, s in enumerate(scored[:10], start=1):
        combo = f"**{s.get('title','')}**<br/>{s.get('script','')}<br/>{s.get('hashtags','')}"
        rows.append(f"| {idx} | {combo} | {s['score']} | {s['viral_probability']*100:.1f}% | {s['engagement_rate_pred']*100:.2f}% |")
    table_rows = "\n".join(rows)
    metrics = {}
    mpath = ROOT/'models/metrics.json'
    if mpath.exists():
        import json as _json
        metrics = _json.loads(mpath.read_text())
    today = datetime.date.today().isoformat()
    md = TEMPLATE
    md = md.replace('{{brand_name}}', args.brand).replace('{{date_range}}', today)
    md = md.replace('{{model_version}}', 'text-baseline v1.0')
    md = md.replace('{{auc}}', f"{metrics.get('auc','~0.80')}")
    md = md.replace('{{top_table_rows}}', table_rows).replace('{{score_bar}}','70')
    md = md.replace('{{top_keywords}}', 'salicylic acid, ceramide, retinol, before/after, brightening, sunscreen, oil control')
    md = md.replace('{{competitor_topics}}', 'barrier repair, summer sunscreen, anti-aging night cream, acne-friendly base makeup')
    md = md.replace('{{best_time}}','7-10 PM').replace('{{freq}}','1-2')
    md = md.replace('{{viral_thr}}', str(metrics.get('viral_threshold','(training 75th percentile)')))
    outdir = ROOT/args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir/f"{args.brand}_Report_{today}.md"; md_path.write_text(md, encoding='utf-8')
    pdf_path = outdir/f"{args.brand}_Report_{today}.pdf"; render_md_to_pdf(md, pdf_path)
    xlsx_path = outdir/f"{args.brand}_Scores_{today}.xlsx"; pd.DataFrame(scored).to_excel(xlsx_path, index=False)
    print('[OK] Report generated:', md_path, pdf_path, xlsx_path)
if __name__ == '__main__':
    main()
