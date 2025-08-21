
import streamlit as st
st.set_page_config(page_title="TikTok Viral Content Predictor", page_icon=":sparkles:")
st.title("TikTok Viral Content Predictor — Beauty Vertical")
st.caption("Enter title / script / hashtags to predict viral probability, score, and engagement rate (text-only baseline).")

title = st.text_input("Title", "Pore care fix: how to use 2% salicylic acid")
script = st.text_area("Script", "Visible in 3 days; gentle but effective; barrier-friendly.")
hashtags = st.text_input("Hashtags", "#skincare #salicylicacid #beforeafter")

if st.button("Predict"):
    try:
        from src.predict import score_one, load_models
        from src.explain import token_contributions
        out = score_one(title, script, hashtags)
        st.metric("Viral Score (0-100)", out['score'])
        st.metric("Viral Probability", f"{out['viral_probability']*100:.1f}%")
        st.metric("Predicted ER", f"{out['engagement_rate_pred']*100:.2f}%")
        with st.expander("Explainability — top keyword contributions"):
            clf, reg, vec = load_models()
            coef = clf.coef_.ravel()
            text = " ".join([title, script, hashtags])
            top = token_contributions(text, vec, coef, topk=10)
            st.write("Positive contributors")
            st.write(top["top_positive"])
            st.write("Negative contributors")
            st.write(top["top_negative"])
    except Exception as e:
        st.error(f"Inference failed: {e}. Run `python -m src.train` first.")
