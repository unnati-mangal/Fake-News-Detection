import re
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


# Must match the cleaning used during training
def clean_text(text):
    text = str(text)
    text = re.sub(r'^.{0,120}?\(reuters\)\s*-?\s*', '', text, flags=re.I)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_pipeline():
    return joblib.load("fake_news_pipeline.pkl")


def predict(pipe, text):
    cleaned = clean_text(text)
    label = pipe.predict([cleaned])[0]
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        conf = float(pipe.predict_proba([cleaned])[0].max())
    else:
        conf = float(1 / (1 + np.exp(-abs(pipe.decision_function([cleaned])[0]))))
    return int(label), conf, cleaned


def top_words(pipe, cleaned, n=8):
    """Words in this article that pushed the prediction most (linear models only)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return [], []
    tfidf = pipe.named_steps["tfidf"]
    vec = tfidf.transform([cleaned])
    feats = np.array(tfidf.get_feature_names_out())
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return [], []
    contrib = vec[0, idx].toarray().ravel() * clf.coef_.ravel()[idx]
    order = np.argsort(contrib)
    fake_words = [feats[idx[i]] for i in order[:n] if contrib[i] < 0]
    real_words = [feats[idx[i]] for i in order[::-1][:n] if contrib[i] > 0]
    return fake_words, real_words


st.title("📰 Fake News Detector")
st.caption("TF-IDF + linear classifier trained on the Fake/True news dataset")

try:
    pipe = load_pipeline()
except FileNotFoundError:
    st.error("`fake_news_pipeline.pkl` not found. Run the notebook first and put the file next to this app.py.")
    st.stop()

examples = {
    "Choose an example...": "",
    "Example 1 (policy news)": "The Senate voted on Tuesday to approve a spending bill that funds federal agencies through the end of the fiscal year, after weeks of negotiations between party leaders.",
    "Example 2 (sensational)": "BREAKING: You won't believe what this politician was caught doing! The mainstream media is hiding the shocking truth from you, share before they delete this!",
}
choice = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area("Paste a news headline or article", value=examples[choice], height=220)

if st.button("Analyze", type="primary"):
    if len(text.split()) < 8:
        st.warning("Please enter at least a few sentences for a reliable result.")
    else:
        label, conf, cleaned = predict(pipe, text)
        if label == 1:
            st.success(f"✅ Looks REAL  (confidence {conf:.0%})")
        else:
            st.error(f"🚩 Looks FAKE  (confidence {conf:.0%})")
        st.progress(min(conf, 1.0))

        fake_w, real_w = top_words(pipe, cleaned)
        if fake_w or real_w:
            with st.expander("Why? Words that influenced the result"):
                c1, c2 = st.columns(2)
                c1.markdown("**Pushed toward FAKE**")
                c1.write(", ".join(fake_w) or "-")
                c2.markdown("**Pushed toward REAL**")
                c2.write(", ".join(real_w) or "-")

st.divider()
st.caption(
    "Limitation: trained on US political news from 2016-2017, so results on other topics "
    "or years may be unreliable. This is a classroom project, not a fact-checker."
)
