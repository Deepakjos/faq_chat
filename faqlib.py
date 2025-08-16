import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------- Utilities ----------------------

def clean_text_encoding(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    replacements = {"’": "'", "‘": "'", "“": '"', "”": '"', "₹": "Rs ", "–": "-", "—": "-", "…": "..."}
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def expand_with_synonyms(text, synonym_dict):
    text = clean_text_encoding(text)
    tokens = text.lower().split()
    expanded = list(tokens)
    for t in tokens:
        if t in synonym_dict:
            expanded.extend(synonym_dict[t])
    # Deduplicate but keep order-stable
    seen = set()
    dedup = []
    for w in expanded:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return " ".join(dedup)

def preprocess(text, lemmatizer, stopwords_set):
    text = clean_text_encoding(text)
    toks = re.findall(r"\b\w+\b", text.lower())
    toks = [lemmatizer.lemmatize(w) for w in toks if w not in stopwords_set]
    return " ".join(toks)

def normalize_array(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-12:
        return arr  # Return as-is to show no variation
    return (arr - min_v) / (max_v - min_v)

def process_faq_data(df, synonym_dict, lemmatizer, stopwords_set):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(clean_text_encoding)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns.")
    df["fulltext"] = df.apply(
        lambda r: expand_with_synonyms(r["question"], synonym_dict) + " " +
                  expand_with_synonyms(r["answer"], synonym_dict),
        axis=1
    )
    faq_texts = df["fulltext"].apply(lambda x: preprocess(x, lemmatizer, stopwords_set)).tolist()
    return df, faq_texts

def initialize_search_engines(faq_texts, model):
    vectorizer = TfidfVectorizer()
    faq_vecs = vectorizer.fit_transform(faq_texts)
    faq_embed = model.encode(faq_texts, show_progress_bar=False)
    faq_embed = np.asarray(faq_embed, dtype=float)
    return vectorizer, faq_vecs, faq_embed

def sort_top_k(scores, k, threshold=0.0):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    mask = scores > float(threshold)
    if not np.any(mask):
        return np.array([], dtype=int), np.array([], dtype=float)
    filt_idx = np.where(mask)[0]
    filt_scores = scores[filt_idx]
    order = np.argsort(filt_scores)[::-1]
    top = order[:k]
    return filt_idx[top], filt_scores[top]

def tfidf_retrieve(vectorizer, faq_vecs, query_text):
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, faq_vecs).reshape(-1)
    return sims

def semantic_retrieve(model, faq_embed, query_text, lemmatizer, stopwords_set):
    q_proc = preprocess(query_text, lemmatizer, stopwords_set)
    if not q_proc:
        return np.zeros(faq_embed.shape[0], dtype=float)
    q_emb = model.encode([q_proc])
    q_emb = np.asarray(q_emb, dtype=float).reshape(1, -1)
    fe = faq_embed.astype(float)
    fe_norm = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-12)
    qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(fe_norm, qn.T).reshape(-1)
    return sims

def build_results_tuple(alg_name, df, idx_array, score_array):
    idx_array = np.asarray(idx_array, dtype=int).reshape(-1)
    score_array = np.asarray(score_array, dtype=float).reshape(-1)
    out = []
    for i, sc in zip(idx_array, score_array):
        if 0 <= int(i) < len(df):
            out.append((alg_name, df.iloc[int(i)], float(sc), int(i)))
    return out

def search_faqs(query, df, faq_texts, vectorizer, faq_vecs, faq_embed, model,
                lemmatizer, stopwords_set, synonym_dict, top_k=5, threshold=0.1):
    q_pre = preprocess(expand_with_synonyms(query, synonym_dict), lemmatizer, stopwords_set)
    results_by_alg = {}

    try:
        tfidf_sims = tfidf_retrieve(vectorizer, faq_vecs, q_pre)
        tfidf_idx, tfidf_sc = sort_top_k(tfidf_sims, k=top_k, threshold=threshold)
        results_by_alg["TF-IDF"] = build_results_tuple("TF-IDF", df, tfidf_idx, tfidf_sc)
    except Exception:
        results_by_alg["TF-IDF"] = []

    try:
        sem_sims = semantic_retrieve(model, faq_embed, query, lemmatizer, stopwords_set)
        sem_idx, sem_sc = sort_top_k(sem_sims, k=top_k, threshold=threshold)
        results_by_alg["Semantic"] = build_results_tuple("Semantic", df, sem_idx, sem_sc)
    except Exception:
        results_by_alg["Semantic"] = []

    weights = {"Semantic": 1.0, "TF-IDF": 0.7}
    all_ranked = []
    for alg, items in results_by_alg.items():
        w = weights.get(alg, 0.5)
        for (alg_name, row, sc, idx) in items:
            all_ranked.append((alg_name, row, float(sc) * w, idx, float(sc)))
    all_ranked.sort(key=lambda x: x[2], reverse=True)
    return results_by_alg, all_ranked

# ---------------------- Streamlit App ----------------------

def main():
    st.title("🔍 FAQ Semantic Search System")
    st.markdown("Upload an FAQ CSV and search using TF-IDF and Semantic methods. Scores are normalized and ranks are aggregated safely.")
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words("english"))

    st.sidebar.title("⚙️ Configuration")
    top_k = st.sidebar.slider("Results per algorithm", 1, 10, 5)
    threshold = st.sidebar.slider("Minimum score threshold", 0.0, 1.0, 0.1, 0.05)

    synonym_dict = {
        "exempted": ["exemption", "excluded", "exclusions", "exempt"],
        "excluded": ["exclusion", "exempted", "exemptions", "exempt"],
        "exclusion": ["exclusions", "exempted", "exemption", "exempt"],
        "exemption": ["exemptions", "excluded", "exclusion", "exempt"],
        "mutual": ["funds", "mf"],
        "prohibited": ["restrict", "violate", "violation", "not", "permitted", "breaches", "banned"],
        "policy": ["policies", "rule", "rules", "regulation", "regulations"],
        "investment": ["invest", "investing", "investments"],
        "exposure": ["credit", "lending", "loan", "facility"],
        "account": ["accounts", "banking"],
        "rbi": ["reserve", "bank", "india", "central", "bank"],
    }
    uploaded = st.file_uploader("Upload FAQ CSV", type=["csv"], help="CSV must contain 'question' and 'answer' columns.")
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return
    try:
        df = pd.read_csv(uploaded)
        df, faq_texts = process_faq_data(df, synonym_dict, lemmatizer, stopwords_set)
    except Exception as e:
        st.error(f"Error loading/processing: {e}")
        return

    st.success(f"Loaded {len(df)} FAQ entries.")
    with st.expander("Preview (first 5 rows)"):
        st.dataframe(df.head())

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectorizer, faq_vecs, faq_embed = initialize_search_engines(faq_texts, model)
    except Exception as e:
        st.error(f"Failed to initialize search engines: {e}")
        return

    st.markdown("## 🔎 Search")
    query = st.text_input("Enter your question", placeholder="e.g., are there any exclusion in the circular?")
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                results_by_alg, all_ranked = search_faqs(
                    query=query,
                    df=df,
                    faq_texts=faq_texts,
                    vectorizer=vectorizer,
                    faq_vecs=faq_vecs,
                    faq_embed=faq_embed,
                    model=model,
                    lemmatizer=lemmatizer,
                    stopwords_set=stopwords_set,
                    synonym_dict=synonym_dict,
                    top_k=top_k,
                    threshold=threshold,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                return
        algs = ["TF-IDF", "Semantic"]
        tabs = st.tabs(algs + ["All Results (Ranked)"])
        for i, alg in enumerate(algs):
            with tabs[i]:
                items = results_by_alg.get(alg, [])
                if not items:
                    st.info("No results.")
                else:
                    for (alg_name, row, sc, idx) in items:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"Q: {row['question']}")
                            st.markdown(f"A: {row['answer']}")
                        with col2:
                            st.metric("Score", f"{sc:.4f}")
                        st.divider()
        with tabs[-1]:
            if not all_ranked:
                st.info("No results.")
            else:
                st.caption("Ranking uses normalized scores and algorithm weights (Semantic>TF-IDF).")
                for (alg_name, row, weighted, idx, original) in all_ranked:
                    col1, col2, col3 = st.columns([2.5, 0.7, 1.3])
                    with col1:
                        st.markdown(f"Q: {row['question']}")
                        st.markdown(f"A: {row['answer']}")
                    with col2:
                        st.markdown(f"Alg: {alg_name}")
                    with col3:
                        st.metric("Weighted", f"{weighted:.4f}")
                        st.caption(f"Raw: {original:.4f}")
                    st.divider()

if __name__ == "__main__":
    main()
