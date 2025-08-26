import streamlit as st
import pandas as pd
import os
from faqlib import (
    load_model, get_nltk_resources, process_faq_data, 
    build_search_engines, search_faqs, get_default_synonyms
)
from flowcharts import current_account_flow
#no changes
def get_available_faq_files():
    """Get available FAQ files from faqs folder"""
    faq_folder = "faqs"
    os.makedirs(faq_folder, exist_ok=True)
    files = [f for f in os.listdir(faq_folder) if f.endswith('.csv')]
    display = [os.path.splitext(f)[0].replace("_", " ").title() for f in files]
    return files, display

def load_selected_faq(selected_file):
    """Load a selected FAQ file"""
    return pd.read_csv(os.path.join("faqs", selected_file))

def faq_search_tab():
    """FAQ search tab UI"""
    # Load cached resources
    model = load_model()
    lemmatizer, stopwords_set = get_nltk_resources()
    synonym_dict = get_default_synonyms()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    top_k = st.sidebar.slider("Results per algorithm", 1, 10, 3)
    threshold = st.sidebar.slider("Minimum score threshold", 0.0, 1.0, 0.1, 0.05)
    
    # File upload/selection
    faq_files, faq_names = get_available_faq_files()
    chosen = st.selectbox("Choose an inbuilt FAQ file", faq_names + ["(None / Use uploaded)"])
    uploaded = st.file_uploader("Or upload your own FAQ CSV", type=["csv"], 
                              help="CSV must contain 'question' and 'answer' columns.")
    
    df = None
    source_info = ""
    
    # Handle file loading
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            source_info = f"Using uploaded file: {uploaded.name}"
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    elif chosen != "(None / Use uploaded)":
        try:
            idx = faq_names.index(chosen)
            df = load_selected_faq(faq_files[idx])
            source_info = f"Using inbuilt FAQ: {faq_files[idx]}"
        except Exception as e:
            st.error(f"Error reading inbuilt file: {e}")
            return
    else:
        st.info("Please upload a file or select a built-in FAQ from the dropdown.")
        return
    
    # Process data
    try:
        df, faq_texts = process_faq_data(df, synonym_dict, lemmatizer, stopwords_set)
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return
    
    st.success(f"Loaded {len(df)} FAQ entries. {source_info}")
    
    with st.expander("Preview (first few rows)"):
        st.dataframe(df.head(20))
    
    # Build search engines
    try:
        vectorizer, faq_vecs, faq_embed = build_search_engines(model, faq_texts)
    except Exception as e:
        st.error(f"Failed to initialize search engines: {e}")
        return
    
    # Search interface
    st.markdown("## 🔎 Search")
    query = st.text_input("Enter your question", 
                         placeholder="e.g., are there any exclusion in the circular?")
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                results_by_alg, all_ranked = search_faqs(
                    query=query, df=df, faq_texts=faq_texts,
                    vectorizer=vectorizer, faq_vecs=faq_vecs, faq_embed=faq_embed,
                    model=model, lemmatizer=lemmatizer, stopwords_set=stopwords_set,
                    synonym_dict=synonym_dict, top_k=top_k, threshold=threshold
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                return
            
            # Display results in tabs
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
                                st.markdown(f"**Q:** {row['question']}")
                                st.markdown(f"**A:** {row['answer']}")
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
                            st.markdown(f"**Q:** {row['question']}")
                            st.markdown(f"**A:** {row['answer']}")
                        with col2:
                            st.markdown(f"**Alg:** {alg_name}")
                        with col3:
                            st.metric("Weighted", f"{weighted:.4f}")
                            st.caption(f"Raw: {original:.4f}")
                        st.divider()

def main():
    st.set_page_config(page_title="FAQ Semantic Search", layout="wide")
    st.title("🔍 FAQ Semantic Search & Flowchart App")
    
    tab_labels = ["FAQ Search", "Current Account Flowchart"]
    tabs = st.tabs(tab_labels)
    
    with tabs[0]:
        faq_search_tab()
    
    with tabs[1]:
        current_account_flow()

if __name__ == "__main__":
    main()






