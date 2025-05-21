import streamlit as st

st.set_page_config(
    page_title="DocuGenie",
    page_icon="🧞‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DocuGenie 🧞‍♂️")

st.markdown("""
Welcome to **DocuGenie**, your on-demand genie for interactive document Q&A.

**What you can do:**  
1. **Upload** your PDFs in the sidebar.  
2. **Ask** any question about their content.  
3. **Get** precise, grounded answers powered by Retrieval-Augmented Generation (RAG).

DocuGenie makes your documents come alive—no more endless scrolling or manual searches.  
Simply upload, ask, and let your genie handle the rest!
""")
