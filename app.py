"""
    RAG (Retrieval Augmented Generation) system for document Q&A.

    This module implements a RAG system using Streamlit, ChromaDB, and Ollama.
    It provides document processing, vector storage, and question answering capabilities.

"""

from src.rag_app import RAGApplication

import streamlit as st

        
def main():
    """Initialize and run the Streamlit application."""
    app = RAGApplication()
    st.title("RAG Question Answer System")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("**📑 Upload PDF files for Quries** ", type=['pdf'], accept_multiple_files=False)
        if uploaded_file and st.button("⚡️ Process"):
            app.process_upload(uploaded_file)
            st.success("Document processed!")

    prompt = st.text_area("**Ask a question related to your document:**")
    
    # prompt = st.text_input("Ask a question:")
    if prompt and st.button("Get Answer"):
        results, relevant_text, relevant_text_ids, response = app.answer_question(prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents :"):
            st.write(results)

        with st.expander("See most relevant document ids :"):
            st.write(relevant_text_ids)
            st.write(relevant_text)

if __name__ == "__main__":
    main()
