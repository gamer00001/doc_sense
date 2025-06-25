import streamlit as st
import os
from rag_pipeline import load_pdf, chunk_documents, build_vector_store, load_vector_store, get_qa_chain, run_query

st.title("📄 RAG PDF Chatbot")

# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = os.path.join("data", "uploaded.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully. Processing...")

    try:
        docs = load_pdf(file_path)
        chunks = chunk_documents(docs)
        build_vector_store(chunks)
        st.success("Vector store created! You can now ask questions.")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.stop()

    try:
        vectorstore = load_vector_store()
        chain = get_qa_chain(vectorstore)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.stop()

    query = st.text_input("Ask something about the PDF:")

    if query:
        try:
            response = run_query(query, chain)
            answer = response.get("result", "No answer found.")
            st.write("🧠 Answer:", answer)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                st.error("You have exceeded your Gemini API quota. Please wait and try again later, or check your API usage and billing.")
            else:
                st.error(f"Error running query: {e}")
