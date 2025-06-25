# DocSense: RAG PDF Chatbot

DocSense is a Retrieval-Augmented Generation (RAG) chatbot that allows you to upload a PDF and ask questions about its content. It leverages Google Gemini and FAISS for document understanding and semantic search.

## Features

- Upload a PDF and process its content
- Chunk and embed PDF text using Google Gemini embeddings
- Store and search document chunks with FAISS vector store
- Ask questions about your PDF and get intelligent answers

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for all dependencies

## Setup

1. **Clone the repository**

   ```sh
   git clone https://github.com/gamer00001/doc_sense.git
   cd doc_sense
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root (see the provided example):

   ```
   MODEL_NAME=
   GEMINI_API_KEY=your-gemini-api-key
   ```

   - You only need the `GEMINI_API_KEY` for Google Gemini features.

4. **Run the app**

   ```sh
   streamlit run app.py
   ```

5. **Usage**

   - Upload a PDF file using the web interface.
   - Wait for processing to complete.
   - Ask questions about the PDF content in the input box.

## Project Structure

```
.
├── app.py                # Streamlit app entry point
├── rag_pipeline.py       # RAG pipeline logic (loading, chunking, embedding, retrieval)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not committed)
├── data/                 # Uploaded PDFs
├── faiss_index/          # FAISS vector store files
```

## Notes

- Make sure your Gemini API key has sufficient quota.
- Only PDF files are supported.
- The FAISS index is stored in the `faiss_index/` directory.

## License

MIT License

---

Built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), and [Google Gemini](https://ai.google.dev/).