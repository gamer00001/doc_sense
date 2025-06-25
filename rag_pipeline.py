import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
FAISS_INDEX_PATH = "faiss_index"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables.")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME not set in environment variables.")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

def load_pdf(file_path):
    """Load a PDF and return documents."""
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def chunk_documents(documents):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    """Build and save a FAISS vector store from document chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def load_vector_store():
    """Load a FAISS vector store from disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(vectorstore):
    """Create a RetrievalQA chain using the vector store."""
    retriever = vectorstore.as_retriever()
    llm = GoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=GEMINI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def run_query(query, chain):
    """Run a query through the QA chain and return the result."""
    return chain.invoke({"query": query})
      