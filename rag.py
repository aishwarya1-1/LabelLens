from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


import pathlib
from dotenv import load_dotenv


load_dotenv()

# Load and process PDF
file_path = "/Users/aishwaryakalburgi/Downloads/cancer.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create FAISS vector store and save it
vector_store = FAISS.from_documents(pages, embeddings)
vector_store.save_local("faiss_index")

print("FAISS vector store created and saved.")