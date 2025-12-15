import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    exit()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def main():
    print("--- 1. Loading Documents ---")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Error: No PDFs found in the 'data' folder.")
        return

    print(f"Loaded {len(documents)} pages.")

    print("--- 2. Splitting Text ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("--- 3. Creating Vector Store (Using Google Gemini) ---")
    
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    print(f"--- Database Created at {CHROMA_PATH} ---")

if __name__ == "__main__":
    main()