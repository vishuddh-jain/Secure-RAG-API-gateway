import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Define where Chroma will save its local database files
CHROMA_PATH = "chroma_db"

# Initialize the local embedding model using Ollama
def get_embedding_model():
    # This uses the nomic-embed-text model you just pulled
    return OllamaEmbeddings(model="nomic-embed-text")

def get_vector_store():
    """
    Initializes and returns the Chroma vector database.
    """
    embedding_model = get_embedding_model()
    
    # Create the directory if it doesn't exist
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_model
    )
    return db

def add_to_vector_store(chunks):
    """
    Takes document chunks and saves them to ChromaDB.
    """
    db = get_vector_store()
    
    # This automatically embeds the text and saves it to the local folder
    db.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
# deleting local database folder if admin wants to, after user uses API 
def reset_vector_store():
    """
    Deletes the ChromaDB directory to wipe all saved documents.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"🚨 Database at {CHROMA_PATH} has been completely wiped.")
        return True
    return False