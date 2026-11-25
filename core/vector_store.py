import os
import shutil
from functools import lru_cache
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


CHROMA_PATH = str(
    Path(__file__).resolve().parents[1] / "chroma_db"
)

COLLECTION_NAME = "secure_rag_documents"


@lru_cache(maxsize=1)
def get_embedding_model():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


@lru_cache(maxsize=1)
def get_vector_store():

    os.makedirs(CHROMA_PATH, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_model()
    )


def find_existing_document(user_id, document_hash):
    print("DEBUG: Checking existing document...")
    print("DEBUG: User ID:", user_id)
    print("DEBUG: Hash:", document_hash)
    
    db = get_vector_store()

    result = db.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"document_hash": document_hash}
            ]
        },
        include=["documents","metadatas"]
    )

    print("DEBUG: Existing chunks found:", len(result["ids"]))
    
    return result


def update_document_security(
    user_id,
    document_hash,
    security_level
):

    print("DEBUG: Starting security metadata update...")
    
    db = get_vector_store()

    result = find_existing_document(
        user_id,
        document_hash
    )

    ids = result["ids"]
    metadatas = result["metadatas"]
    documents = result["documents"]

    print("DEBUG: IDs found:", len(ids))
    
    if not ids:
        return False

    # Preserve all existing metadata.
    updated_documents = []

    for content, metadata in zip(documents, metadatas):

        updated_metadata = dict(metadata)

        updated_metadata["security_level"] = security_level

        updated_documents.append(
            Document(
                page_content=content,
                metadata=updated_metadata
            )
        )

    print("DEBUG: Updating ChromaDB metadata...")
    
    # Update metadata only.
    # No new embedding generation is requested.
    db.update_documents(
        ids=ids,
        documents=updated_documents
    )

    print("DEBUG: Metadata updated successfully!")

    return True


def add_to_vector_store(chunks, batch_size=16):

    if not chunks:
        return

    db = get_vector_store()

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        db.add_documents(batch)

        print(
            f"Embedded and saved "
            f"{min(start + batch_size, len(chunks))}/{len(chunks)} chunks"
        )


def reset_vector_store():

    get_vector_store.cache_clear()
    get_embedding_model.cache_clear()

    if os.path.exists(CHROMA_PATH):

        shutil.rmtree(CHROMA_PATH)

        print("ChromaDB database deleted.")

        return True

    return False


'''import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Define where Chroma will save its local database files
CHROMA_PATH = "chroma_db"

# Initialize the local embedding model using Ollama
def get_embedding_model():
    # this uses the nomic-embed-text model I just pulled
    return OllamaEmbeddings(model="nomic-embed-text")

def get_vector_store():
    """
    Initializes and returns the Chroma vector database
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

def add_to_vector_store(chunks, batch_size=50):
    """
    Slices the document chunks into manageable batches and embeds them 
    using matrix multiplication to optimize CPU latency.
    """
    db = get_vector_store()
    total_chunks = len(chunks)
    
    print(f"Total chunks to process: {total_chunks}")
    
    # Loop through the chunks in steps of 'batch_size'
    for i in range(0, total_chunks, batch_size):
        # Slice the array to create a batch (e.g., chunks 0-50, then 50-100)
        batch = chunks[i : i + batch_size]
        
        # Add only this specific batch to ChromaDB
        db.add_documents(batch)
        
        print(f"Successfully embedded batch {i // batch_size + 1} (Chunks {i} to {i + len(batch)})")
    
# deleting local database folder if admin wants to, after user uses API 
def reset_vector_store():
    """
    Deletes the ChromaDB directory to wipe all saved documents.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Database at {CHROMA_PATH} has been completely wiped.")
        return True
    return False
'''