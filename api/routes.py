from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.document_loader import process_and_chunk_pdf
from core.vector_store import add_to_vector_store, reset_vector_store

# Initialize the router
router = APIRouter()

from enum import Enum

class SecurityLevel(str, Enum):
    public = "public"
    confidential = "confidential"

    
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    security_level: SecurityLevel = Form(..., description="Must be 'public' or 'confidential'"),
    user_id: str = Form(..., description="Unique ID for the user uploading the file (Or enter your name as user_id)", examples=[""])
):
    
    if not user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="user_id is required"
    )
    """
    Ingestion Endpoint: Uploads a PDF, chunks it, and saves to Vector DB.
    """
    # 1. process and chunk the PDF, injecting the security level metadata
    chunks = await process_and_chunk_pdf(file, security_level, user_id)
    
    # 2. Save the embedded chunks to ChromaDB
    add_to_vector_store(chunks)
    
    return {
        "filename": file.filename,
        "security_level": security_level,
        "user_id": user_id,
        "chunks_saved": len(chunks),
        "message": "File successfully ingested and embedded into ChromaDB!"
    }


from core.rag_chain import process_query
   
@router.get("/query")
async def query_document(question: str, user_id: str):
    """
    Query Endpoint: Ask questions against the uploaded documents.
    """
    result = process_query(question, user_id)
    
    return {
        "question": question,
        "answer": result["answer"],
        "user_id": user_id,
        "security_routed_locally": result["routed_to_local"],
        "sources_used": result.get("sources", [])
    }
    
# delete endpoint for the admin to directly delete the data from local chromaDB
# @router.delete("/reset-database")
# async def reset_database():
#     """
#     Admin Endpoint: Wipes the entire vector database clean.
#     """
#     success = reset_vector_store()
    
#     if success:
#         return {
#             "status": "success",
#             "message": "Database successfully wiped. All previous documents have been deleted."
#         }
#     else:
#         return {
#             "status": "skipped",
#             "message": "Database is already empty or does not exist."
#         }