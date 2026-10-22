from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.document_loader import process_and_chunk_pdf
from core.vector_store import add_to_vector_store

# Initialize the router
router = APIRouter()

from enum import Enum

class SecurityLevel(str, Enum):
    public = "public"
    confidential = "confidential"
    
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    security_level: SecurityLevel = Form(..., description="Must be 'public' or 'confidential'")
):
    """
    Ingestion Endpoint: Uploads a PDF, chunks it, and saves to Vector DB.
    """
    # 1. Process and chunk the PDF, injecting the security level metadata
    chunks = await process_and_chunk_pdf(file, security_level)
    
    # 2. Save the embedded chunks to ChromaDB
    add_to_vector_store(chunks)
    
    if security_level not in ["public", "confidential"]:
        raise HTTPException(
            status_code=400,
            detail="security_level must be 'public' or 'confidential'"
        )
    
    return {
        "filename": file.filename,
        "security_level": security_level,
        "chunks_saved": len(chunks),
        "message": "File successfully ingested and embedded into ChromaDB!"
    }


from core.rag_chain import process_query
   
@router.get("/query")
async def query_document(question: str):
    """
    Query Endpoint: Ask questions against the uploaded documents.
    """
    result = process_query(question)
    
    return {
        "question": question,
        "answer": result["answer"],
        "security_routed_locally": result["routed_to_local"],
        "sources_used": result.get("sources", [])
    }