import hashlib

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.concurrency import run_in_threadpool
from enum import Enum

from core.document_loader import process_and_chunk_document

from core.vector_store import (
    add_to_vector_store,
    find_existing_document,
    update_document_security
)

from core.rag_chain import process_query
import logging
logger = logging.getLogger(__name__)

router = APIRouter()


class SecurityLevel(str, Enum):
    public = "public"
    confidential = "confidential"


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    security_level: SecurityLevel = Form(...,description="Must be Public or Confidential"),
    user_id: str = Form(..., description="Unique ID for the user uploading the file (Or enter your name as user_id)", examples=[""])
):

    if not user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="user_id is required"
        )

    filename = file.filename or ""

    # 1. Read uploaded file
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # 2. Calculate SHA-256 fingerprint
    document_hash = hashlib.sha256(content).hexdigest()

    try:

        # 3. Check whether this document already exists
        existing = await run_in_threadpool(
            find_existing_document,
            user_id,
            document_hash
        )

        if existing["ids"]:

            # 4. Update security metadata only
            await run_in_threadpool(
                update_document_security,
                user_id,
                document_hash,
                security_level.value
            )

            return {
                "filename": filename,
                "user_id": user_id,
                "document_hash": document_hash,
                "security_level": security_level.value,
                "chunks_saved": len(existing["ids"]),
                "duplicate": True,
                "embeddings_generated": False,
                "message": (
                    "Document already exists. "
                    "Security metadata updated. "
                    "No new embeddings generated."
                )
            }

        # 5. Process a new document
        chunks = await run_in_threadpool(
            process_and_chunk_document,
            content,
            filename,
            security_level.value,
            user_id,
            document_hash
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )

        # 6. Generate embeddings and save in batches
        await run_in_threadpool(
            add_to_vector_store,
            chunks
        )

        return {
            "filename": filename,
            "user_id": user_id,
            "document_hash": document_hash,
            "security_level": security_level.value,
            "chunks_saved": len(chunks),
            "duplicate": False,
            "embeddings_generated": True,
            "message": "New document successfully embedded and stored."
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        logger.exception("Document upload failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    finally:
        await file.close()


@router.get("/query")
async def query_document(
    question: str,
    user_id: str
):

    result = await run_in_threadpool(
        process_query,
        question,
        user_id
    )

    return {
        "question": question,
        "answer": result["answer"],
        "user_id": user_id,
        "security_routed_locally": result["routed_to_local"],
        "sources_used": result.get("sources", [])
    }

"""
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
    
    # Ingestion Endpoint: Uploads a PDF, chunks it, and saves to Vector DB.
    
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
    
    # Query Endpoint: Ask questions against the uploaded documents.
    
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
#     
#     # Admin Endpoint: Wipes the entire vector database clean.
#     
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
"""