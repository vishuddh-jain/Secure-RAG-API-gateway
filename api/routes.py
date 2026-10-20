from fastapi import APIRouter, UploadFile, File, Form

# Initialize the router
router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    security_level: str = Form(..., description="Must be 'public' or 'confidential'")
):
    """
    Ingestion Endpoint: Uploads a PDF and categorizes its sensitivity.
    """
    # TODO: Add Langchain parsing, chunking, and ChromaDB logic here later
    
    return {
        "filename": file.filename,
        "security_level": security_level,
        "message": "File received successfully! (AI processing coming soon)"
    }

@router.get("/query")
async def query_document(question: str):
    """
    Query Endpoint: Ask questions against the uploaded documents.
    """
    # TODO: Add Langchain retrieval and Ollama/Cloud LLM routing logic here later
    
    return {
        "question": question,
        "answer": "This is a mock answer. The RAG pipeline and Hybrid LLM Router are not connected yet!"
    }