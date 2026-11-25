
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile
import os


def process_and_chunk_document(
    file_content: bytes,
    filename: str,
    security_level: str,
    user_id: str,
    document_hash: str
):
    """
    Extract, split and attach metadata to uploaded documents.
    """

    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension not in [".pdf", ".docx", ".txt"]:
        raise ValueError("Unsupported file format")

    tmp_path = None

    try:
        # 1. Write file content to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as tmp:

            tmp.write(file_content)
            tmp_path = tmp.name

        # 2. Select document loader
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_path)

        elif file_extension == ".docx":
            loader = Docx2txtLoader(tmp_path)

        else:
            loader = TextLoader(
                tmp_path,
                autodetect_encoding=True
            )

        # 3. Extract text
        documents = loader.load()

        # 4. Attach metadata to each page
        for doc in documents:
            doc.metadata["security_level"] = security_level
            doc.metadata["user_id"] = user_id
            doc.metadata["document_hash"] = document_hash

        # 5. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)

        return chunks

    finally:
        # 6. Always clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


"""
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

async def process_and_chunk_pdf(upload_file, security_level, user_id):
    
    # Takes a FastAPI UploadFile, reads it, adds the security tag, 
    # and splits it into smaller chunks for the Vector DB.
    
    # 1. Create a temporary file to save the uploaded PDF
    
    file_extension = os.path.splitext(upload_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # 2. Load the PDF using Langchain
    if file_extension == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(tmp_path)
    elif file_extension == ".txt":
        loader = TextLoader(tmp_path)
    else:
        os.remove(tmp_path)
        raise ValueError(f"Unsupported file format: {file_extension}. Please upload a PDF, DOCX, or TXT file.")
    documents = loader.load()

    # 3. Add the security clearance and user_id metadata to every page
    for doc in documents:
        doc.metadata["security_level"] = security_level
        doc.metadata["user_id"] = user_id

    # 4. Split the text into smaller chunks (RAG best practice)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)

    # 5. Clean up the temporary file
    os.remove(tmp_path)

    return chunks
"""