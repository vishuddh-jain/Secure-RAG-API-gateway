from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

async def process_and_chunk_pdf(upload_file, security_level):
    """
    Takes a FastAPI UploadFile, reads it, adds the security tag, 
    and splits it into smaller chunks for the Vector DB.
    """
    # 1. Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await upload_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # 2. Load the PDF using Langchain
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 3. Add the security clearance metadata to every page
    for doc in documents:
        doc.metadata["security_level"] = security_level

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
