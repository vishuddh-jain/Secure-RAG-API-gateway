from core.vector_store import get_vector_store
from core.llm_router import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def get_bm25_retriever(user_id: str, index_version: int):

    db = get_vector_store()

    collection_data = db.get(
        where={"user_id": user_id}
    )

    user_docs = [
        Document(
            page_content=doc,
            metadata=meta
        )
        for doc, meta in zip(
            collection_data["documents"],
            collection_data["metadatas"]
        )
    ]

    if not user_docs:
        return None

    retriever = BM25Retriever.from_documents(user_docs)
    retriever.k = 3

    return retriever


def process_query(question: str, user_id: str):

    total_start = time.perf_counter()

    # 1. Get vector store
    start = time.perf_counter()

    db = get_vector_store()

    print(
        f"Vector store initialization/access: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # 2. Vector retriever
    start = time.perf_counter()

    vector_retriever = db.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"user_id": user_id}
        }
    )

    # 3. Fetch user documents and build BM25
    collection_data = db.get(
        where={"user_id": user_id}
    )

    if not collection_data["documents"]:
        return {
            "answer": "No uploaded documents found.",
            "routed_to_local": False,
            "sources": [],
            "contexts": []
        }

    user_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(
            collection_data["documents"],
            collection_data["metadatas"]
        )
    ]

    bm25_retriever = BM25Retriever.from_documents(user_docs)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    print(
        f"Retriever and BM25 preparation: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # 4. Hybrid retrieval
    start = time.perf_counter()

    docs = ensemble_retriever.invoke(question)

    print(
        f"Hybrid retrieval: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # 5. Security routing
    is_confidential = any(
        doc.metadata.get("security_level") == "confidential"
        for doc in docs
    )

    llm = get_llm(is_confidential)

    # 6. Prepare context and prompt
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    
    prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant answering questions
    based on the provided document context.

    Follow these response formatting rules strictly:

    1. Write answers in clear, natural and simple English.
    2. Use normal paragraphs instead of Markdown tables.
    3. Do not use Markdown tables, code blocks or JSON.
    4. Avoid unnecessary headings and bold formatting.
    5. Keep answers concise but informative.
    6. Use bullet points or numbered lists only when
       the question specifically requires multiple steps
       or a list of items.
    7. Answer directly without unnecessary introductions.
    8. Do not repeat the question in your answer.
    9. Do not introduce information that is not supported
       by the provided context.
    10. If the answer is not available in the context,
        clearly say that the information is not available.

    Important:
    Your response must be plain, readable text.
    Do not format the answer like a technical report
    unless the user explicitly requests that format.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )


    chain = prompt | llm

    # 7. LLM generation
    start = time.perf_counter()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print(
        f"LLM generation: "
        f"{time.perf_counter() - start:.2f}s"
    )

    print(
        f"TOTAL QUERY TIME: "
        f"{time.perf_counter() - total_start:.2f}s"
    )

    return {
        "answer": response.content,
        "routed_to_local": is_confidential,
        "sources": [doc.metadata for doc in docs],
        "contexts": [doc.page_content for doc in docs]
    }
    
"""
from core.vector_store import get_vector_store
from core.llm_router import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def process_query(question: str, user_id: str):
    db = get_vector_store()
    
    vector_retriever = db.as_retriever(
            search_kwargs={"k": 3, "filter": {"user_id": user_id}}
        )
        
        # 2. Get the BM25 Retriever (Keyword Search)
        # We dynamically fetch only THIS user's documents to build a secure keyword index
    collection_data = db.get(where={"user_id": user_id})
        
    if not collection_data["documents"]:
        return {"answer": "I don't have any uploaded documents for your user profile that answer this.", "routed_to_local": False}
            
    user_docs = [
        Document(page_content=doc, metadata=meta) 
        for doc, meta in zip(collection_data["documents"], collection_data["metadatas"])
        ]
        
    bm25_retriever = BM25Retriever.from_documents(user_docs)
    bm25_retriever.k = 3 # Retrieve top 3 keyword matches
        
    # 3. Fuse them together into an Ensemble Retriever
    # Weights are 50% Vector / 50% Keyword
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5] 
        )
        
        # 4. Execute the Hybrid Search!
    docs = ensemble_retriever.invoke(question)
        
    # 4. Check the security clearance of the retrieved chunks
    # If even ONE chunk is confidential, the whole query becomes confidential
    is_confidential = any(doc.metadata.get("security_level") == "confidential" for doc in docs)
    
    # 5. Route to the correct LLM
    llm = get_llm(is_confidential)
    
    # 6. Prepare the Prompt
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    
    # 7. Generate the answer (Langchain Expression Language)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": response.content,
        "routed_to_local": is_confidential,
        "sources": [doc.metadata for doc in docs]
    }
"""
