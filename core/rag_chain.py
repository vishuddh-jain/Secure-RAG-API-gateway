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