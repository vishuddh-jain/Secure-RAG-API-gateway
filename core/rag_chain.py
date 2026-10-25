from core.vector_store import get_vector_store
from core.llm_router import get_llm
from langchain_core.prompts import ChatPromptTemplate

def process_query(question: str, user_id: str):
    db = get_vector_store()
    
    # 1. Retrieve the top 3 most relevant chunks from ChromaDB
    docs = db.similarity_search(question, k=5, filter={"user_id" : user_id})
    
    if not docs:
        return {"answer": "I don't have any uploaded documents that answer this.", "routed_to_local": False}
        
    # 2. Check the security clearance of the retrieved chunks
    # If even ONE chunk is confidential, the whole query becomes confidential
    is_confidential = any(doc.metadata.get("security_level") == "confidential" for doc in docs)
    
    # 3. Route to the correct LLM
    llm = get_llm(is_confidential)
    
    # 4. Prepare the Prompt
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    
    # 5. Generate the answer (Langchain Expression Language)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": response.content,
        "routed_to_local": is_confidential,
        "sources": [doc.metadata for doc in docs]
    }