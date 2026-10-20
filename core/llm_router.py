from langchain_community.chat_models import ChatOllama
# from langchain_groq import ChatGroq # (You can uncomment this later when you have an API key)

def get_llm(is_confidential: bool):
    """
    Routes the prompt based on the security level of the retrieved data.
    """
    if is_confidential:
        print("CONFIDENTIAL DATA DETECTED: Routing to Local Ollama...")
        # Zero data leaves the machine
        return ChatOllama(model="gemma2:2b")
    else:
        print("PUBLIC DATA DETECTED: Routing to Cloud API...")
        # TODO: Replace with ChatOpenAI or ChatGroq when you get an API key!
        # return ChatGroq(api_key="your_api_key_here", model="gpt-3.5-turbo")
        
        # For testing right now, we will just use Ollama for both, 
        # but in your final project, this is where the Cloud LLM goes!
        return ChatOllama(model="gemma2:2b")