from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("GROQ_API_KEY")

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
        return ChatGroq(api_key=api_key, model="openai/gpt-oss-120b")
        