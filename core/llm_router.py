
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


def get_llm(is_confidential: bool):

    # Fail-safe: only explicitly public data may use cloud.
    if is_confidential is not False:
        print("Routing to Local Ollama...")
        return ChatOllama(
            model="gemma2:2b",
            keep_alive="5m",
            temperature=0.1,
            num_predict=256
            )

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    print("Routing to Cloud Groq...")
    return ChatGroq(
        api_key=api_key,
        max_tokens=250,
        model="openai/gpt-oss-120b",
        temperature=0
    )

"""
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("GROQ_API_KEY")

def get_llm(is_confidential: bool):
    
    # Routes the prompt based on the security level of the retrieved data.
    
    if is_confidential:
        print("CONFIDENTIAL DATA DETECTED: Routing to Local Ollama...")
        # Zero data leaves the machine
        return ChatOllama(model="gemma2:2b")
    else:
        print("PUBLIC DATA DETECTED: Routing to Cloud API...")
        return ChatGroq(api_key=api_key, model="openai/gpt-oss-120b")
"""
        