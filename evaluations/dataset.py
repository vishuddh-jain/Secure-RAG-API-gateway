from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
client = Client()

dataset_name = "secure-rag-public-evaluation"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Public document RAG evaluation dataset"
)

examples = [
    {
        "inputs": {
            "question": "How does the Secure RAG API handle standard RAG routing differently?"
        },
        "outputs": {
            "answer": "It introduces a Dynamic Router that inspects chunk metadata to route confidential data to a local on-premise model and public data to a fast Cloud API."
        }
    },
    {
        "inputs": {
            "question": "What local vector database and inference engine does the Secure RAG API utilize to ensure Zero Data Egress?"
        },
        "outputs": {
            "answer": "It utilizes ChromaDB as the local vector database and Ollama as the local inference engine."
        }
    },
    {
        "inputs": {
            "question": "What compliance laws can companies adhere to by using the Secure RAG API?"
        },
        "outputs": {
            "answer": "Companies can adhere to HIPAA, SOC 2, and GDPR."
        }
    },
    {
        "inputs": {
            "question": "How does the Secure RAG API achieve multi-tenancy and data isolation?"
        },
        "outputs": {
            "answer": "It implements metadata filtering based on user_id so users can only query their own documents."
        }
    },
    {
        "inputs": {
            "question": "What are the three steps for the 'Quick Manual Fix' to reset the database?"
        },
        "outputs": {
            "answer": "Stop the FastAPI server, delete the chroma_db folder directly in the VS Code file explorer, and restart the server."
        }
    }
]
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)

print("Evaluation dataset created!")