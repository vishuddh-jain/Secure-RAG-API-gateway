from fastapi import FastAPI
import uvicorn
from api.routes import router as api_router  # IMPORT YOUR NEW ROUTER

# Initialize the FastAPI app
app = FastAPI(
    title="Secure-RAG API Gateway",
    description="Hybrid-Cloud Document Intelligence API",
    version="1.0.0"
)

# INCLUDE THE ROUTER IN YOUR APP
app.include_router(api_router)

# A simple root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Secure-RAG API Gateway!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)