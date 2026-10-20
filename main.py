from fastapi import FastAPI
import uvicorn
from api.routes import router as api_router

# Initialize the FastAPI app
app = FastAPI(
    title="Secure-RAG API Gateway",
    description="Hybrid-Cloud Document Intelligence API",
    version="1.0.0"
)

app.include_router(api_router)

# A simple root endpoint to test if the server is running
@app.get("/")
async def root():
    return {"message": "Welcome to the Secure-RAG API Gateway!"}

if __name__ == "__main__":
    # Run the server using Uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    
    
