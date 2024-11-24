from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from src.generator.generation import QABot
import uvicorn
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QA Bot API",
    description="API for question answering system using PDFs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QABot instance
try:
    bot = QABot(
            es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
            es_api_key=os.getenv("ELASTIC_API_KEY"),
            openai_api_key=os.getenv("openai_api_key"),
            index_name="hr_policies_new",
            retreival_statergy="dense"
    )
except Exception as e:
    logger.error(f"Failed to initialize QABot: {str(e)}")
    raise

# Request and Response Models
class QuestionRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="The question to be answered")
    
class QuestionResponse(BaseModel):
    response: str = Field(..., description="The generated answer")
    sources: List[Dict[str, str]] = Field(..., description="Sources used for the answer")
    
class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    timestamp: str = Field(..., description="Current timestamp")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint to check API health"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Get answer for a question using the QA system
    """
    try:
        logger.info(f"Received question: {request.user_query}")
        
        # Get answer from QABot
        result = bot.get_answer(request.user_query)
        
        # Prepare response
        response = {
            "response": result["response"],
            "sources": result["sources"]
        }
        
        logger.info("Successfully generated response")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"System unhealthy: {str(e)}"
        )

# Configuration for running the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )