from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.advanced_rag import answer_question


app = FastAPI(
    title="FinRAG API",
    description="Enterprise Financial Document RAG System",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():

    return {
        "message": "FinRAG API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        result = answer_question(request.question)

        return result

    except HTTPException:
        raise

    except Exception as e:

        error_message = str(e)

        print("ERROR IN /ask:")
        print(error_message)

        if "503" in error_message or "UNAVAILABLE" in error_message:

            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily unavailable. Please try again later."
            )

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

            raise HTTPException(
                status_code=429,
                detail="Gemini API quota has been exceeded. Please try again later."
            )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question."
        )