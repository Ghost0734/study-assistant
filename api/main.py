import io
import os
import uuid
from typing import Annotated

import chromadb
import PyPDF2
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from pydantic import BaseModel

app = FastAPI(title="StudyMind AI API")

chroma_client = chromadb.Client()
study_sessions = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    index = 0

    while index < len(words):
        chunks.append(" ".join(words[index:index + chunk_size]))
        index += chunk_size - overlap

    return chunks


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: Annotated[UploadFile, File(...)]):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")

    contents = await file.read()

    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF must be 15 MB or smaller.")

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise HTTPException(status_code=400, detail="This PDF could not be read.") from error

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No selectable text was found in this PDF.",
        )

    chunks = chunk_text(text)

    session_id = str(uuid.uuid4())
    collection_name = f"study_{session_id.replace('-', '')}"

    collection = chroma_client.create_collection(collection_name)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{index}" for index in range(len(chunks))],
    )

    study_sessions[session_id] = collection

    return {
        "session_id": session_id,
        "filename": file.filename,
        "page_count": len(reader.pages),
        "chunk_count": len(chunks),
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    collection = study_sessions.get(request.session_id)

    if not collection:
        raise HTTPException(
            status_code=404,
            detail="Study session not found. Upload the PDF again.",
        )

    results = collection.query(
        query_texts=[request.question],
        n_results=min(5, collection.count()),
    )

    relevant_chunks = results["documents"][0]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful study assistant.

Answer the student's question using ONLY the study material below.
If the answer is not in the material, say:
"I couldn't find this in your study material."

Keep the answer clear, concise, and student-friendly.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{request.question}

ANSWER:"""

   
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    client_ai = genai.Client(api_key=api_key)

    response = client_ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    

    return {
        "answer": response.text,
        "sources": relevant_chunks,
    }