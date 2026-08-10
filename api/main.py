import io
from typing import Annotated

import PyPDF2
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="StudyMind AI API")


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

    return {
        "filename": file.filename,
        "page_count": len(reader.pages),
        "character_count": len(text),
    }

