from fastapi import FastAPI

app = FastAPI(title="StudyMind AI API")

@app.get("/health")
def health():
    return {"status": "ok"}