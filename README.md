---
title: StudyMind AI
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# StudyMind AI

<p align="center">
  An AI-powered study assistant that transforms PDFs into interactive learning experiences using Retrieval-Augmented Generation (RAG).
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Ghost0734/StudyMind-AI">Live Demo</a> ·
  <a href="https://github.com/Ghost0734/study-assistant">GitHub Repository</a>
</p>

---

## Demo

See StudyMind AI in action, from document-based question answering to quiz and flashcard generation.

https://github.com/user-attachments/assets/14daf72e-0681-4402-9ef1-0548b59933d8

## Overview

StudyMind AI helps students learn more effectively by turning static PDF documents into interactive study material. Upload a document, ask questions about its contents, generate a quiz, or create flashcards for revision.

Powered by Google Gemini 2.5 Flash and ChromaDB, the application retrieves relevant passages from uploaded documents to provide context-grounded responses.

## Features

- **Document Chat:** Ask questions about uploaded PDFs and receive concise answers based on retrieved study material.
- **Quiz Generation:** Generate a five-question multiple-choice quiz with answer options and correct answers.
- **Flashcards:** Generate eight flashcards containing key terms and definitions.
- **Semantic Retrieval:** Search document chunks using ChromaDB's default embedding function.
- **PDF Processing:** Extract selectable text from PDFs using PyPDF2.

## How It Works

1. **Upload:** Submit a PDF of up to 15 MB.
2. **Extract and Chunk:** PyPDF2 extracts the text, which is divided into 500-word chunks with a 50-word overlap.
3. **Index:** ChromaDB stores the chunks for semantic retrieval.
4. **Retrieve:** The application retrieves up to five relevant chunks for a question or study-material generation request.
5. **Generate:** Gemini 2.5 Flash uses the retrieved content to generate answers, quizzes, and flashcards.

For document questions, the prompt instructs the model to respond with *"I couldn't find this in your study material."* when the answer is not present in the retrieved context.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Language Model | Google Gemini 2.5 Flash |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF2 |
| Containerization | Docker |
| Deployment | AWS EC2, Hugging Face Spaces |
| CI/CD | GitHub Actions |

## Limitations

- Scanned or image-only PDFs without selectable text are not supported without OCR.
- AI-generated answers and study material may contain inaccuracies.
- Retrieval is limited to five chunks per request, so relevant information may be missed.
- Document text used for generation is sent to the Google Gemini API. Do not upload confidential or sensitive documents.
- Study sessions are held in memory and may be lost when the backend restarts.

## Source Code

Explore the implementation, API, and deployment configuration in the [GitHub repository](https://github.com/Ghost0734/study-assistant).
