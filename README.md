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

StudyMind AI helps students study more efficiently by letting them upload PDFs and interact with their study material through AI-powered chat, quizzes, and flashcards.

## Live Demo

[Try it on Hugging Face](https://huggingface.co/spaces/Ghost0734/StudyMind-AI)

## Features

- **Chat** — ask any question about your uploaded study material and get concise, accurate answers grounded in the document
- **Quiz** — auto-generate a 5-question multiple choice quiz to test your understanding
- **Flashcards** — generate 8 flashcards with key terms and definitions for quick revision

## How It Works

StudyMind AI uses a RAG (Retrieval-Augmented Generation) pipeline:

1. PDF text is extracted and split into chunks
2. Chunks are stored in ChromaDB, a vector database
3. When you ask a question, the most relevant chunks are retrieved
4. Google Gemini 2.5 Flash uses those chunks to generate a grounded response

## Tech Stack

- **Streamlit** — frontend and app framework
- **Google Gemini 2.5 Flash** — LLM for generating responses
- **ChromaDB** — vector database for semantic search
- **PyPDF2** — PDF text extraction
