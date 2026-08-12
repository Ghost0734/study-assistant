from urllib import response

import streamlit as st
from google import genai
import chromadb
import PyPDF2
import io
import hashlib
import json
import os 
import requests 

client_ai = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="StudyMind AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    * { font-family: 'Google Sans', system-ui, sans-serif; }
    .stApp { background-color: #131314; }

    [data-testid="stSidebar"] {
        background-color: #1e1f20;
        border-right: 1px solid #2d2e2f;
    }
    [data-testid="stSidebar"] * { color: #e3e3e3; }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        max-width: 700px !important;
        margin: 0 auto !important;
        padding-top: 8vh !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    .welcome-heading {
        font-size: 2.8rem;
        font-weight: 500;
        color: #e3e3e3;
        margin-bottom: 2rem;
        line-height: 1.2;
    }

    .cards-row {
        display: flex;
        gap: 12px;
        margin-bottom: 2rem;
    }
    .welcome-card {
        background: #1e1f20;
        border: 1px solid #2d2e2f;
        border-radius: 16px;
        padding: 1.2rem;
        flex: 1;
        transition: border-color 0.2s;
    }
    .welcome-card:hover { border-color: #8ab4f8; }
    .welcome-card h4 { color: #e3e3e3; margin: 0 0 6px 0; font-size: 0.9rem; }
    .welcome-card p { color: #9aa0a6; margin: 0; font-size: 0.82rem; line-height: 1.4; }

    [data-testid="stChatInput"] > div {
    background: #1e1f20 !important;
    border: 1px solid #3c3c3c !important;
    border-radius: 24px !important;
    padding: 8px 16px !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #8ab4f8 !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea {
    color: #e3e3e3 !important;
    background: transparent !important;
    font-size: 1rem !important;
    border: none !important;
    outline: none !important;
}
    

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0;
    }

    .stButton button {
        background: #1e1f20;
        color: #e3e3e3;
        border: 1px solid #3c3c3c;
        border-radius: 20px;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #8ab4f8;
        color: #8ab4f8;
    }

    .stAlert {
        border-radius: 16px !important;
        background: #1e1f20 !important;
        border: 1px solid #2d2e2f !important;
        text-align: center;
    }
    

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:invalid {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
    -webkit-appearance: none !important;
}

    h1, h2, h3 { color: #e3e3e3; }
    p, li { color: #9aa0a6; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

client = get_chroma_client()

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def store_chunks(chunks, collection_name):
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def get_relevant_chunks(collection, question, n=5):
    results = collection.query(
        query_texts=[question],
        n_results=min(n, collection.count())
    )
    return results["documents"][0]

def ask_gemini(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful study assistant. Answer the student's question 
using ONLY the information from the study material provided below.
If the answer isn't in the material, say "I couldn't find this in your study material."
Keep answers clear, concise and student-friendly.

STUDY MATERIAL:
{context}

STUDENT'S QUESTION:
{question}

ANSWER:"""
    response = client_ai.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def generate_quiz(collection):
    chunks = get_relevant_chunks(collection, "key concepts and important topics", n=5)
    context = "\n\n".join(chunks)
    prompt = f"""You are a quiz generator. Based on the study material below, generate 5 multiple choice questions.

Return ONLY a JSON array in this exact format, no extra text:
[
  {{
    "question": "Question text here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1"
  }}
]

STUDY MATERIAL:
{context}"""
    response = client_ai.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = response.text.strip()
    # Extract just the JSON array from the response
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end])

def generate_flashcards(collection):
    chunks = get_relevant_chunks(collection, "key concepts definitions and important terms", n=5)
    context = "\n\n".join(chunks)
    prompt = f"""You are a flashcard generator. Based on the study material below, generate 8 flashcards.

Return ONLY a JSON array in this exact format, no extra text:
[
  {{
    "term": "Term or concept here",
    "definition": "Clear, concise definition or explanation here"
  }}
]

STUDY MATERIAL:
{context}"""
    response = client_ai.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = response.text.strip()
    # Extract just the JSON array from the response
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end])

if "flashcard_data" not in st.session_state:
    st.session_state.flashcard_data = None
if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0
if "flashcard_flipped" not in st.session_state:
    st.session_state.flashcard_flipped = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_answer" not in st.session_state:
    st.session_state.quiz_answer = None

with st.sidebar:
    st.markdown("### 📚 StudyMind AI")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your study material", type="pdf")

    if uploaded_file:
     if uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Uploading and processing PDF..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                st.session_state.session_id = data["session_id"]
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

                st.success(f"✅ {data['chunk_count']} chunks ready")
            except requests.RequestException as error:
                st.error(f"Could not process the PDF: {error}")

     if st.session_state.session_id:
        st.markdown("---")
        st.markdown(f"📄 **{st.session_state.pdf_name}**")
        st.markdown("---")
        if st.button("🧠 Generate Quiz", use_container_width=True):
            st.session_state.mode = "quiz"
            st.session_state.quiz_data = None
            st.rerun()
        
        if st.button("🃏 Flashcards", use_container_width=True):
             st.session_state.mode = "flashcards"
             st.rerun()

        if st.button("💬 Back to Chat", use_container_width=True):
             st.session_state.mode = "chat"
             st.rerun()

        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.pdf_name = ""
            st.session_state.messages = []
            st.rerun()

if not st.session_state.session_id:
    st.markdown('<p class="welcome-heading">Hello, what would you<br>like to study?</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cards-row">
        <div class="welcome-card">
            <h4>💬 Chat with your notes</h4>
            <p>Ask anything about your uploaded study material</p>
        </div>
        <div class="welcome-card">
            <h4>🧠 Auto-generate quizzes</h4>
            <p>Test yourself with AI-generated questions</p>
        </div>
        <div class="welcome-card">
            <h4>🃏 Flashcard mode</h4>
            <p>Review key concepts with smart flashcards</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("👈 Upload a PDF from the sidebar to get started")

else:
    if st.session_state.mode == "quiz":
        st.markdown("### Quiz Mode")

        if st.session_state.quiz_data is None:
            with st.spinner("Generating quiz..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/quiz",
                        json={"session_id": st.session_state.session_id},
                        timeout=60,
                    )
                    response.raise_for_status()

                    st.session_state.quiz_data = response.json()["questions"]
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answer = None

                except requests.RequestException as error:
                    st.error(f"Failed to generate quiz: {error}")

        if st.session_state.quiz_data:
            questions = st.session_state.quiz_data
            idx = st.session_state.quiz_index

            if idx < len(questions):
                q = questions[idx]
                st.markdown(f"**Question {idx + 1} of {len(questions)}**")
                st.markdown(f"#### {q['question']}")

                selected = st.radio("Choose an answer:", q["options"], key=f"q_{idx}")

                if st.button("Submit Answer"):
                    st.session_state.quiz_answer = selected
                    if selected == q["answer"]:
                        st.session_state.quiz_score += 1

                if st.session_state.quiz_answer:
                    if st.session_state.quiz_answer == q["answer"]:
                        st.success("Correct!")
                    else:
                        st.error(f"Wrong. Correct answer: {q['answer']}")

                    if st.button("Next Question"):
                        st.session_state.quiz_index += 1
                        st.session_state.quiz_answer = None
                        st.rerun()
            else:
                total = len(questions)
                score = st.session_state.quiz_score
                st.markdown("## Quiz Complete!")
                st.markdown(f"### Your score: {score} / {total}")
                if st.button("Retake Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answer = None
                    st.rerun()

    elif st.session_state.mode == "flashcards":
        st.markdown("### 🃏 Flashcard Mode")

        if st.session_state.flashcard_data is None:
            with st.spinner("Generating flashcards..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/flashcards",
                        json={"session_id": st.session_state.session_id},
                        timeout=60,
                    )
                    response.raise_for_status()

                    st.session_state.flashcard_data = response.json()["flashcards"]
                    st.session_state.flashcard_index = 0
                    st.session_state.flashcard_flipped = False

                except requests.RequestException as error:
                    st.error(f"Failed to generate flashcards: {error}")

        if st.session_state.flashcard_data:
            cards = st.session_state.flashcard_data
            idx = st.session_state.flashcard_index
            card = cards[idx]

            st.markdown(f"**Card {idx + 1} of {len(cards)}**")

            if not st.session_state.flashcard_flipped:
                st.markdown(
                    f"""
                    <div class="welcome-card" style="padding: 2rem; text-align: center;">
                        <p style="color: #9aa0a6;">TERM</p>
                        <h2 style="color: #e3e3e3;">{card['term']}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Flip Card 👆", use_container_width=True):
                    st.session_state.flashcard_flipped = True
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div class="welcome-card" style="padding: 2rem; text-align: center;">
                        <p style="color: #8ab4f8;">DEFINITION</p>
                        <p style="color: #e3e3e3;">{card['definition']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.flashcard_index = max(0, idx - 1)
                    st.session_state.flashcard_flipped = False
                    st.rerun()

            with col2:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.flashcard_index = min(len(cards) - 1, idx + 1)
                    st.session_state.flashcard_flipped = False
                    st.rerun()

    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if question := st.chat_input("Ask anything about your study material..."):
     st.session_state.messages.append({"role": "user", "content": question})

     with st.chat_message("user"):
        st.write(question)

     with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": question,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
            except requests.RequestException as error:
                st.error(f"Could not get an answer: {error}")
                answer = None

        if answer:
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )