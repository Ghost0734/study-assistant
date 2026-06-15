from urllib import response

import streamlit as st
from google import genai
import chromadb
import PyPDF2
import io
import hashlib
import json

client_ai = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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

    [data-testid="stChatInput"] {
        background: #1e1f20 !important;
        border-radius: 24px !important;
        border: 1px solid #3c3c3c !important;
        padding: 8px 16px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #8ab4f8 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #e3e3e3 !important;
        background: transparent !important;
        font-size: 1rem !important;
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
    response = client_ai.models.generate_content(model="gemini-2.0-flash", contents=prompt)
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
    response = client_ai.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = response.text.strip()
    # Strip markdown code fences if Gemini wraps the response in them
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("python") or text.startswith("json"):
            text = text[7:]
    return json.loads(text.strip())

if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
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
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            collection_name = f"study_{file_hash}"

            with st.spinner("Reading PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            with st.spinner("Processing..."):
                chunks = chunk_text(text)
                st.session_state.collection = store_chunks(chunks, collection_name)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

            st.success(f"✅ {len(chunks)} chunks ready")

    if st.session_state.collection:
        st.markdown("---")
        st.markdown(f"📄 **{st.session_state.pdf_name}**")
        st.markdown("---")
        if st.button("🧠 Generate Quiz", use_container_width=True):
            st.session_state.mode = "quiz"
            st.session_state.quiz_data = None
            st.rerun()

        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()

        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.collection = None
            st.session_state.pdf_name = ""
            st.session_state.messages = []
            st.rerun()

if not st.session_state.collection:
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
                    st.session_state.quiz_data = generate_quiz(st.session_state.collection)
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answer = None
                except Exception as e:
                    st.error(f"Failed to generate quiz: {e}")

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

    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if question := st.chat_input("Ask anything about your study material..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    chunks = get_relevant_chunks(st.session_state.collection, question)
                    answer = ask_gemini(question, chunks)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})