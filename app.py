import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("AI PDF Chatbot")
st.write("Upload a PDF and chat with your document.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, llm_model


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    return full_text


def create_chunks(full_text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - overlap

    return chunks


def retrieve_best_chunk(question, chunks, chunk_embeddings, embedding_model):
    question_embedding = embedding_model.encode(question)
    similarities = cosine_similarity([question_embedding], chunk_embeddings)
    best_chunk_index = similarities.argmax()
    return chunks[best_chunk_index]


def is_summary_question(question):
    q = question.lower().strip()
    summary_phrases = [
        "what is this pdf about",
        "what is the pdf about",
        "what is this document about",
        "what is the document about",
        "this pdf is about what",
        "pdf is about what",
        "summarize this pdf",
        "summarise this pdf",
        "summarize the pdf",
        "summarise the pdf",
        "summarize this document",
        "summarise this document",
        "summary of this pdf",
        "summary of the pdf",
        "give me a summary",
        "give me an overview",
        "overview of this pdf",
        "overview of the document"
    ]
    return any(phrase in q for phrase in summary_phrases)


def build_summary_context(chunks, max_chunks=4):
    # Use several parts of the document instead of only one retrieved chunk.
    if len(chunks) <= max_chunks:
        selected = chunks
    else:
        indices = [
            round(i * (len(chunks) - 1) / (max_chunks - 1))
            for i in range(max_chunks)
        ]
        selected = [chunks[i] for i in indices]

    return "\n\n---\n\n".join(selected)


def generate_answer(question, context, tokenizer, llm_model, summary_mode=False):
    if summary_mode:
        prompt = f"""
Read the document excerpts below and explain what the document is mainly about.
Give a concise summary covering the main topics. Do not invent information.

Document excerpts:
{context}

Summary:
"""
        max_tokens = 120
    else:
        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Give a short and direct answer.
If the answer is not available in the context, say:
"I could not find this information in the document."
"""
        max_tokens = 80

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_tokens
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


embedding_model = load_embedding_model()
tokenizer, llm_model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.session_state.pdf_name != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.pdf_name = uploaded_file.name

    full_text = extract_text_from_pdf(uploaded_file)

    if not full_text.strip():
        st.error("No readable text could be extracted from this PDF.")
        st.stop()

    chunks = create_chunks(full_text)

    with st.spinner("Preparing document embeddings..."):
        chunk_embeddings = embedding_model.encode(chunks)

    st.success(f"PDF ready. {len(chunks)} chunks created.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "context" in message:
                label = message.get("context_label", "Show retrieved source context")
                with st.expander(label):
                    st.write(message["context"])

    question = st.chat_input("Ask a question about the PDF")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                summary_mode = is_summary_question(question)

                if summary_mode:
                    context = build_summary_context(chunks)
                    context_label = "Show document excerpts used for summary"
                else:
                    context = retrieve_best_chunk(
                        question,
                        chunks,
                        chunk_embeddings,
                        embedding_model
                    )
                    context_label = "Show retrieved source context"

                answer = generate_answer(
                    question,
                    context,
                    tokenizer,
                    llm_model,
                    summary_mode=summary_mode
                )

            st.write(answer)

            with st.expander(context_label):
                st.write(context)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context": context,
            "context_label": context_label
        })

else:
    st.info("Upload a PDF to start chatting with the document.")
