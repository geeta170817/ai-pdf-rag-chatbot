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

    similarities = cosine_similarity(
        [question_embedding],
        chunk_embeddings
    )

    best_chunk_index = similarities.argmax()

    return chunks[best_chunk_index]


def generate_answer(question, context, tokenizer, llm_model):
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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=80
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

    st.success(
        f"PDF ready. {len(chunks)} chunks created."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "context" in message:
                with st.expander("Show retrieved source context"):
                    st.write(message["context"])

    question = st.chat_input(
        "Ask a question about the PDF"
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching the document and generating an answer..."):

                best_chunk = retrieve_best_chunk(
                    question,
                    chunks,
                    chunk_embeddings,
                    embedding_model
                )

                answer = generate_answer(
                    question,
                    best_chunk,
                    tokenizer,
                    llm_model
                )

            st.write(answer)

            with st.expander("Show retrieved source context"):
                st.write(best_chunk)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "context": best_chunk
            }
        )

else:
    st.info("Upload a PDF to start chatting with the document.")
