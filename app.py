import re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="centered")
st.title("AI PDF Chatbot")
st.write("Upload a text-based PDF and ask questions about it.")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    name = "google/flan-t5-base"
    return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)


def extract_pages(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": number, "text": text})
    return pages


def create_chunks(pages, chunk_size=500, overlap=100):
    chunks = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append({"page": item["page"], "text": chunk})
            start += chunk_size - overlap
    return chunks


def keywords(text):
    stop = {"what", "which", "when", "where", "who", "why", "how", "is", "are", "was", "were",
            "the", "a", "an", "of", "to", "in", "on", "for", "this", "that", "it", "me", "tell",
            "please", "pdf", "document", "file", "give"}
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in stop]


def retrieve_top_chunks(question, chunks, chunk_embeddings, embedding_model, top_k=3):
    q_embedding = embedding_model.encode(question)
    semantic = cosine_similarity([q_embedding], chunk_embeddings)[0]
    q_words = keywords(question)
    scored = []
    for i, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        keyword_hits = sum(1 for word in q_words if word in text_lower)
        important_phrase = " ".join(q_words)
        phrase_bonus = 1.0 if important_phrase and important_phrase in text_lower else 0.0
        score = float(semantic[i]) + (0.35 * keyword_hits) + phrase_bonus
        scored.append((score, i))
    best = sorted(scored, reverse=True)[:top_k]
    return [chunks[i] for _, i in best]


def format_context(selected):
    return "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in selected)


def is_summary_question(question):
    q = question.lower().strip()
    if any(word in q for word in ["summarize", "summarise", "summary", "overview"]):
        return True
    doc_words = ["pdf", "document", "file", "report", "article"]
    intents = ["about", "explain", "describe", "contain", "cover"]
    if any(d in q for d in doc_words) and any(i in q for i in intents):
        return True
    return q in ["what is it about", "tell me about it", "explain it", "describe it", "what is this about"]


def compact_text(text):
    return re.sub(r"\s+", "", text).lower()


def clean_value(value):
    value = re.sub(r"\s+", " ", value).strip(" :-\t")
    return value


def extract_with_patterns(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = clean_value(match.group(1))
            if value:
                return value
    return None


def extract_exact_field(question, pages):
    """Deterministically answer common business-document label/value questions."""
    q = question.lower().strip()
    text = "\n".join(p["text"] for p in pages)
    compact = compact_text(text)

    # Identify the user's intended field using natural aliases.
    field_aliases = [
        ("service_mode", ["service mode"]),
        ("booking_number", ["booking number", "booking no", "booking id"]),
        ("destination", ["destination", "where is it going", "where is the shipment going", "ship to"]),
        ("origin", ["origin", "where is it from", "shipment from", "ship from"]),
        ("commodity", ["commodity", "cargo description", "commodity description", "what is being shipped", "what cargo"]),
        ("booked_by", ["booked by party", "booked by", "who booked"]),
        ("business_unit", ["business unit"]),
        ("price_owner", ["price owner"]),
        ("service_contract", ["service contract"]),
        ("print_date", ["print date"]),
        ("price_calculation_date", ["price calculation date"]),
        ("merchant_release", ["merchant haulage release reference", "release reference"]),
        ("vessel", ["vessel", "ship name"]),
        ("container", ["container number", "container no", "container"])
    ]

    requested = None
    for field, aliases in field_aliases:
        if any(alias in q for alias in aliases):
            requested = field
            break
    if requested is None:
        return None

    patterns = {
        "service_mode": [
            r"Service\s*Mode\s*:\s*(CY\s*/\s*CY|CFS\s*/\s*CFS|CY\s*/\s*CFS|CFS\s*/\s*CY)",
            r"Service\s*Mode\s*:\s*([^\n]{1,30})"
        ],
        "booking_number": [
            r"Booking\s*No\s*\.?\s*:\s*([A-Za-z0-9\-/]+)"
        ],
        "destination": [
            r"\bTo\s*:\s*([^\n]{2,120})",
            r"Destination\s*:\s*([^\n]{2,120})"
        ],
        "origin": [
            r"\bFrom\s*:\s*([^\n]{2,120})",
            r"Origin\s*:\s*([^\n]{2,120})"
        ],
        "commodity": [
            r"Commodity\s*Description\s*:\s*([^\n]{2,160})",
            r"Customer\s*Cargo\s*:\s*([^\n]{2,160})"
        ],
        "booked_by": [
            r"Booked\s*by\s*Party\s*:\s*([^\n]{2,120})"
        ],
        "business_unit": [
            r"Business\s*Unit\s*:\s*([^\n]{2,120})"
        ],
        "price_owner": [
            r"Price\s*Owner\s*:\s*([^\n]{2,120})"
        ],
        "service_contract": [
            r"Service\s*Contract\s*:\s*([^\n]{2,120})"
        ],
        "print_date": [
            r"Print\s*Date\s*:\s*([^\n]{2,60})"
        ],
        "price_calculation_date": [
            r"Price\s*Calculation\s*Date\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"
        ],
        "merchant_release": [
            r"Merchant\s*Haulage\s*Release\s*Reference\s*:\s*([^\n]{2,160})"
        ],
        "vessel": [
            r"Vessel\s*(?:Name)?\s*:\s*([^\n]{2,120})"
        ],
        "container": [
            r"Container\s*(?:No\.?|Number)?\s*:\s*([A-Z]{4}\s*\d{6,7}|[A-Za-z0-9\-/]+)"
        ]
    }

    value = extract_with_patterns(text, patterns.get(requested, []))

    # Fix standard service-mode formatting.
    if value and requested == "service_mode":
        code = re.search(r"\b(CY|CFS)\s*/\s*(CY|CFS)\b", value, flags=re.IGNORECASE)
        if code:
            return f"{code.group(1).upper()}/{code.group(2).upper()}"

    if value:
        return value

    # Compact fallbacks handle PDF extraction that breaks labels into letters.
    if requested == "service_mode":
        m = re.search(r"servicemode:(cy/cy|cfs/cfs|cy/cfs|cfs/cy)", compact)
        if m:
            return m.group(1).upper()

    if requested == "booking_number":
        m = re.search(r"bookingno\.?[:]?([0-9]{5,})", compact)
        if m:
            return m.group(1)

    # Destination/origin are particularly useful in logistics PDFs. Search
    # compact text between nearby known labels when normal line extraction fails.
    if requested == "destination":
        m = re.search(r"to:(.{3,100}?)(?:customercargo:|servicecontract:|priceowner:|businessunit:)", compact)
        if m:
            raw = m.group(1)
            # Compact fallback cannot reliably restore spaces, so only use it
            # when normal extraction failed and show the extracted raw value.
            return raw

    if requested == "origin":
        m = re.search(r"from:(.{3,100}?)(?:contactname:|bookedbyref|to:)", compact)
        if m:
            return m.group(1)

    return None


def extract_eta_answer(question, pages):
    q = question.lower()
    if "eta" not in q and "arrival" not in q:
        return None
    text = "\n".join(p["text"] for p in pages)
    dates = re.findall(r"20\d{2}-\d{2}-\d{2}", text)
    if not dates:
        return None
    if any(word in q for word in ["final", "destination", "auckland"]):
        unique_dates = sorted(set(dates))
        return f"The final ETA shown in the document is {unique_dates[-1]}."
    return None


def run_llm(prompt, tokenizer, llm_model, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def answer_question(question, selected, pages, tokenizer, llm_model):
    context = format_context(selected)
    exact = extract_exact_field(question, pages)
    if exact:
        return exact, context
    eta_answer = extract_eta_answer(question, pages)
    if eta_answer:
        return eta_answer, context
    prompt = f"""Answer the question using ONLY the context below.
Read tables and label-value fields carefully.
Copy exact names, codes, numbers and dates from the context when answering factual questions.
If several values match the question, list the relevant values instead of guessing one.
If the answer is absent, say exactly: I could not find this information in the document.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""
    return run_llm(prompt, tokenizer, llm_model, 100), context


def summarize_document(chunks, tokenizer, llm_model):
    if len(chunks) <= 6:
        selected = chunks
    else:
        indices = sorted(set(round(i * (len(chunks) - 1) / 5) for i in range(6)))
        selected = [chunks[i] for i in indices]
    mini_summaries = []
    for item in selected:
        prompt = f"""Extract the main useful facts from this PDF excerpt in one short sentence.
Ignore boilerplate/legal wording unless it is the main subject.
Use only the excerpt.

Page {item['page']} excerpt:
{item['text']}

Main facts:"""
        mini = run_llm(prompt, tokenizer, llm_model, 55)
        if mini:
            mini_summaries.append(f"Page {item['page']}: {mini}")
    combined = "\n".join(mini_summaries)
    final_prompt = f"""Using the notes below, explain what the PDF is mainly about in 2 to 4 clear sentences.
Mention the document type, main purpose, and important details when available.
Do not invent information.

NOTES:
{combined}

DOCUMENT SUMMARY:"""
    answer = run_llm(final_prompt, tokenizer, llm_model, 120)
    return answer, combined


embedding_model = load_embedding_model()
tokenizer, llm_model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is None:
    st.info("Upload a PDF to start chatting with the document.")
    st.stop()

if st.session_state.pdf_name != uploaded_file.name:
    st.session_state.messages = []
    st.session_state.pdf_name = uploaded_file.name

pages = extract_pages(uploaded_file)
if not pages:
    st.error("No readable text could be extracted. This app currently supports text-based PDFs, not scanned/image-only PDFs.")
    st.stop()

chunks = create_chunks(pages)
chunk_texts = [c["text"] for c in chunks]
with st.spinner("Preparing document search..."):
    chunk_embeddings = embedding_model.encode(chunk_texts)

st.success(f"PDF ready. {len(pages)} pages and {len(chunks)} searchable chunks prepared.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander(message.get("context_label", "Show source context")):
                st.text(message["context"])

question = st.chat_input("Ask a question about the PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Reading the document and preparing the answer..."):
            if is_summary_question(question):
                answer, context = summarize_document(chunks, tokenizer, llm_model)
                label = "Show summary notes"
            else:
                selected = retrieve_top_chunks(question, chunks, chunk_embeddings, embedding_model, top_k=3)
                answer, context = answer_question(question, selected, pages, tokenizer, llm_model)
                label = "Show top retrieved source context"
        st.write(answer)
        with st.expander(label):
            st.text(context)
    st.session_state.messages.append({"role": "assistant", "content": answer, "context": context, "context_label": label})
