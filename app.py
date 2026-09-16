import re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Document Assistant")
st.caption("Upload a text-based PDF and ask questions grounded in your document.")

with st.expander("ℹ️ How it works"):
    st.write("Your PDF is split into searchable chunks. The app retrieves the most relevant sections for your question and sends that document context to the AI model to prepare the answer.")

HF_MODEL = "Qwen/Qwen3-8B"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm_client():
    token = st.secrets.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from Streamlit Secrets.")
    return InferenceClient(provider="auto", api_key=token)


def clean_pdf_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except (TypeError, ValueError):
            text = page.extract_text() or ""
        text = clean_pdf_text(text)
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
    stop = {"what", "which", "when", "where", "who", "why", "how", "is", "are", "was", "were", "the", "a", "an", "of", "to", "in", "on", "for", "this", "that", "it", "me", "tell", "please", "pdf", "document", "file", "give"}
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in stop]


def retrieve_top_chunks(
    question,
    chunks,
    chunk_embeddings,
    embedding_model,
    top_k=5
):
    # Create embedding for the user's question
    q_embedding = embedding_model.encode(question)

    # Semantic similarity between question and every chunk
    semantic_scores = cosine_similarity(
        [q_embedding],
        chunk_embeddings
    )[0]

    question_lower = question.lower()
    q_words = keywords(question)

    # Add common document-field synonyms to improve retrieval
    expanded_words = set(q_words)

    synonym_map = {
        "origin": ["origin", "from", "departure", "place of receipt"],
        "destination": ["destination", "to", "arrival", "place of delivery"],
        "booking": ["booking", "booking no", "booking number"],
        "service": ["service", "service mode"],
        "mode": ["mode", "service mode"],
        "eta": ["eta", "arrival", "estimated arrival"],
        "vessel": ["vessel", "ship"],
        "commodity": ["commodity", "cargo", "goods"],
    }

    for key, synonyms in synonym_map.items():
        if key in question_lower:
            expanded_words.update(synonyms)

    scored = []

    for i, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()

        # Base semantic score
        score = float(semantic_scores[i])

        # Keyword/synonym matches
        for word in expanded_words:
            if word in text_lower:
                score += 0.35

        # Give stronger weight when important document labels occur
        for key, synonyms in synonym_map.items():
            if key in question_lower:
                for synonym in synonyms:
                    if synonym in text_lower:
                        score += 0.75

        scored.append((score, i))

    # Highest scoring chunks first
    best = sorted(
        scored,
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [chunks[i] for _, i in best]


def format_context(selected):
    return "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in selected)


def is_summary_question(question):
    q = question.lower().strip()
    if any(word in q for word in ["summarize", "summarise", "summary", "overview"]): return True
    doc_words = ["pdf", "document", "file", "report", "article"]
    intents = ["about", "explain", "describe", "contain", "cover"]
    if any(d in q for d in doc_words) and any(i in q for i in intents): return True
    return q in ["what is it about", "tell me about it", "explain it", "describe it", "what is this about"]


def compact_text(text): return re.sub(r"\s+", "", text).lower()
def clean_value(value): return re.sub(r"\s+", " ", value).strip(" :-\t")


def extract_with_patterns(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = clean_value(match.group(1))
            if value: return value
    return None


def extract_exact_field(question, pages):
    q = question.lower().strip(); text = "\n".join(p["text"] for p in pages); compact = compact_text(text)
    field_aliases = [("service_mode", ["service mode"]),("booking_number", ["booking number", "booking no", "booking id"]),("destination", ["destination", "where is it going", "where is the shipment going", "ship to"]),("origin", ["origin", "where is it from", "shipment from", "ship from"]),("commodity", ["commodity", "cargo description", "commodity description", "what is being shipped", "what cargo"]),("booked_by", ["booked by party", "booked by", "who booked"]),("business_unit", ["business unit"]),("price_owner", ["price owner"]),("service_contract", ["service contract"]),("print_date", ["print date"]),("price_calculation_date", ["price calculation date"]),("merchant_release", ["merchant haulage release reference", "release reference"]),("vessel", ["vessel", "ship name"]),("container", ["container number", "container no", "container"])]
    requested = next((field for field, aliases in field_aliases if any(alias in q for alias in aliases)), None)
    if requested is None: return None
    patterns = {"service_mode":[r"Service\s*Mode\s*:\s*(CY\s*/\s*CY|CFS\s*/\s*CFS|CY\s*/\s*CFS|CFS\s*/\s*CY)"],"booking_number":[r"Booking\s*No\s*\.?\s*:\s*([A-Za-z0-9\-/]+)"],"destination":[r"\bTo\s*:\s*([^\n]{2,120})",r"Destination\s*:\s*([^\n]{2,120})"],"origin":[r"\bFrom\s*:\s*([^\n]{2,120})",r"Origin\s*:\s*([^\n]{2,120})"],"commodity":[r"Commodity\s*Description\s*:\s*([^\n]{2,160})",r"Customer\s*Cargo\s*:\s*([^\n]{2,160})"],"booked_by":[r"Booked\s*by\s*Party\s*:\s*([^\n]{2,120})"],"business_unit":[r"Business\s*Unit\s*:\s*([^\n]{2,120})"],"price_owner":[r"Price\s*Owner\s*:\s*([^\n]{2,120})"],"service_contract":[r"Service\s*Contract\s*:\s*([^\n]{2,120})"],"print_date":[r"Print\s*Date\s*:\s*([^\n]{2,60})"],"price_calculation_date":[r"Price\s*Calculation\s*Date\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"],"merchant_release":[r"Merchant\s*Haulage\s*Release\s*Reference\s*:\s*([^\n]{2,160})"],"vessel":[r"Vessel\s*(?:Name)?\s*:\s*([^\n]{2,120})"],"container":[r"Container\s*(?:No\.?|Number)?\s*:\s*([A-Z]{4}\s*\d{6,7}|[A-Za-z0-9\-/]+)"]}
    value = extract_with_patterns(text, patterns.get(requested, []))
    if value and requested == "service_mode":
        code = re.search(r"\b(CY|CFS)\s*/\s*(CY|CFS)\b", value, flags=re.IGNORECASE)
        if code: return f"{code.group(1).upper()}/{code.group(2).upper()}"
    if value: return value
    if requested == "service_mode":
        m = re.search(r"servicemode:(cy/cy|cfs/cfs|cy/cfs|cfs/cy)", compact)
        if m: return m.group(1).upper()
    if requested == "booking_number":
        m = re.search(r"bookingno\.?[:]?([0-9]{5,})", compact)
        if m: return m.group(1)
    return None


def extract_eta_answer(question, pages):
    q = question.lower()
    if "eta" not in q and "arrival" not in q: return None
    text = "\n".join(p["text"] for p in pages); dates = re.findall(r"20\d{2}-\d{2}-\d{2}", text)
    if not dates: return None
    if any(word in q for word in ["final", "destination", "auckland"]): return f"The final ETA shown in the document is {sorted(set(dates))[-1]}."
    return None


def run_llm(prompt, client, max_new_tokens=200):
    response = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions using only the supplied document context. "
                    "Return ONLY the final answer. "
                    "Do not show reasoning, analysis, thinking, or explanation. "
                    "Be factual and concise. "
                    "Never invent information."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_new_tokens,
        temperature=0.1
    )

    message = response.choices[0].message

    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        cleaned = content.strip()

        # Remove Qwen thinking blocks if present
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        if cleaned:
            return cleaned

    reasoning = getattr(message, "reasoning_content", None)

    if isinstance(reasoning, str) and reasoning.strip():
        cleaned = reasoning.strip()

        # If the provider puts the final answer inside reasoning_content,
        # try to extract only the final answer.
        match = re.search(
            r"(?:final answer|answer)\s*[:\-]\s*(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip()

    raise RuntimeError(
        "The hosted model did not return a final answer. Please try again."
    )
def answer_question(question, selected, pages, client):
    context = format_context(selected)

    prompt = f"""
Answer the user's question using ONLY the PDF context below.

IMPORTANT RULES:

1. Find the exact field or information requested.

2. Return the COMPLETE value associated with that field.
   Do not shorten, summarize, or return only part of the value.

3. For locations, return the complete location exactly as shown
   in the document, including port/place name, city/state/region,
   and country when available.

4. For identifiers such as booking numbers, reference numbers,
   container numbers, dates, service modes, and codes, copy the
   exact value from the document.

5. For ETA, ETD, arrival, departure, or transport-plan questions,
   carefully interpret table rows using their column headers.

   If the table contains columns such as:
   From | To | Mode | Vessel | Voy No. | ETD | ETA

   first locate the row where the "To" destination matches the
   destination asked by the user.

   Then return the value from the ETA column of THAT SAME ROW.

   Do not use an ETA from an earlier transport leg.
   Do not confuse ETD with ETA.

   Example:
   If a row contains:
   Tauranga Container | Auckland Metroport | Inland RR |
   2026-11-02 | 2026-11-03

   then Auckland Metroport ETA is 2026-11-03.

6. PDF text may come from tables. A label and its value may appear
   on the same line, next line, or nearby line. Read the surrounding
   context carefully.

7. Do not return a field label such as "From", "To", "Print Date",
   "Booking No.", or "ETA" as the answer.

8. Do not guess or use outside knowledge.

9. Return ONLY the final answer.
   Do not provide reasoning or explanation.

10. If the requested information genuinely cannot be found in the
    supplied context, return exactly:
    I could not find this information in the document.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""

    answer = run_llm(prompt, client, 300)

    return answer, context
def remove_summary_noise(text):
    kept=[]; seen=set(); noise_terms=["maerskline.com","maersk.com","terms and conditions","sanctions laws","all rights reserved","http://","https://","www.","warrant and represent","identified on any list","sanctioned party"]
    for raw_line in text.splitlines():
        line=clean_value(raw_line)
        if not line or len(line)<3: continue
        low=line.lower()
        if any(term in low for term in noise_terms): continue
        normalized=re.sub(r"[^a-z0-9]+"," ",low).strip()
        if not normalized or normalized in seen: continue
        seen.add(normalized); kept.append(line)
    return "\n".join(kept)


def build_business_summary_facts(pages):
    facts=[]
    for label, question in [("Booking number","what is the booking number"),("Origin","what is the origin"),("Destination","what is the destination"),("Service mode","what is the service mode"),("Commodity","what is the commodity"),("Booked by","who booked")]:
        value=extract_exact_field(question,pages)
        if value: facts.append(f"{label}: {value}")
    all_text="\n".join(p["text"] for p in pages); dates=sorted(set(re.findall(r"20\d{2}-\d{2}-\d{2}",all_text)))
    if dates: facts.append(f"Dates appearing in document: {', '.join(dates[:8])}")
    first_page=pages[0]["text"] if pages else ""; cleaned=remove_summary_noise(first_page)
    if cleaned: facts.append("Important first-page content:\n"+cleaned[:3000])
    return "\n".join(facts)


def summarize_document(pages, client):
    summary_context=build_business_summary_facts(pages)
    prompt=f"Explain what this PDF is mainly about using ONLY the reliable document facts below. Write 2 to 4 clear sentences. Identify the document type and business purpose when visible. Mention important booking/shipment information such as parties, origin, destination, cargo and schedule when available. Ignore website names, repeated headers/footers, legal boilerplate and sanctions wording. Do not invent information. Return only the summary.\n\nRELIABLE DOCUMENT FACTS:\n{summary_context}\n\nDOCUMENT SUMMARY:"
    return run_llm(prompt,client,400),summary_context


embedding_model=load_embedding_model()
try:
    llm_client=load_llm_client()
except Exception as exc:
    st.error(f"Could not initialize the hosted LLM: {exc}"); st.stop()
if "messages" not in st.session_state: st.session_state.messages=[]
if "pdf_name" not in st.session_state: st.session_state.pdf_name=None
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], help="Upload a text-based PDF. Scanned/image-only PDFs are not currently supported.")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("About")
    st.caption("Answers are generated from the content retrieved from your uploaded PDF.")
    st.caption("RAG • Sentence Transformers • Qwen")

if uploaded_file is None:
    st.info("👈 Upload a PDF from the sidebar to start chatting with your document.")
    st.stop()
if st.session_state.pdf_name!=uploaded_file.name:
    st.session_state.messages=[]; st.session_state.pdf_name=uploaded_file.name
pages=extract_pages(uploaded_file)
if not pages:
    st.error("No readable text could be extracted. This app currently supports text-based PDFs, not scanned/image-only PDFs."); st.stop()
chunks=create_chunks(pages); chunk_texts=[c["text"] for c in chunks]
with st.spinner("Preparing document search..."): chunk_embeddings=embedding_model.encode(chunk_texts)
st.success("✅ Document ready")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 File", uploaded_file.name)
with col2:
    st.metric("Pages", len(pages))
with col3:
    st.metric("Searchable chunks", len(chunks))
st.divider()
if st.button("📑 Summarize Document"):
    try:
        with st.spinner("📑 Reading the document and preparing a summary..."):
            summary, summary_context = summarize_document(
                pages,
                llm_client
            )

        st.subheader("📑 Document Summary")
        st.write(summary)

        with st.expander("🔍 View facts used for summary"):
            st.text(summary_context)

    except Exception as exc:
        st.error("Could not summarize the document.")
        st.caption(str(exc))

st.divider()

st.subheader("💬 Chat with your document")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"]=="assistant" and message.get("context"):
            with st.expander(message.get("context_label","Show source context")): st.text(message["context"])
question=st.chat_input("Ask anything about this PDF...")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"): st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔎 Searching the document and preparing your answer..."):
                if is_summary_question(question): answer,context=summarize_document(pages,llm_client); label="🔍 View facts used for summary"
                else:
                    selected=retrieve_top_chunks(question,chunks,chunk_embeddings,embedding_model,top_k=5); answer,context=answer_question(question,selected,pages,llm_client); label="🔍 View retrieved source context"
            st.write(answer)
            with st.expander(label): st.text(context)
            st.session_state.messages.append({"role":"assistant","content":answer,"context":context,"context_label":label})
        except Exception as exc:
            st.error("The hosted AI model could not answer this request.")
            st.caption(str(exc))
