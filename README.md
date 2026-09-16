# AI Document Assistant using RAG

A deployed Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. Upload a text-based PDF, ask natural-language questions, view the retrieved source context, or generate a short document summary.

## 🚀 Live Demo

**[Open AI Document Assistant](https://ai-pdf-rag-chatbot-bfe6fmjoegbh9k6muiti9f.streamlit.app/)**

## ✨ Features

- Upload and process text-based PDF documents
- Layout-aware PDF text extraction with PyPDF
- Split pages into overlapping searchable chunks
- Create embeddings with `all-MiniLM-L6-v2`
- Hybrid retrieval using semantic similarity plus keyword/synonym matching
- Retrieve the top 5 relevant chunks
- Generate grounded answers with `Qwen/Qwen3-8B` through Hugging Face Inference Providers
- Interpret common document fields, locations, identifiers, dates, and table content
- Generate a short document summary
- Display retrieved source context for transparency
- Maintain chat history during the Streamlit session
- Clear Chat control and document information UI
- Secure hosted-model token through Streamlit Secrets

## 🧠 RAG Architecture

```text
Uploaded PDF
    ↓
PyPDF Text Extraction (layout mode)
    ↓
Text Cleaning
    ↓
Overlapping Page Chunks
    ↓
Sentence Transformer Embeddings
(all-MiniLM-L6-v2)
    ↓
User Question
    ↓
Question Embedding
    ↓
Hybrid Retrieval
Semantic Similarity + Keywords/Synonyms
    ↓
Top 5 Relevant Chunks
    ↓
Prompt + Retrieved PDF Context
    ↓
Qwen3-8B via Hugging Face Inference
    ↓
Grounded Final Answer
```

## 🧪 Tested Examples

The application has been tested with different document types, including a shipping/booking PDF and an employee handbook.

Example questions include:

- What is the booking number?
- What is the origin?
- What is the destination?
- What is the service mode?
- What is the final ETA for Auckland?
- What is the document ID?
- How many annual leave days are provided?
- What is the notice period after confirmation?
- What is the maximum hotel reimbursement in a metro city?
- How much health insurance does the company provide?

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- Scikit-learn
- Hugging Face Hub / InferenceClient
- Qwen3-8B

## 📁 Project Structure

```text
ai-pdf-rag-chatbot/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Create a Hugging Face access token with permission to use Inference Providers. For local development, make the token available securely to the application. For Streamlit Community Cloud, store it in the application's Secrets configuration as:

```toml
HF_TOKEN = "your_hugging_face_token"
```

Never commit the real token to GitHub.

## ▶️ Run Locally

```bash
streamlit run app.py
```

## 🔎 How the Application Works

1. The user uploads a text-based PDF.
2. PyPDF extracts page text using layout-aware extraction where supported.
3. The application cleans the text and creates overlapping chunks.
4. Sentence Transformers converts chunks into embeddings.
5. The user's question is also converted into an embedding.
6. Hybrid retrieval combines cosine similarity with keyword and synonym signals.
7. The five highest-scoring chunks are supplied as context to Qwen3-8B.
8. The hosted model answers using only the retrieved document context.
9. The user can expand the source-context section to inspect what was retrieved.

## 🔐 Security

- Do not commit API keys, passwords, access tokens, or `.env` files.
- `HF_TOKEN` should be stored in Streamlit Secrets for the deployed application.
- The application sends retrieved document context to the configured Hugging Face inference provider when generating answers or summaries.
- Avoid uploading confidential or sensitive documents unless the deployment and model-provider configuration is appropriate for that data.

## ⚠️ Current Limitations

- Supports one PDF at a time.
- Designed for text-based PDFs; scanned/image-only PDFs require OCR, which is not included in the current version.
- Retrieval currently uses in-memory embeddings rather than a persistent vector database.
- Hosted model availability, quotas, latency, and cost depend on the Hugging Face inference provider/account configuration.
- Complex PDF layouts and tables can still require additional parsing improvements.

## 🚧 Future Improvements

- Multiple-PDF support
- Persistent vector database such as FAISS, Chroma, pgvector, Pinecone, or Azure AI Search
- OCR support for scanned documents
- Page-level clickable citations
- Conversation-aware retrieval
- Automated RAG evaluation
- Authentication and user-specific document storage
- Production monitoring and cost controls

## 🎯 Learning Outcomes

This project demonstrates an end-to-end RAG workflow: document ingestion, PDF extraction, chunking, embeddings, hybrid retrieval, prompt grounding, hosted LLM inference, source transparency, Streamlit UI development, secure secret management, and cloud deployment.
