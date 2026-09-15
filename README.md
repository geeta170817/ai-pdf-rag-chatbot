# AI PDF Chatbot using RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python and Streamlit.

The app allows a user to upload a PDF document, ask natural-language questions, retrieve the most relevant section of the document using semantic similarity, and generate a short answer with a local language model.

## Features

- Upload a PDF document
- Extract text from the PDF
- Split document text into overlapping chunks
- Create embeddings with `all-MiniLM-L6-v2`
- Convert user questions into embeddings
- Retrieve the most relevant chunk using cosine similarity
- Generate answers with `google/flan-t5-base`
- Chat-style Streamlit interface
- Maintain chat history during the session
- Show the retrieved source context for transparency
- No paid API key required for the current version

## RAG Flow

```text
PDF
 ↓
Text Extraction
 ↓
Chunking + Overlap
 ↓
Embeddings
 ↓
User Question
 ↓
Question Embedding
 ↓
Cosine Similarity
 ↓
Relevant Chunk
 ↓
Prompt + Context
 ↓
FLAN-T5
 ↓
Final Answer
```

## Project Structure

```text
PDF_RAG_Chatbot/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── practice_company_policy.pdf
```

## Installation

Create and activate a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

## Example Questions

Using the included practice company policy PDF, try questions such as:

- How many annual leave days do employees get?
- How many sick leave days are available?
- How many annual leave days can be carried forward?
- What are the normal working hours?

## Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- Scikit-learn
- Hugging Face Transformers
- FLAN-T5

## Learning Purpose

This project demonstrates the core concepts of a basic RAG application:

1. Document ingestion
2. Text extraction
3. Chunking
4. Embeddings
5. Semantic retrieval
6. Prompt construction
7. LLM-based answer generation
8. Simple web deployment interface

## Notes

- The included company policy PDF is fictional and intended only for demonstration and practice.
- The current version retrieves the single most relevant chunk.
- For larger or production systems, a vector database such as FAISS, Chroma, Pinecone, pgvector, or Azure AI Search can be added.
- Production systems should also include stronger document parsing, metadata, citations, security, monitoring, and evaluation.

## Security

Do not commit API keys, passwords, tokens, or `.env` files to GitHub. The `.gitignore` file already excludes `.env`.

## Future Improvements

- Retrieve the top 3 most relevant chunks instead of only one
- Add page-number citations
- Support multiple PDFs
- Add a vector database
- Improve model quality
- Add conversation-aware retrieval
- Deploy the Streamlit app
