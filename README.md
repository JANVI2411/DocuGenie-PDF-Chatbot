# PDF Question-Answering App (Streamlit + FastAPI + LangChain + RAG)

This project allows users to upload PDFs via a Streamlit frontend. The FastAPI backend processes the uploaded PDF, creates a vector store, and saves the document. Users can ask questions related to the PDF, and the app provides accurate answers using Retrieval-Augmented Generation (RAG) with LangChain and a local HuggingFace LLM pipeline.

## Features

- **PDF Upload**: Upload a PDF file through the frontend (Streamlit).
- **Backend PDF Processing**: The backend (FastAPI) and LangChain process the uploaded PDF, convert it into embeddings, and store it in a ChromaDB vector store.
- **Question-Answering System**: Users can ask questions related to the content of the PDF, and the system will answer based on the processed data using RAG.
- **Local Language Model**: A local language model (LLM) pipeline from HuggingFace and LangChain is used to generate responses based on the retrieved document embeddings.

## Tech Stack

- **Streamlit**: Frontend for uploading PDFs and interacting with the system.
- **FastAPI**: Backend for processing the uploaded PDFs and handling API requests.
- **LangChain**: Provides the chain logic to handle Retrieval-Augmented Generation (RAG) and conversational chain.
- **HuggingFace Local LLM**: The language model used for generating answers based on retrieved data.
- **Vector Store (ChromaDB)**: A database of document embeddings for semantic search.

## Installation

**Prerequisites**: Python 3.8 or later

### Clone the repository

Clone the repository
```bash
git clone https://github.com/JANVI2411/LLM-PDF-QA-Summarizer.git
cd llm-pdf-qa-summarizer
```

Install Dependencies
```bash
pip install -r requirements.txt
pip install uvicorn
pip install streamlit
```

### Usage

Start the FastAPI Backend

```bash
uvicorn fastapi_app:app --reload
```
This will start the backend server for PDF processing and question-answering.

Start the Streamlit Frontend

In another terminal, run:

```bash
streamlit run upload_pdf.py
```

This will launch the Streamlit UI where you can upload PDFs and ask questions.


## Future Enhancements
- Multi-document support: Allow users to upload and query multiple PDFs.
- Advanced search: Improve the vector search with better ranking algorithms.

## Contributing

Feel free to submit issues, feature requests, and pull requests!
