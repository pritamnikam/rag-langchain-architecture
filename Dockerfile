# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all example scripts into the container
COPY "01-embeddings-tf-idf.py" ./
COPY "02-openai-embedding.py" ./
COPY "03-embeddings-langchain-chromadb.py" ./
COPY "04-langchain-augmented-query.py" ./
COPY "05-langchain-generation.py" ./
COPY "06-streamlit-hello.py" ./
COPY "07-langchain-streamlit-rag-qa.py" ./
COPY "08-langchain-qdrant-vector-store-rag-qa.py" ./
COPY "09-langchain-rag-pdf-document.py" ./

# No default command; specify which script to run via CMD or docker run
# Example:
#   docker run --rm rag-examples python 01-embeddings-tf-idf.py
#   docker run --rm -e OPENAI_API_KEY=your-key rag-examples python 02-openai-embedding.py
#   docker run --rm -e OPENAI_API_KEY=your-key rag-examples python 03-embeddings-langchain-chromadb.py
#   docker run --rm rag-examples python 04-langchain-augmented-query.py
#   docker run --rm -e OPENAI_API_KEY=your-key rag-examples python 05-langchain-generation.py
#   # To run the Streamlit hello app (map port 8501):
#   docker run --rm -p 8501:8501 rag-examples streamlit run 06-streamlit-hello.py
#   # To run the Streamlit RAG QA app:
#   docker run --rm -p 8501:8501 rag-examples streamlit run 07-langchain-streamlit-rag-qa.py
#   # To run the Qdrant RAG QA script:
#   docker run --rm -e OPENAI_API_KEY=your-key rag-examples python 08-langchain-qdrant-vector-store-rag-qa.py
#   # To run the PDF RAG Streamlit app (requires PyMuPDF):
#   docker run --rm -p 8501:8501 rag-examples streamlit run 09-langchain-rag-pdf-document.py
#
# Note: PyMuPDF is required for PDF support in 09-langchain-rag-pdf-document.py
