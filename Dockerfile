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

# No default command; specify which script to run via CMD or docker run
# Example:
#   docker run --rm rag-examples python 01-embeddings-tf-idf.py
#   docker run --rm -e OPENAI_API_KEY=your-key rag-examples python 02-openai-embedding.py
#   docker run --rm -e OPENAI_API_KEY=your-key rag-examples python 03-embeddings-langchain-chromadb.py
#   docker run --rm rag-examples python 04-langchain-augmented-query.py
