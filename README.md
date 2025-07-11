# Retrieval-Augmented Generation (RAG) with LangChain: Hands-On Examples

Welcome to a practical, educational repository for learning and experimenting with Retrieval-Augmented Generation (RAG) architectures using Python and [LangChain](https://python.langchain.com/). This project includes:
- Classic and modern retrieval pipelines
- End-to-end RAG with OpenAI LLMs
- A beginner-friendly Streamlit demo
- Docker support for easy reproducibility

## Project Goals
- **Learn** about RAG architectures, vector search, and prompt engineering
- **Experiment** with code for embeddings, retrieval, and LLM integration
- **Prototype** your own RAG pipelines and web demos

---

## Project Structure
- `01-embeddings-tf-idf.py`: TF-IDF vectorization and retrieval demo (no API key needed).
- `02-openai-embedding.py`: OpenAI Embedding API-based retrieval demo (requires OpenAI API key).
- `03-embeddings-langchain-chromadb.py`: LangChain + OpenAI Embeddings + ChromaDB vector store demo (requires OpenAI API key).
- `04-langchain-augmented-query.py`: LangChain prompt augmentation demo (shows how to build a RAG prompt and augment a query; no API key needed).
- `05-langchain-generation.py`: **LangChain RAG pipeline with OpenAI LLM generation**. Shows how to chain a retriever, prompt, and LLM for end-to-end Retrieval-Augmented Generation. Fully commented for learning. **Requires OpenAI API key.**
- `06-streamlit-hello.py`: **Streamlit web demo**. A simple, interactive Streamlit app to demonstrate Python web UIs.
- `08-langchain-qdrant-vector-store-rag-qa.py`: **Minimal RAG pipeline using Qdrant vector store and OpenAI LLMs**. Loads a text file, splits it, embeds with OpenAI, stores in Qdrant (in-memory), and runs a RAG QA chain. **Requires OpenAI API key.**
- `09-langchain-rag-pdf-document.py`: **Streamlit RAG QA app for PDF/TXT documents**. Upload a PDF or TXT, ask questions, and get answers using LangChain, Chroma, and OpenAI. **Requires OpenAI API key and PyMuPDF.**
- `10-langchain-rag-multiple-files-qa.py`: **Streamlit RAG QA app for multiple PDF/TXT files**. Upload multiple PDFs or TXTs, ask questions, and get answers using LangChain, Chroma, and OpenAI. **Requires OpenAI API key and PyMuPDF.**
- `requirements.txt`: Python dependencies for all examples.
- `Dockerfile`: Containerizes all scripts for easy, reproducible runs.
- `.gitignore`: Standard Python, Docker, and editor ignores.

---

## Features
- Local and cloud vector search (TF-IDF, OpenAI, ChromaDB)
- Prompt engineering and LLM generation
- Educational comments and clear code for each step
- Streamlit UI demo for beginners
- Docker support for easy setup and reproducibility

---

## Setup

### 1. Clone the repository
```sh
git clone https://github.com/pritamnikam/rag-langchain-architecture.git
cd rag-langchain-architecture
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. (Optional) Set up your OpenAI API key
Some scripts require an OpenAI API key. Get yours at https://platform.openai.com/account/api-keys

- **Linux/macOS:**
  ```sh
  export OPENAI_API_KEY=sk-...
  ```
- **Windows (Powershell):**
  ```sh
  $env:OPENAI_API_KEY="sk-..."
  ```

### 4. (Optional) Build Docker image
```sh
docker build -t rag-examples .
```

---

## Usage

### Run scripts locally
- **TF-IDF Example:**
  ```sh
  python 01-embeddings-tf-idf.py
  ```
- **OpenAI Embedding Example:**
  ```sh
  python 02-openai-embedding.py
  ```
- **ChromaDB + LangChain Example:**
  ```sh
  python 03-embeddings-langchain-chromadb.py
  ```
- **LangChain Prompt Augmentation:**
  ```sh
  python 04-langchain-augmented-query.py
  ```
- **LangChain RAG Generation:**
  ```sh
  python 05-langchain-generation.py
  ```
- **Qdrant RAG QA Example:**
  ```sh
  python 08-langchain-qdrant-vector-store-rag-qa.py
  ```
- **Streamlit Demo:**
  ```sh
  streamlit run 06-streamlit-hello.py
  ```
- **PDF RAG QA Streamlit App:**
  ```sh
  streamlit run 09-langchain-rag-pdf-document.py
  ```
- **Multiple Files RAG QA Streamlit App:**
  ```sh
  streamlit run 10-langchain-rag-multiple-files-qa.py
  ```

### Run scripts in Docker
- **General pattern:**
  ```sh
  docker run --rm -e OPENAI_API_KEY=sk-... rag-examples python <script.py>
  ```
- **Qdrant RAG QA Example:**
  ```sh
  docker run --rm -e OPENAI_API_KEY=sk-... rag-examples python 08-langchain-qdrant-vector-store-rag-qa.py
  ```
- **Streamlit app:**
  ```sh
  docker run --rm -p 8501:8501 rag-examples streamlit run 06-streamlit-hello.py
  # Then open http://localhost:8501
  ```
- **PDF RAG QA Streamlit App:**
  ```sh
  docker run --rm -p 8501:8501 rag-examples streamlit run 09-langchain-rag-pdf-document.py
  # Then open http://localhost:8501
  ```
- **Multiple Files RAG QA Streamlit App:**
  ```sh
  docker run --rm -p 8501:8501 rag-examples streamlit run 10-langchain-rag-multiple-files-qa.py
  # Then open http://localhost:8501
  ```

---

---

## 08-langchain-qdrant-vector-store-rag-qa.py: Minimal RAG with Qdrant Vector Store

This script demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline using LangChain, Qdrant (in-memory), and OpenAI LLMs. Loads a text file, splits it into chunks, embeds with OpenAI, stores in Qdrant, and runs a RAG QA chain.

**Requirements:**
- Python 3.8+
- Dependencies in `requirements.txt`
- **OpenAI API key** (set the `OPENAI_API_KEY` environment variable)
- `Lakers.txt` (or another text file in the same directory)

**Usage:**
```sh
# Set your OpenAI API key (Linux/macOS)
export OPENAI_API_KEY=sk-...
# Or on Windows (Powershell)
$env:OPENAI_API_KEY="sk-..."

python 08-langchain-qdrant-vector-store-rag-qa.py
```

**Docker:**
```sh
# Place Lakers.txt in the same directory or mount it via -v
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...
docker run --rm -e OPENAI_API_KEY=sk-... -v $PWD/Lakers.txt:/app/Lakers.txt rag-examples python 08-langchain-qdrant-vector-store-rag-qa.py
```

---

## 10-langchain-rag-multiple-files-qa.py: Streamlit RAG QA for Multiple PDF/TXT Documents

This Streamlit app lets you upload multiple PDF and/or TXT files, ask questions, and get concise answers using LangChain, Chroma, and OpenAI LLMs. All files are combined and chunked for retrieval-augmented QA.

**Requirements:**
- Python 3.8+
- Dependencies in `requirements.txt` **including `pymupdf` for PDF support**
- **OpenAI API key**

**Usage:**
```sh
streamlit run 10-langchain-rag-multiple-files-qa.py
```
Open your browser to [http://localhost:8501](http://localhost:8501)

**Docker:**
```sh
docker run --rm -p 8501:8501 rag-examples streamlit run 10-langchain-rag-multiple-files-qa.py
```

---

## 09-langchain-rag-pdf-document.py: Streamlit RAG QA for PDF/TXT Documents

This Streamlit app lets you upload a PDF or TXT, ask questions, and get concise answers using LangChain, Chroma, and OpenAI LLMs.

**Requirements:**
- Python 3.8+
- Dependencies in `requirements.txt` **including `pymupdf` for PDF support**
- **OpenAI API key**

**Usage:**
```sh
streamlit run 09-langchain-rag-pdf-document.py
```
Open your browser to [http://localhost:8501](http://localhost:8501)

**Docker:**
```sh
docker run --rm -p 8501:8501 rag-examples streamlit run 09-langchain-rag-pdf-document.py
```

---

## 05-langchain-generation.py: RAG Pipeline with LLM Generation

This script demonstrates a simple, fully-commented Retrieval-Augmented Generation (RAG) pipeline using LangChain and OpenAI's GPT model. It chains together a retriever, a custom prompt, and an LLM for a complete RAG workflow.

**Requirements:**
- Python 3.8+
- Dependencies in `requirements.txt`
- **OpenAI API key** (set the `OPENAI_API_KEY` environment variable)

**Usage:**
```sh
# Set your OpenAI API key (Linux/macOS)
export OPENAI_API_KEY=sk-...
# Or on Windows (Powershell)
$env:OPENAI_API_KEY="sk-..."

python 05-langchain-generation.py
```

**Docker:**
```sh
docker build -t rag-examples .
docker run --rm -e OPENAI_API_KEY=sk-... rag-examples python 05-langchain-generation.py
```

---

## 06-streamlit-hello.py: Interactive Streamlit Demo

This script is a simple, interactive Streamlit web app to demonstrate how easy it is to build Python-powered web interfaces. It includes user input, sidebar info, and dynamic output, all with clear comments for learning.

**Requirements:**
- Python 3.8+
- `streamlit` (see `requirements.txt`)

**Usage (local):**
```sh
pip install -r requirements.txt
streamlit run 06-streamlit-hello.py
```

**Docker:**
```sh
docker build -t rag-examples .
docker run --rm -p 8501:8501 rag-examples streamlit run 06-streamlit-hello.py
```

Open your browser to [http://localhost:8501](http://localhost:8501) to view the app.

---

## 07-langchain-streamlit-rag-qa.py: Streamlit RAG QA App

This app lets you upload a `.txt` file and ask questions about its content using Retrieval-Augmented Generation (RAG) with LangChain, OpenAI, and ChromaDB. The app splits your document, creates embeddings, and answers your questions using GPT-4o.

**Requirements:**
- Python 3.8+
- `streamlit`, `langchain`, `langchain-openai`, `chromadb`, `openai` (see `requirements.txt`)
- **OpenAI API key** (for LLM and embeddings)

**Usage (local):**
```sh
pip install -r requirements.txt
streamlit run 07-langchain-streamlit-rag-qa.py
```

**Docker:**
```sh
docker build -t rag-examples .
docker run --rm -p 8501:8501 rag-examples streamlit run 07-langchain-streamlit-rag-qa.py
```

**Sample workflow:**
1. Upload a `.txt` file (e.g., an article or notes).
2. Enter your OpenAI API key (required for GPT-4o and embeddings).
3. Ask a question about the document.
4. View the concise, AI-generated answer!

Open your browser to [http://localhost:8501](http://localhost:8501) to use the app.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized runs)
- (For OpenAI example) An OpenAI API key with sufficient quota ([get yours here](https://platform.openai.com/account/api-keys))

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/pritamnikam/rag-langchain-architecture.git
   cd rag-langchain-architecture
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Run the TF-IDF Example (Local)
```sh
python 01-embeddings-tf-idf.py
```

### Run the OpenAI Embedding Example (Local)
Set your OpenAI API key as an environment variable:
```sh
# Linux/macOS
export OPENAI_API_KEY=your-key-here
# Windows (cmd)
set OPENAI_API_KEY=your-key-here
```
Then run:
```sh
python 02-openai-embedding.py
```

### Run the LangChain + ChromaDB Example (Local)
Set your OpenAI API key as an environment variable:
```sh
export OPENAI_API_KEY=your-key-here  # or set on Windows
```
Then run:
```sh
python 03-embeddings-langchain-chromadb.py
```

### Run with Docker
Build the Docker image (includes all scripts):
```sh
docker build -t rag-examples .
```

#### Run the TF-IDF example in Docker
```sh
docker run --rm rag-examples python 01-embeddings-tf-idf.py
```

#### Run the OpenAI Embedding example in Docker
```sh
docker run --rm -e OPENAI_API_KEY=your-key-here rag-examples python 02-openai-embedding.py
```

#### Run the LangChain + ChromaDB example in Docker
```sh
docker run --rm -e OPENAI_API_KEY=your-key-here rag-examples python 03-embeddings-langchain-chromadb.py
```

#### Run the LangChain Augmented Query example in Docker
```sh
docker run --rm rag-examples python 04-langchain-augmented-query.py
```

#### Run the LangChain Generation example in Docker
```sh
docker run --rm -e OPENAI_API_KEY=your-key-here rag-examples python 05-langchain-generation.py
```

**Note:** If you see an `insufficient_quota` or `invalid_api_key` error, check your OpenAI account status and quota.

## Features
- Converts a set of documents into TF-IDF or OpenAI embeddings
- Builds a simple vector database (index) using `NearestNeighbors` from scikit-learn or ChromaDB
- Demonstrates retrieval: Given a query, finds and displays the most similar document
- Shows how to use both local and API-powered embeddings for RAG
- Demonstrates modern RAG workflow with LangChain and ChromaDB
- Shows how to build and format a RAG prompt for LLMs using LangChain

## Example Output
### TF-IDF Example
```
----------------------------------------------------------------------------------------------------
This is how it looks after being vectorized:

TF-IDF Matrix:
             ai  available   course  ...      rag    using  writing
Doc 1  0.000000   0.000000  0.57735  ...  0.57735  0.00000  0.00000
Doc 2  0.344315   0.000000  0.00000  ...  0.00000  0.00000  0.00000
Doc 3  0.382743   0.485461  0.00000  ...  0.00000  0.00000  0.00000
Doc 4  0.000000   0.000000  0.00000  ...  0.00000  0.57735  0.57735

[4 rows x 15 columns]
NearestNeighbors(metric='cosine', n_neighbors=1)
Query Vector for 'What course is this?':
Retrieved document: This is the `Fundamentals of RAG course`
Nearest document index: 0
```

### OpenAI Embedding Example
```
----------------------------------------------------------------------------------------------------
This is how it looks after going through an embedding model:
[[...]]
Query: What is JS?
Retrieved document: JavaScript is a good programming language :)
```

### LangChain + ChromaDB Example
```
Generated 5 embeddings. Each vector size: 1536
Chroma vector store created with 5 documents.

Query: Which course is about RAG?
Top retrieved document: ['This is the Fundamentals of RAG course.']
```

### LangChain Augmented Query Example
```
[INFO] Custom RAG Prompt Template:
 input_variables=['context', 'question']

[INFO] Retrieved Context:
 AI continues to evolve rapidly, with advancements in deep learning and generative models. 
Experts predict increased integration of AI in everyday life and industry.

[INFO] Augmented Query (Final Prompt to LLM):
 Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

AI continues to evolve rapidly, with advancements in deep learning and generative models. 
Experts predict increased integration of AI in everyday life and industry.
Question: What is the future of AI?

Helpful Answer:
```

## Troubleshooting
- **OpenAI errors**: If you see errors about invalid API keys or insufficient quota, check your [OpenAI account usage](https://platform.openai.com/account/usage) and billing.
- **Docker issues**: Make sure Docker is running and you use the correct container/image names.
- **Windows line endings**: If you see warnings about LF/CRLF, you can safely ignore them for code execution.

## License
This project is for educational purposes and learning about RAG architectures.
