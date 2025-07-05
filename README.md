# RAG with LangChain: In-Depth Architecture Examples

This project demonstrates two practical approaches for Retrieval-Augmented Generation (RAG) using Python:
- **TF-IDF Embedding + NearestNeighbors**: Classic local vectorization and retrieval.
- **OpenAI Embedding API + NearestNeighbors**: Modern semantic search using OpenAI's embedding models.

Both examples show how to build a simple vector database and perform document retrieval, laying the foundation for more advanced RAG systems or LangChain integrations.

## Project Structure
- `01-embeddings-tf-idf.py`: TF-IDF vectorization and retrieval demo (no API key needed).
- `02-openai-embedding.py`: OpenAI Embedding API-based retrieval demo (requires OpenAI API key).
- `03-embeddings-langchain-chromadb.py`: LangChain + OpenAI Embeddings + ChromaDB vector store demo (requires OpenAI API key).
- `requirements.txt`: Python dependencies for all examples.
- `Dockerfile`: Containerizes all scripts for easy, reproducible runs.
- `.gitignore`: Standard Python, Docker, and editor ignores.

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

**Note:** If you see an `insufficient_quota` or `invalid_api_key` error, check your OpenAI account status and quota.

## Features
- Converts a set of documents into TF-IDF or OpenAI embeddings
- Builds a simple vector database (index) using `NearestNeighbors` from scikit-learn or ChromaDB
- Demonstrates retrieval: Given a query, finds and displays the most similar document
- Shows how to use both local and API-powered embeddings for RAG
- Demonstrates modern RAG workflow with LangChain and ChromaDB

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

## Troubleshooting
- **OpenAI errors**: If you see errors about invalid API keys or insufficient quota, check your [OpenAI account usage](https://platform.openai.com/account/usage) and billing.
- **Docker issues**: Make sure Docker is running and you use the correct container/image names.
- **Windows line endings**: If you see warnings about LF/CRLF, you can safely ignore them for code execution.

## License
This project is for educational purposes and learning about RAG architectures.
