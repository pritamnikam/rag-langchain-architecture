# TF-IDF Embeddings & Simple RAG Demo

This project demonstrates how to use the TF-IDF vectorization technique to convert a list of documents into numerical vectors using Python, scikit-learn, and pandas. It also includes a simple Retrieval-Augmented Generation (RAG) demonstration using NearestNeighbors as a conceptual vector database for document retrieval.

## Project Structure
- `01-embeddings-tf-idf.py`: Main script that performs TF-IDF vectorization, displays the results in a DataFrame, and demonstrates basic retrieval using NearestNeighbors.
- `02-openai-embedding.py`: Example using OpenAI's embedding API and NearestNeighbors for semantic search. Requires an OpenAI API key.
- `requirements.txt`: Lists Python dependencies for the project.
- `Dockerfile`: Enables running the app in a containerized environment.

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized runs)

### Install dependencies
```sh
pip install -r requirements.txt
```

### Run the TF-IDF example
```sh
python "01-embeddings-tf-idf.py"
```

### Run the OpenAI Embedding example
Set your OpenAI API key as an environment variable:
```sh
export OPENAI_API_KEY=your-key-here  # On Windows use: set OPENAI_API_KEY=your-key-here
```
Then run:
```sh
python "02-openai-embedding.py"
```

### Run with Docker
Build the Docker image:
```sh
docker build -t rag-examples .
```

#### Run the TF-IDF example in Docker
```sh
docker run --rm rag-examples python 01-embeddings-tf-idf.py
```

#### Run the OpenAI Embedding example in Docker
You must provide your OpenAI API key as an environment variable:
```sh
docker run --rm -e OPENAI_API_KEY=your-key-here rag-examples python 02-openai-embedding.py
```

## Features
- Converts a set of documents into TF-IDF vectors
- Displays the TF-IDF matrix in a readable format
- Builds a simple vector database (index) using `NearestNeighbors` from scikit-learn
- Demonstrates retrieval: Given a query, finds and displays the most similar document

## Example Retrieval Output
```
NearestNeighbors(metric='cosine', n_neighbors=1)
Query Vector for 'What course is this?':
[[...]]
Nearest document index: 0
Distance from query: ...
Retrieved document: This is the `Fundamentals of RAG course`
```

## License
This project is for educational purposes.
