# TF-IDF Embeddings & Simple RAG Demo

This project demonstrates how to use the TF-IDF vectorization technique to convert a list of documents into numerical vectors using Python, scikit-learn, and pandas. It also includes a simple Retrieval-Augmented Generation (RAG) demonstration using NearestNeighbors as a conceptual vector database for document retrieval.

## Project Structure
- `01-embeddings-tf-idf.py`: Main script that performs TF-IDF vectorization, displays the results in a DataFrame, and demonstrates basic retrieval using NearestNeighbors.
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

### Run the script
```sh
python "01-embeddings-tf-idf.py"
```

### Run with Docker
Build the Docker image:
```sh
docker build -t tfidf-app .
```
Run the container:
```sh
docker run --rm tfidf-app
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
