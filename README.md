# TF-IDF Embeddings Project

This project demonstrates how to use the TF-IDF vectorization technique to convert a list of documents into numerical vectors using Python, scikit-learn, and pandas.

## Project Structure
- `01-embeddings-tf-idf.py`: Main script that performs TF-IDF vectorization and displays the results in a DataFrame.
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

## License
This project is for educational purposes.
