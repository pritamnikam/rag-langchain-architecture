# TF-IDF Embeddings: use the TF-IDF vectorization technique

# List of documents to be processed
documents = [
    "This is the `Fundamentals of RAG course`",
    "Educative is an AI-powered online learning platform",
    "There are several Generative AI courses available on Educative",
    "I am writing this using my keyboard"
]

# Importing the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Initializing TfidfVectorizer to remove English stop words
vectorizer = TfidfVectorizer(stop_words='english')

# Transforming the documents into a matrix of TF-IDF features
tfidf_matrix = vectorizer.fit_transform(documents)

# Create a DataFrame to display the TF-IDF matrix more readably
feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=[f"Doc {i+1}" for i in range(len(documents))])

# Print the TF-IDF matrix using DataFrame for better formatting
print("--" * 50)
print("This is how it looks after being vectorized:\n")
print("TF-IDF Matrix:")
print(df)

# Importing NearestNeighbors from sklearn for creating the nearest neighbor index
# This module is used to efficiently find the closest vector(s) in high-dimensional space, which is
# crucial for the retrieval functionality in our RAG system
from sklearn.neighbors import NearestNeighbors

# Initializing NearestNeighbors to create a conceptual vector database (index) for the RAG system
# This index, utilizing cosine similarity, functions effectively as the vector database,
# storing all document vectors and enabling their retrieval based on similarity measures
index = NearestNeighbors(n_neighbors=1, metric='cosine').fit(tfidf_matrix)
print(index)

# Function to query the index with a new document/query
def query_index(query):
    # Transforming the query into the same TF-IDF vector space as the documents
    query_vec = vectorizer.transform([query])
    print(f"Query Vector for '{query}':")
    print(query_vec.toarray())
    
    # Finding the nearest neighbor to the query vector in the index
    distance, indices = index.kneighbors(query_vec)
    print("Nearest document index:", indices[0][0])
    print("Distance from query:", distance[0][0])
    
    return documents[indices[0][0]]

# Example query to test the indexing and retrieval system
query = "What course is this?"
result = query_index(query)

# Printing the document retrieved as the closest match to the query
print("--" * 50)
print("Retrieved document:", result)