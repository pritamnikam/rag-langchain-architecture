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