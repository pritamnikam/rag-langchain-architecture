import os
import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

import numpy as np
from sklearn.neighbors import NearestNeighbors

# We define a function get_gpt4_embedding that takes a text input and uses the OpenAI API 
# to generate an embedding using the specified model. The function returns the embedding 
# as a list of numerical values. Next, we create a list of sample documents that we want 
# to index and generate embeddings for each document using the function we defined earlier. 
# These embeddings will be used to index the documents for retrieval.

def get_gpt4_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    # Access the embedding directly from the response object
    return response.data[0].embedding

# List of example documents to be used in the index
documents = [
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language :)"
]

# Get embeddings for each document using the get_gpt4_embedding function
embeddings = [get_gpt4_embedding(doc) for doc in documents]
embeddings = np.array(embeddings)
print("--" * 50)
print("This is how it looks after going through an embedding model:\n")
print(embeddings)

# We then fit a NearestNeighbors model on the generated embeddings. This model will 
# allow us to perform efficient similarity searches to find the most relevant documents. 
# We also define a function to query the NearestNeighbors model. This function will 
# convert a query into an embedding and use the model to find and return the most 
# relevant document.

# Fit a NearestNeighbors model on the document embeddings using cosine similarity
index = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

# Function to query the index with a given text query
def query_index(query):
    query_embedding = get_gpt4_embedding(query)
    query_embedding = np.array([query_embedding])
    distance, indices = index.kneighbors(query_embedding)
    return documents[indices[0][0]]

# Example Query
query = "What is JS?"
print("Query:", query)
result = query_index(query) # Retrieve the most similar document to the query

print("Retrieved document:", result) # Print the retrieved document