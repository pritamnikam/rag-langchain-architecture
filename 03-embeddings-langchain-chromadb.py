import os
import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Documents to embed and store
documents = [
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language"
]

# Step 1: Generate embeddings for the documents
embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(documents)
print(f"Generated {len(embeddings)} embeddings. Each vector size: {len(embeddings[0])}")

# Step 2: Create a Chroma vector store from the documents and embeddings
vectorstore = Chroma.from_texts(documents, embeddings_model)
print(f"Chroma vector store created with {vectorstore._collection.count()} documents.")

# Step 3: Configure the retriever to use similarity search (top 1 result)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Step 4: Perform a similarity search with a sample query
query = "Which course is about RAG?"
result = retriever.invoke(query)
print("\nQuery:", query)
print("Top retrieved document:", result)

# Additional example documents to be used in the database
additional_documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
]

# Create a Chroma database from the documents using OpenAI embeddings
db = Chroma.from_texts(documents, OpenAIEmbeddings())
print(db)


# Configure the database to act as a retriever, setting the search type to
# similarity and returning the top 1 result
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)


# Perform a similarity search with the given query
result = retriever.invoke("Where can I see Mona Lisa?")
print(result)