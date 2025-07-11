"""
08-langchain-qdrant-vector-store-rag-qa.py

A minimal Retrieval-Augmented Generation (RAG) pipeline using LangChain, Qdrant vector store, and OpenAI LLMs.
Loads a text file, splits it into chunks, embeds with OpenAI, stores in Qdrant (in-memory), and runs a RAG QA chain.
"""

# --- Imports ---
from langchain_community.document_loaders import TextLoader  # For loading plain text documents
from langchain_openai import OpenAIEmbeddings  # For creating OpenAI embeddings
from langchain_qdrant import Qdrant  # Qdrant vector store integration
from langchain_text_splitters import CharacterTextSplitter  # For chunking text
from langchain_core.prompts import PromptTemplate  # For custom prompt templates
from langchain_core.runnables import RunnablePassthrough  # For chaining inputs
from langchain_core.output_parsers import StrOutputParser  # For parsing LLM output
from langchain_openai import ChatOpenAI  # For OpenAI LLMs

# --- Step 1: Create Embeddings ---
embeddings = OpenAIEmbeddings()

# --- Step 2: Load and Split Document ---
loader = TextLoader("Lakers.txt")  # Loads text file (one document)
documents = loader.load()  # Returns list of Document objects
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Chunk size and overlap can be tuned
docs = text_splitter.split_documents(documents)  # Split into overlapping chunks

# --- Step 3: Create Qdrant Vector Store ---
# Stores the document chunks as vectors in an in-memory Qdrant collection
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Use in-memory Qdrant DB (no persistent storage)
    collection_name="my_documents",
)

# --- Step 4: Set Up Retriever ---
retriever = qdrant.as_retriever()  # Converts Qdrant store into a retriever interface

# --- Step 5: Define a Custom RAG Prompt ---
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say \"thanks for asking!\" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# --- Step 6: Initialize OpenAI LLM ---
llm = ChatOpenAI(model="gpt-4o")  # Requires your OpenAI API key in env var OPENAI_API_KEY

# --- Step 7: Build the RAG Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# --- Step 8: Run a Sample Query ---
result = rag_chain.invoke("Who joined Lakers in 2018?")
print(result)  # Output the answer