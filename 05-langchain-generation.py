"""
05-langchain-generation.py
Demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain and OpenAI's GPT model.
This script is fully commented for educational clarity.
"""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

# --- 1. Define a custom RAG prompt template ---
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

# Create a PromptTemplate object from the template string
custom_rag_prompt = PromptTemplate.from_template(template)
print("\n[INFO] Custom RAG Prompt Template Loaded.")

# --- 2. Simulate a retriever (replace with real retriever in production) ---
class MockRetriever:
    """
    Mock retriever for demonstration purposes.
    Replace with actual retrieval logic (e.g., Chroma, FAISS, etc.) for real use.
    """
    def invoke(self, query):
        # In a real application, this would return relevant context for the query.
        return (
            "AI continues to evolve rapidly, with advancements in deep learning and generative models.\n"
            "Experts predict increased integration of AI in everyday life and industry."
        )

retriever = MockRetriever()

# --- 3. User question and context retrieval ---
question = "What is the future of AI?"
context = retriever.invoke(question)
print("\n[INFO] Retrieved Context:\n", context)

# --- 4. Format the prompt with context and question ---
augmented_query = custom_rag_prompt.format(context=context, question=question)
print("\n[INFO] Augmented Query (Final Prompt to LLM):\n", augmented_query)

# --- 5. Initialize the language model ---
# Ensure the OPENAI_API_KEY environment variable is set
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable not set. Please set it to use OpenAI models.")

llm = ChatOpenAI(model="gpt-4o")

# --- 6. Build the RAG chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Pass context and question
    | custom_rag_prompt                                         # Format prompt
    | llm                                                      # Generate response
    | StrOutputParser()                                        # Parse output to string
)

# --- 7. Invoke the chain and print the response ---
try:
    response = rag_chain.invoke(question)
    print("\n[INFO] LLM Response:\n", response)
except Exception as e:
    print("[ERROR] Failed to get response from LLM:", str(e))