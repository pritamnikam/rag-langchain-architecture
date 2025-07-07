from langchain_core.prompts import PromptTemplate

# --- Step 1: Define a custom RAG prompt template ---
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

# Create a PromptTemplate object from the template string
custom_rag_prompt = PromptTemplate.from_template(template)
print("\n[INFO] Custom RAG Prompt Template:\n", custom_rag_prompt)

# --- Step 2: Simulate a retriever for demo purposes ---
# In a real RAG pipeline, 'retriever' would fetch relevant documents from a vector store.
# Here, we use a mock retriever for demonstration so the script is self-contained.
class MockRetriever:
    def invoke(self, query):
        # This would be replaced by actual retrieval logic (e.g., Chroma, FAISS, etc.)
        # For demonstration, return a static context string
        return "AI continues to evolve rapidly, with advancements in deep learning and generative models. \nExperts predict increased integration of AI in everyday life and industry."

retriever = MockRetriever()

# --- Step 3: Define the user question and retrieve context ---
question = "What is the future of AI?"
context = retriever.invoke(question)  # Retrieve context based on the question
print("\n[INFO] Retrieved Context:\n", context)

# --- Step 4: Format the prompt with context and question ---
augmented_query = custom_rag_prompt.format(context=context, question=question)
print("\n[INFO] Augmented Query (Final Prompt to LLM):\n", augmented_query)