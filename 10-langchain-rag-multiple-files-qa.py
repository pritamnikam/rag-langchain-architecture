"""
10-langchain-rag-multiple-files-qa.py

A Streamlit web app for Retrieval-Augmented Generation (RAG) QA over multiple uploaded PDF or TXT documents.
Allows users to upload several files, combines and chunks their content, and runs a RAG pipeline using LangChain, Chroma, and OpenAI APIs.
"""

import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import fitz  # PyMuPDF for PDF reading

# --- UI: Display a logo or banner if present ---
image_filename = 'Educative.png'
if image_filename:
    st.image(image_filename, use_column_width=True)


def format_docs(docs):
    """
    Format a list of LangChain Document objects into a single string for the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def read_pdf(file):
    """
    Extract all text from a PDF file-like object using PyMuPDF.
    """
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def generate_response(uploaded_files, openai_api_key, query_text):
    """
    Given a list of uploaded files, OpenAI API key, and user query, run a RAG QA pipeline:
    - Extract text from each file (PDF or TXT)
    - Split all documents into chunks
    - Embed and store in Chroma vector store
    - Retrieve relevant chunks
    - Run LLM to answer the question
    """
    documents = []
    for uploaded_file in uploaded_files:
        # --- Step 1: Read document content ---
        if uploaded_file.type == "application/pdf":
            document_text = read_pdf(uploaded_file)
        else:
            document_text = uploaded_file.read().decode()
        documents.append(document_text)

    # --- Step 2: Split all documents into chunks ---
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = []
    for document in documents:
        texts.extend(text_splitter.create_documents([document]))

    # --- Step 3: Initialize LLM and Embeddings ---
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    # --- Step 4: Create a vector store (Chroma) ---
    database = Chroma.from_documents(texts, embeddings)
    retriever = database.as_retriever()

    # --- Step 5: Get a RAG prompt template from LangChain Hub ---
    prompt = hub.pull("rlm/rag-prompt")

    # --- Step 6: Build and run the RAG chain ---
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(query_text)
    return response

# --- Streamlit UI: File Upload and Query ---
uploaded_files = st.file_uploader(
    'Upload one or more articles',
    type=['txt', 'pdf'],
    accept_multiple_files=True
)
query_text = st.text_input(
    'Enter your question:',
    placeholder='Please provide a short summary.',
    disabled=not uploaded_files
)

# --- Streamlit Form for API Key and Submission ---
result = None
with st.form('myform', clear_on_submit=False, border=False):
    openai_api_key = st.text_input(
        'OpenAI API Key',
        type='password',
        disabled=not (uploaded_files and query_text)
    )
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, openai_api_key, query_text)
            result = response

# --- Display the Result ---
if result:
    st.info(result)