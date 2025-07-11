import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# --- Streamlit UI Setup ---
# Display a logo or banner if the image exists (prevents errors if missing)
import os
if os.path.exists('Educative.png'):
    st.image('Educative.png', use_column_width=True)

# Add a title and description
st.title(' LangChain RAG QA Demo')
st.write('''
Upload a text file and ask questions about its content. This demo uses LangChain, OpenAI, and Chroma for Retrieval-Augmented Generation (RAG) in a Streamlit web app.
''')

# Sidebar for info and instructions
with st.sidebar:
    st.header('Instructions')
    st.markdown('''
    1. Upload a `.txt` article or document.
    2. Enter your OpenAI API key.
    3. Type a question about the uploaded document.
    4. Submit to get an AI-generated answer based on your file!
    ''')
    st.markdown('---')
    st.info('Your API key and files are never stored.')
    st.caption('Powered by LangChain • OpenAI • Chroma • Streamlit')

def format_docs(docs):
    # Join the page content of each document with a newline separator
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.create_documents(documents)
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        # Select embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        database = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = database.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
)
        # Create QA chain
        response = rag_chain.invoke(query_text)
        return response

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = None
with st.form('myform', clear_on_submit=False):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result = response
if result:
    st.info(result)