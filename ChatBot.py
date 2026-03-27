import streamlit as st

from PyPDF2 import PdfReader

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.header("AskDocs")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

# extracting the text from pdf file
if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text() or ""  # safer extraction

    # break it into Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
    chunks = splitter.split_text(text)

    # Use HuggingFaceEmbeddings for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # You can change to any supported model
    )

    # Creating VectorDB & Storing embeddings into it
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user query
    user_query = st.text_input("Type your query here")

    # semantic search from vector store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # define our LLM using Groq
        # ...existing code...
        # define our LLM using Groq
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            # max_tokens=1000,
            temperature=0.6
        )

        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor. Answer the question based on the following context and if you did not get the context simply say "I don't know" : {context} Question: {input} """
        )

        chain = create_stuff_documents_chain(llm, customized_prompt)
        output = chain.invoke({"input": user_query, "context": matching_chunks})

        st.write(output)
