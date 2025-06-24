import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

if "vectors" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = WebBaseLoader("https://python.langchain.com/docs/tutorials/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc, st.session_state.embeddings)

st.title("Chatgroq Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
"""
)

chain = prompt | llm
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, chain)

query = st.text_input("Input your Query here")

if query:
    response = retriever_chain.invoke({"input": query})
    st.write(response["answer"].content)
