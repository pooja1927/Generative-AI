from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.please response to the user queries"),
        ("user","Question:{question}")
    ]
)

##streamlit framework

st.title("Langchain demo with Ollama")
input_text= st.text_input("search the topic u want")

#openAI llm
llm= Ollama(model="llama2")
output_parser= StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))