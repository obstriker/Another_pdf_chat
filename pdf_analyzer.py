from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import streamlit as st
import os
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

PAPER_DIRECTORY = "./papers/"
load_dotenv()

# TODO: if file exists don't process it
# TODO: change embedding to universal sentence loader
# TODO: use faiss semantic search

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

st.title("PDF questions")


file = st.file_uploader("Upload a PDF file", accept_multiple_files=False, type=["pdf"])

loader = DirectoryLoader(PAPER_DIRECTORY, glob="./*.pdf", loader_cls=PyPDFLoader)

if file and not os.path.exists(file.name):
    with open(PAPER_DIRECTORY + file.name, "wb+") as f:
        f.write(file.read())

    # Load and process the text files
    # loader = TextLoader('single_text_file.txt')
    loader = DirectoryLoader(PAPER_DIRECTORY, glob="./*.pdf", loader_cls=PyPDFLoader)

    #loader = TextLoader("./papers/longtermeffects.md")
    documents = loader.load()

    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=texts, 
                                     embedding=embedding,
                                     persist_directory=persist_directory)

    # persiste the db to disk
    vectordb.persist()
    vectordb = None

query = st.text_input("Enter a question:")

if query:
    # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)              
    retriever = vectordb.as_retriever()
    
    chat = ChatOpenAI(temperature=0.9)
    
    
    docs = retriever.get_relevant_documents(query)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, verbose=True)
    answer = qa.run(query)

    st.write("Answer: ", answer)