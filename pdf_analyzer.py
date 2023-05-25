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
import os
import streamlit as st
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

PAPER_DIRECTORY = "./papers/"
load_dotenv()

# TODO: change embedding to universal sentence loader
# TODO: use faiss semantic search

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()
loader = DirectoryLoader(PAPER_DIRECTORY, glob="./*.pdf", loader_cls=PyPDFLoader)

st.title("PDF questions")
file = st.file_uploader("Upload a PDF file", accept_multiple_files=False, type=["pdf"])

if (file or os.getenv('DEBUG_EMBED') == 'y') and not os.path.exists(file.name):
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


if os.getenv('DEBUG_MODE') == "y":
    query = os.getenv('DEBUG_QUERY')
else:
    query = st.text_input("Enter a question:")

if query or os.getenv('DEBUG_MODE') == "y":
    # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)              
    retriever = vectordb.as_retriever()
    
    chat = ChatOpenAI(temperature=0.9)
    
    docs = retriever.get_relevant_documents(query)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, verbose=True)
    answer = qa.run(query)

    if os.getenv('DEBUG_MODE') == "y":
        print("Answer: ", answer)
    else:
        st.write("Answer: ", answer)