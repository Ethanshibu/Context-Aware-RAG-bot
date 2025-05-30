import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA 


load_dotenv()

st.set_page_config(page_title="Context Aware Chatbot")
st.title("Context Aware Chatbot")

if "vector_store" in st.session_state:
    if os.getenv("GOOGLE_API_KEY"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        retriever = st.session_state.vector_store.as_retriever()
        st.session_state.retriver = retriever
        st.success("llm and retriever initialized successfully!")
    else:
        st.error("Please set the GOOGLE_API_KEY in your .env file to use the Google LLM.")


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file_name = uploaded_file.name
    save_path = os.path.join(".", file_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing the PDF..."):
        pdf_loader = PyPDFLoader(save_path)
        pdf = pdf_loader.load()
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        text = txt_splitter.split_documents(pdf)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(
            documents=text,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        st.session_state.vector_store = vector_store

    st.success(f"File '{file_name}' uploaded and saved successfully!")
   # st.info(f"You can find it at: {os.path.abspath(save_path)}")
 
else:
    if "vector_store" in st.session_state:
        st.success("Vector store already uploaded and ready to chat")
    elif os.path.exists("./chroma_db"):
        with st.spinner("Uploading existing Knowledge base"):
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            st.session_state.vector_store = vector_store
        st.success("Existing Knowledge base loaded successfully!")
    else:
        st.write("Upload a pdf to start chatting.")