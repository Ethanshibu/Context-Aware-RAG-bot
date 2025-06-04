import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA
import tempfile


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    api_key=os.getenv("GOOGLE_API_KEY")
)


file_path = "RAG for Knowledge-Intensive NLP Tasks.pdf"
loader = PyPDFLoader(file_path)
pdf_data = loader.load()
print(f"PDF data loaded successfully. Size: {len(pdf_data)} bytes")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text_data = text_splitter.split_documents(pdf_data)
print(f"Text data split into {len(text_data)} chunks.",end='\n')
print(f"First chunk: {text_data[0].page_content[:100]}...",end='\n')

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
with tempfile.TemporaryDirectory() as temp_dir:
    vector_store = Chroma.from_documents(
        documents=text_data,
        embedding=embeddings,
        persist_directory=None
    )
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = qa_chain({"query": query})
        print(f"Response: {response['result']}")
    del vector_store
