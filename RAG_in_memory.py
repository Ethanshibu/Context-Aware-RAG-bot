import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import tempfile

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

file_path = "RAG for Knowledge-Intensive NLP Tasks.pdf"
loader = PyPDFLoader(file_path)
pdf_data = loader.load()
print(f"PDF data loaded successfully. Size: {len(pdf_data)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text_data = text_splitter.split_documents(pdf_data)
print(f"Text data split into {len(text_data)} chunks.")
print(f"First chunk: {text_data[0].page_content[:100]}...")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Choose persistence option:
use_persistence = True  # Set to False for in-memory only

if use_persistence:
    temp_dir_context = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_context.name  # Get the directory path
    persist_directory = temp_dir
else:
    persist_directory = None  # In-memory only

try:
    vector_store = Chroma.from_documents(
        documents=text_data,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    retriever = vector_store.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    chat_history = []
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = qa_chain({"question": query, "chat_history": chat_history})
        print(f"Response: {response['answer']}")
        chat_history.append((query, response['answer']))
    del vector_store

except Exception as e:
    print(f"Fatal error: {e}")

finally:
    if use_persistence:
        temp_dir_context.cleanup()  # Clean up the temporary directory