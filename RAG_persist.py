import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA
import tempfile

# Config
CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'model_name': 'all-MiniLM-L6-v2',
    'llm_model': 'gemini-1.5-flash',
    'max_file_size_mb': 10,
    'max_chunks': 500,
}

def main():
    load_dotenv()
    
    # Validate API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Validate file path
    file_path = input("Enter PDF file path: ").strip()
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        sys.exit(1)
    
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=CONFIG["llm_model"], api_key=api_key)
        
        # Load and process PDF
        print("Loading PDF...")
        loader = PyPDFLoader(file_path)
        pdf_data = loader.load()
        print(f"Loaded {len(pdf_data)} pages")
        
        # Split text
        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap']
        )
        text_data = text_splitter.split_documents(pdf_data)
        print(f"Created {len(text_data)} chunks")
        
        # Create embeddings and vector store
        print("Creating embeddings (this may take a while)...")
        embeddings = HuggingFaceEmbeddings(model_name=CONFIG['model_name'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = Chroma.from_documents(
                documents=text_data,
                embedding=embeddings,
                persist_directory=temp_dir
            )
            
            retriever = vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False
            )
            
            print("\n🤖 Ready to answer questions! Type 'exit' to quit.\n")
            
            while True:
                query = input("❓ Your question: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                if not query:
                    print("Please enter a question.")
                    continue
                    
                try:
                    print("🤔 Thinking...")
                    response = qa_chain({"query": query})
                    print(f"✅ Answer: {response['result']}\n")
                except Exception as e:
                    print(f"❌ Error: {e}\n")
            del vector_store
    
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()