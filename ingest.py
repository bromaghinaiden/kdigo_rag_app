from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from embeddings import get_embeddings
import sys

def load_document(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs

def chunk_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

def ingest_to_pinecone(filepath):
    load_dotenv()
    docs = load_document(filepath)
    chunks = chunk_document(docs)
    embeddings = get_embeddings()

    PineconeVectorStore.from_documents(index_name='kdigo-guidelines-dense-index',
                                       documents=chunks,
                                       embedding=embeddings)
    
    print(f"Successfully upserted {len(chunks)} chunks from {filepath}")
    
if __name__ == "__main__":
    for filepath in sys.argv[1:]:
        ingest_to_pinecone(filepath)