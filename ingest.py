from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


'''
Document loader
Document chunking
Embed chunks
Store in vector store
Build retreiver
'''

def load_document(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs

def chunk_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks


def store_as_vector(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return vector_store


# main function for developing file, will switch to load_vector_store() and call in app.py file later
def load_vector_store(filepath):
    # Main function to drive script
    docs = load_document(filepath)
    chunks = chunk_document(docs)
    vector_store = store_as_vector(chunks)
    return vector_store