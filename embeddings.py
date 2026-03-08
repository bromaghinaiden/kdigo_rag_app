from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Sole purpose of file is to define embeddings. 
    This is broken out into separate file since both the ingest.py and kdigo_rag_app.py files will need to access the embeddings
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    