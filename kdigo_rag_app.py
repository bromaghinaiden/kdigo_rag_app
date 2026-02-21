import ingest
import streamlit as st
import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate




# load api key
load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")


# get data
data_filepath = r"kdigo_guidelines\KDIGO-2024-CKD-Guideline.pdf"


# load, chunk, embed, store as vector
# use st.cache_resource to ensure that vectorestore is only generated once
@st.cache_resource
def get_vectorstore():
    return ingest.load_vector_store(data_filepath)
vector_store = get_vectorstore()

# prep llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# prep chain
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_template("""
You are a helpful clinical assistant that answers questions about KDIGO clinical guidelines for CKD management. 
Do not provide answers about other topics, only state that you cannot answer questions not relevant to KDIGO guidlines for CKD management.
Use only the context provided to answer the question. If you don't know, say so. Do not hallucinate. Do not provide wrong information.

Context: {context}
Question: {input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


st.title("KDIGO Guidelines Assistant")

question = st.text_input("Ask a question about KDIGO guidelines")

if question:
    response = rag_chain.invoke({"input": question})
    st.write(response["answer"])

