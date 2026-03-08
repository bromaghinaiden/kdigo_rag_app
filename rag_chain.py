import os 
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Prepare LLM for use in RAG chain
def initialize_llm():
    # load api key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")    

    # prep llm
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key)
    return llm

# Prepare prompts that will be used in chains
def initialize_prompts():
    # Use to summarize chat history and question before retrieving from vector store
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the conversation above, rephrase my question into a standalone search query that captures the full context")
    ])

    # main rag prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful clinical assistant that answers questions about KDIGO clinical guidelines for CKD management. 
    Do not provide answers about other topics, only state that you cannot answer questions not relevant to KDIGO guidelines for CKD management.
    Use only the context provided to answer the question. If you don't know, say so. Do not hallucinate.

    Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    return rephrase_prompt, rag_prompt



def build_rag_chain(retriever):
    llm = initialize_llm()
    rephrase_prompt, rag_prompt = initialize_prompts()
     # initialize history aware retreiver. this will rephrase the question from user in the context of the chat history,
     # then retrieve relevant chunks
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    # history_aware_retreiver runs first to rephrase the prompt and retrieve chunks, 
    # then combine_docs_chain runs the chunks + question and chat history through the llm to generate a response
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)


# driver function
def run_rag(rag_chain, question, chat_history):
    return rag_chain.invoke({"input": question, 
                            "chat_history":chat_history})['answer']