import ingest
import rag_chain
import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from embeddings import get_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

#  Access pinecone vector store
@st.cache_resource
def get_vectorstore():
    load_dotenv()
    embeddings = get_embeddings()
    return PineconeVectorStore.from_existing_index(
        index_name="kdigo-guidelines-dense-index",
        embedding=embeddings
    )

# Initialize pinecone vector store as retreiver for use in RAG chain
vector_store = get_vectorstore()
retriever = vector_store.as_retriever()

# intialize rag chain
chain = rag_chain.build_rag_chain(retriever)

# initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ui
st.title("KDIGO Guidelines Assistant")

# write disclaimer
st.sidebar.write("**Disclaimer:** This web app does not provide medical advice. This web app is built soley as an educational exercise in the implementation of RAG in LLMs. For medical advice regarding CKD, please consult a physician.")

# display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    else:
        st.chat_message("ai").write(message.content)

# Obtain input from user & run RAG to generate response
question = st.chat_input("Ask a question about KDIGO guidelines")
if question:
    st.chat_message("human").write(question)
    
    # Loop several times in case rate limit error is encountered
    max_retries = 3
    for attempt in range(max_retries):
        try:
            answer = rag_chain.run_rag(chain,
                                         question,
                                         st.session_state.chat_history
                                         )
            st.chat_message("ai").write(answer)
            
            # update chat history
            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.chat_history.append(AIMessage(content=answer))
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    st.info("Rate limited, retrying...")
                    time.sleep(20)
                else:
                    st.error("Rate limited. Please try again in a minute.")
            else:
                st.error(f"An error occurred: {str(e)}")
                break