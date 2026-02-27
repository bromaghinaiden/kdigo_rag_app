import ingest
import rag_chain
import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage


# get data
data_filepath = r"kdigo_guidelines\KDIGO-2024-CKD-Guideline.pdf"

# load, chunk, embed, store as vector
# use st.cache_resource to ensure that vectorestore is only generated once
@st.cache_resource
def get_vectorstore():
    return ingest.load_vector_store(data_filepath)
vector_store = get_vectorstore()


# retreiver to augment response
retriever = vector_store.as_retriever()

# intialize rag chain
chain = rag_chain.build_rag_chain(retriever)

# initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ui
st.title("KDIGO Guidelines Assistant")

# display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    else:
        st.chat_message("ai").write(message.content)

question = st.chat_input("Ask a question about KDIGO guidelines")

if question:
    st.chat_message("human").write(question)
    
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