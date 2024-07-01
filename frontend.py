import streamlit as st
from backend import get_text_from_pdf, get_text_chunks, get_vector_store, get_conversational_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os 

EMBEDDING_MODEL = "mxbai-embed-large"

def chatbot_interface():
    st.title("10-k Filings Chatbot")
    user_input = st.text_input("User Input", "")
    
    # Process user input and generate chatbot response
    chatbot_response = ""
    if user_input != "":
        process_user_input(user_input)

def process_user_input(user_question, embedding_model=EMBEDDING_MODEL):
    
    embeddings = OllamaEmbeddings(model=embedding_model)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=10)
    chain = get_conversational_chain()
    response = chain.invoke({"context": docs, "question": user_question})
    print(response['text'], end="")
    st.write("Chatbot Response: ", response['text'])
    
    # write in sidebar
    st.sidebar.write("Context ", response['context'])


if __name__ == "__main__":
    if not os.path.exists("./faiss_index"):
        text = get_text_from_pdf()
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)

    chatbot_interface()