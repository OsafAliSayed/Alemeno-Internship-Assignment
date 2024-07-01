from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from pypdf import PdfReader
import os
import glob


MODEL = "llama3:latest"
EMBEDDING_MODEL = "mxbai-embed-large"

def get_text_from_pdf():
    print("Importing PDFs...")
    # getting all pdfs
    DIR = "./PDFS"

    # Get a list of all pdf files in the specified directory
    pdf_files = glob.glob(os.path.join(DIR, '*.pdf'))
    
    text = ""
    for pdf_file in pdf_files:
        # Open the PDF file in read-binary mode
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    print("Creating chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, embedding_model=EMBEDDING_MODEL):  
    print("Adding chunks to vector store...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain(model=MODEL):
    print("making conversational chain...")
    prompt_template = """
    I am providing you with relevant information from companies 10-k filings to help you with your question. Information only includes context about three companies namely Google(Alphabet Inc.), Tesla, and Uber.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Irrelevant Context!", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    """
    model = Ollama(
        model=model,
        temperature = 0.5,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def user_input(user_question, embedding_model=EMBEDDING_MODEL):
    embeddings = OllamaEmbeddings(model=embedding_model)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"context": docs, "question": user_question})
    print(response['text'], end="")
    print()
    
def main():
    print("In main!")
    inp = ""
    while inp != "exit":
        user_question = input("Enter your question: ")
        user_input(user_question)
        inp = input("Press enter to ask another question or type 'exit' to exit: ")

if __name__ == "__main__":
    if not os.path.exists("./faiss_index"):
        text = get_text_from_pdf()
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)
    main()  
