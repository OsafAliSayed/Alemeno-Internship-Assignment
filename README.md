# 10-k Filings Analyzer using LLM 

## Description
This is a local LLM model trained on 10-k filings data from Google(Alphabet Inc.), Uber, and Tesla. The model lets user ask questions relevant to these filings. 

## Setup 
For setup I have used the following:

1. Backend: **Langchain V0.2**
2. Frontend: **Streamlit**
3. Vector Store: **FAISS**
4. Embedding Model: **mxbai-embed-large**
5. LLM Model: **Llama3:latest**

## Screenshots

TODO

## Development

1. I have extracted text from the PDF by using PyPDF. (Probably should use Unstructureed as their are a lot of tables that need to be read properly)
2. The text was then divided into smaller chunks so that the information can be retained properly (I have tried huge chunks such as 10000, but the context was lost completely). These chunks are created by using a technique known as recursive chunking.
3. Using a "mxbai-embed-large" as embedding model, I generated embedding vectors for the text extracted from the PDFs.
4. This embedded vectors are then stored in FAISS vector store. This database is saved locally, so that We do not have to scan to PDF's again and again
5. This database is accessed again to give context to LLM model so that it can answer the relevant queries.


## Prerequisites

Use Ollama to download embedding and LLM models easily. Visit [Ollama official page](https://ollama.com/download).
## How to setup

1. Clone the Github repo ```git clone https://github.com/OsafAliSayed/Alemeno-Internship-Assignment/```
2. Create a python virtual environment in the project repository ```python -m venv venv```.
3. access the environment using terminal ```venv/Scripts/activate``` for windows and ```venv/bin/activate``` for linux.
4. Run ```pip install -r requirements.txt```.
5. Finally run ```streamlit run frontend.py``` to launch the application.
