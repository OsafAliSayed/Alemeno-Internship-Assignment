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

## UI Design and Sample Queries
UI Design features a sidebar that shows all the text that was taken as context. Each element is a chunk of size 256 with an overlap of 20. For the given queries I took four relevant contexts.
Here, are the response to the given questions:

![image](https://github.com/OsafAliSayed/Alemeno-Internship-Assignment/assets/99737087/a9bd705b-d804-4137-a6ff-88e7f7100d48)

![image](https://github.com/OsafAliSayed/Alemeno-Internship-Assignment/assets/99737087/0a3ed295-6317-42cc-9498-edd6a565f267)

![image](https://github.com/OsafAliSayed/Alemeno-Internship-Assignment/assets/99737087/a2d17b40-3402-4dc4-8719-4cc6953a9edb)

We can take more chunks as reference to get detailed answers for same questions, as shown below:

![image](https://github.com/OsafAliSayed/Alemeno-Internship-Assignment/assets/99737087/606dc421-17c8-4ae3-9398-103cb621521a)


## Development

1. I have extracted text from the PDF by using PyPDF. (Probably should use Unstructureed as their are a lot of tables that need to be read properly)
2. The text was then divided into smaller chunks so that the information can be retained properly (I have tried huge chunks such as 10000, but the context was lost completely). These chunks are created by using a technique known as recursive chunking.
3. Using a "mxbai-embed-large" as embedding model, I generated embedding vectors for the text extracted from the PDFs.
4. This embedded vectors are then stored in FAISS vector store. This database is saved locally, so that We do not have to scan to PDF's again and again
5. This database is accessed again to give context to LLM model so that it can answer the relevant queries.


## Prerequisites

Use Ollama to download embedding and LLM models easily. Visit [Ollama official page](https://ollama.com/download). Make sure Ollama is running and you have downloaded the relevant models i.e. llama3 and mxbai-embed-large from ollama website.

## How to setup

1. Clone the Github repo ```git clone https://github.com/OsafAliSayed/Alemeno-Internship-Assignment/```
2. Create a python virtual environment in the project repository ```python -m venv venv```.
3. access the environment using terminal ```venv/Scripts/activate``` for windows and ```venv/bin/activate``` for linux.
4. Run ```pip install -r requirements.txt```.
5. Finally run ```streamlit run frontend.py``` to launch the application.
