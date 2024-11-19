from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from langchain import LangChain  # Asegúrate de importar correctamente
import ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.base import StuffDocumentsChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma  # O FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings  # O Hugging Face
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

class CustomWrapper:
    def __init__(self):
        self.langchain = LangChain()
        self.ollama = ollama.Ollama()

    def process_csv(self, file):
        # Leer el archivo CSV
        df = pd.read_csv(file)
        
        # Procesar el DataFrame con LangChain
        # Aquí puedes agregar la lógica específica que necesitas
        processed_data = self.langchain.process_dataframe(df)
        
        # Usar Ollama para alguna tarea adicional
        ollama_response = self.ollama.some_method(processed_data)
        
        return ollama_response

wrapper = CustomWrapper()

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    try:
        # Usa el Wrapper para procesar el archivo CSV
        processed_data = wrapper.process_csv(file.file)
        return {"response": processed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))