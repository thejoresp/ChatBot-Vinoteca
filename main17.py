from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.base import StuffDocumentsChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma  # O FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings  # O Hugging Face
from typing import List

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Historial de mensajes para la conversación
messages = [
    {
        'role': 'system',
        'content': 'Eres un asistente especializado en vinos y vinotecas. Debes responder únicamente basándote en la información que te proporcionamos. No agregues información adicional que no esté en nuestros datos.'
    },
]

# Cargar datos desde los archivos CSV
try:
    df_ubicaciones = pd.read_csv('/home/jl/Development/IFTS11/Procesamiento de habla/ChatBot-Vinoteca/date/Ubicaciones.csv')
    df_precios = pd.read_csv('/home/jl/Development/IFTS11/Procesamiento de habla/ChatBot-Vinoteca/date/Lista de precios.csv')
    print("Datos de ubicaciones cargados correctamente:")
    print(df_ubicaciones.head())
    print("Datos de precios cargados correctamente:")
    print(df_precios.head())
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Archivo CSV no encontrado: {e.filename}")
except pd.errors.EmptyDataError:
    raise HTTPException(status_code=500, detail="Archivo CSV vacío")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {e}")

class Message(BaseModel):
    content: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Ollama para Vinotecas"}

@app.post("/chat")
async def chat(message: Message):
    global messages  # Declarar 'messages' como global

    # Agregar la pregunta al historial de mensajes
    messages.append({'role': 'user', 'content': message.content})

    # Buscar información relevante en los CSV usando LangChain
    respuesta_ubicaciones = buscar_informacion_langchain(df_ubicaciones, message.content)
    respuesta_precios = buscar_informacion_langchain(df_precios, message.content)

    # Combinar las respuestas
    informacion_adicional = ''

    if respuesta_ubicaciones:
        informacion_adicional += f"Información sobre ubicaciones:\n{respuesta_ubicaciones}\n\n"

    if respuesta_precios:
        informacion_adicional += f"Información sobre precios:\n{respuesta_precios}\n\n"

    # Añadir la información adicional al contexto del asistente
    if informacion_adicional:
        messages.append({
            'role': 'system',
            'content': f'Aquí está la información relevante de nuestros datos:\n{informacion_adicional}'
        })

    # Realizar la consulta a Ollama
    try:
        stream = ollama.chat(
            model='llama3.2:latest',
            messages=messages,
            stream=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al consultar el modelo generativo: {e}")

    # Recoger la respuesta del modelo
    respuesta = ''
    for chunk in stream:
        respuesta += chunk['message']['content']

    # Agregar la respuesta del modelo al historial
    messages.append({'role': 'assistant', 'content': respuesta})

    # Limitar el tamaño del historial de mensajes
    if len(messages) > 100:
        messages = messages[-100:]

    # Devolver la respuesta al cliente
    return {"response": respuesta}

def buscar_informacion_langchain(df, query):
    # Crear un vectorstore con los datos del DataFrame
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents([Document(page_content=row.to_json()) for _, row in df.iterrows()], embeddings)

    # Crear una cadena de recuperación de QA
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA(retriever=retriever)

    # Realizar la consulta
    result = qa_chain.run(query)
    return result