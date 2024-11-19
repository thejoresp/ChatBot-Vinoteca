from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ollama
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

app = FastAPI()

# Definir la variable global messages
messages = []

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Cargar datos desde los archivos CSV
try:
    df_ubicaciones = pd.read_csv('/home/jl/Development/IFTS11/Procesamiento de habla/ChatBot-Vinoteca/date/Ubicaciones.csv')
    df_precios = pd.read_csv('/home/jl/Development/IFTS11/Procesamiento de habla/ChatBot-Vinoteca/date/Lista de precios.csv')
    print("Datos de ubicaciones cargados correctamente")
    print("Datos de precios cargados correctamente")
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Archivo CSV no encontrado: {e.filename}")
except pd.errors.EmptyDataError:
    raise HTTPException(status_code=500, detail="Archivo CSV vacío")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {e}")

# Cargar el modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Convertir DataFrame a lista de documentos
ubicaciones_docs = [Document(page_content=str(row)) for row in df_ubicaciones.to_dict(orient="records")]
precios_docs = [Document(page_content=str(row)) for row in df_precios.to_dict(orient="records")]

# Crear Vector Store para Ubicaciones y Precios usando Chroma
vectorstore_ubicaciones = Chroma.from_documents(ubicaciones_docs, embeddings)
vectorstore_precios = Chroma.from_documents(precios_docs, embeddings)

# Paso 3: Configuración del modelo de Ollama
llm = OllamaLLM(model="llama3.2:latest")

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Ollama para Vinotecas"}

class Message(BaseModel):
    content: str

@app.post("/chat")
async def chat(message: Message):
    global messages

    try:
        # Agregar la pregunta al historial de mensajes
        messages.append({'role': 'user', 'content': message.content})

        # Realizar la consulta en los vectorstores usando LangChain
        query = message.content
        respuesta_ubicaciones = buscar_informacion_con_langchain(query, vectorstore_ubicaciones)
        respuesta_precios = buscar_informacion_con_langchain(query, vectorstore_precios)

        # Combinar las respuestas
        informacion_adicional = ''
        if respuesta_ubicaciones and 'result' in respuesta_ubicaciones:
            informacion_adicional += f"Información sobre ubicaciones:\n{respuesta_ubicaciones['result']}\n\n"
        if respuesta_precios and 'result' in respuesta_precios:
            informacion_adicional += f"Información sobre precios:\n{respuesta_precios['result']}\n\n"

        if informacion_adicional:
            messages.append({'role': 'system', 'content': f'Aquí está la información relevante de nuestros datos:\n{informacion_adicional}'})

        # Realizar la consulta a Ollama
        stream = ollama.chat(
            model='llama3.2:latest',
            messages=messages,
            stream=True,
        )

        respuesta = ''
        for chunk in stream:
            print(f"Chunk recibido: {chunk}")  # Depuración para verificar cada chunk recibido
            if 'message' in chunk and 'content' in chunk['message']:
                respuesta += chunk['message']['content']

        print(f"Respuesta generada: {respuesta}")  # Depuración para verificar la respuesta generada

        # Agregar la respuesta del modelo al historial
        messages.append({'role': 'assistant', 'content': respuesta})

        # Limitar el tamaño del historial de mensajes
        if len(messages) > 100:
            messages = messages[-100:]

        return {"response": respuesta.strip()}  # Devuelve la respuesta completa

    except Exception as e:
        print(f"Error: {e}")  # Mensaje de depuración para el error
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {e}")

def buscar_informacion_con_langchain(query, vectorstore):
    try:
        # Realizar la búsqueda en el vectorstore usando LangChain
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        respuesta = qa_chain(query)  # Usar __call__ en lugar de invoke o run
        print(f"Consulta: {query}, Respuesta: {respuesta}")  # Mensaje de depuración
        return {"result": respuesta} if isinstance(respuesta, str) else respuesta
    except Exception as e:
        print(f"Error al buscar información con LangChain: {e}")
        return {"result": "Error al buscar información"}