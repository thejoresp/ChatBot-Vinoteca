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

# Configuración del modelo de Ollama
llm = OllamaLLM(model="llama3.2:latest")

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
    query = message.content
    respuesta_ubicaciones = buscar_informacion_con_langchain(query, vectorstore_ubicaciones)
    respuesta_precios = buscar_informacion_con_langchain(query, vectorstore_precios)

    # Combinar las respuestas
    informacion_adicional = ''

    if respuesta_ubicaciones and 'result' in respuesta_ubicaciones:
        informacion_adicional += f"Información sobre ubicaciones:\n{respuesta_ubicaciones['result']}\n\n"

    if respuesta_precios and 'result' in respuesta_precios:
        informacion_adicional += f"Información sobre precios:\n{respuesta_precios['result']}\n\n"

    print(f'Más información adicional: {informacion_adicional}')
    print(f'Aquí están los mensajes: {messages}\n')

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

    respuesta = ''
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            content = chunk['message']['content']
            respuesta += content  # Acumula el contenido del chunk

    # Verificar si se recibió una respuesta
    if not respuesta:
        respuesta = "No se recibió una respuesta válida."

    # Agregar la respuesta del modelo al historial
    messages.append({'role': 'assistant', 'content': respuesta})
    print(f'Aquí está el mensaje: {respuesta}')

    # Limitar el tamaño del historial de mensajes
    if len(messages) > 100:
        messages = messages[-100:]

    # Devolver la respuesta al cliente
    return {"response": respuesta.strip()}

def buscar_informacion_con_langchain(query, vectorstore):
    try:
        # Realizar la búsqueda en el vectorstore usando LangChain
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        respuesta = qa_chain.invoke(query)  # Usar invoke en lugar de __call__ o run
        return {"result": respuesta} if isinstance(respuesta, str) else respuesta
    except Exception as e:
        return {"result": "Error al buscar información"}