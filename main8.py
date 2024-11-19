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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Configuración del modelo de Ollama
llm = OllamaLLM(model="llama3.2:latest")

# Variable global para mensajes
messages = []

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Ollama para Vinotecas"}

class Message(BaseModel):
    content: str

def buscar_informacion_con_langchain(query, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    respuesta = qa_chain.invoke(query)
    return respuesta

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
        response = ollama.chat(
            model='llama3.2:latest',
            messages=messages
        )

        respuesta = response['message']['content']

        # Agregar la respuesta del modelo al historial
        messages.append({'role': 'assistant', 'content': respuesta})

        # Limitar el tamaño del historial de mensajes
        if len(messages) > 100:
            messages = messages[-100:]

        return {"response": respuesta}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)