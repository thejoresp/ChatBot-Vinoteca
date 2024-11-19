from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ollama

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
        'content': 'Eres un asistente virtual de "Vinoteca Enotek". Proporciona información sobre nuestros productos y ubicaciones basándote en los datos suministrados. Siempre saluda al usuario al iniciar una conversación y ofrece tus servicios.'
    },
]

# Cargar datos desde los archivos CSV
try:
    df_ubicaciones = pd.read_csv('/home/jl/Development/Ciencia Datos/Procesamiento de habla/ChatBot-Vinoteca/date/Ubicaciones.csv')
    df_precios = pd.read_csv('/home/jl/Development/Ciencia Datos/Procesamiento de habla/ChatBot-Vinoteca/date/Lista de precios.csv')
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
    return {"message": "Bienvenido a la API de Vinoteca Enotek"}

@app.post("/chat")
async def chat(message: Message):
    global messages  # Declarar 'messages' como global

    # Agregar la pregunta al historial de mensajes
    messages.append({'role': 'user', 'content': message.content})

    # Buscar información relevante en los CSV
    respuesta_ubicaciones = buscar_informacion_ubicaciones(message.content)
    respuesta_precios = buscar_informacion_precios(message.content)

    # Combinar las respuestas
    informacion_adicional = ''

    if respuesta_ubicaciones:
        informacion_adicional += f"Nuestras ubicaciones relevantes:\n{respuesta_ubicaciones}\n\n"

    if respuesta_precios:
        informacion_adicional += f"Información sobre precios y productos:\n{respuesta_precios}\n\n"

    # Añadir la información adicional al contexto del asistente
    if informacion_adicional:
        messages.append({
            'role': 'system',
            'content': f'Usa la siguiente información para ayudar al usuario:\n{informacion_adicional}'
        })

    # Limitar el tamaño del historial de mensajes
    if len(messages) > 100:
        messages = messages[-100:]

    # Realizar la consulta a Ollama
    try:
        stream = ollama.chat(
            model='llama3.2:3b',
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

    # Devolver la respuesta al cliente
    return {"response": respuesta}

def buscar_informacion_ubicaciones(query):
    resultados = df_ubicaciones[df_ubicaciones.apply(
        lambda row: query.lower() in ' '.join(row.astype(str).str.lower().values), axis=1)]
    if not resultados.empty:
        respuesta = ""
        for index, row in resultados.iterrows():
            respuesta += f"- **Sucursal:** {row['Sucursal']} en {row['Ciudad']}, ubicada en {row['Dirección']}. Horario: {row['Horarios']}\n"
        return respuesta
    else:
        return None

def buscar_informacion_precios(query):
    resultados = df_precios[df_precios.apply(
        lambda row: query.lower() in ' '.join(row.astype(str).str.lower().values), axis=1)]
    if not resultados.empty:
        respuesta = ""
        for index, row in resultados.iterrows():
            respuesta += f"- **{row['Producto']}** ({row['Categoría']}): ${row['Precio']}\n"
        return respuesta
    else:
        return None