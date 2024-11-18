import ollama
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las orígenes, puedes restringir esto según sea necesario
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Historial de mensajes para la conversación
messages = [
    {'role': 'system', 'content': 'Eres un asistente útil especializado en vinos y vinotecas, siempre cuando empieza un chat saludo y representa a la "Vinoteca Enotek", ofreciendo tu servicio.'},  # Configuración inicial
]

# Cargar datos desde el archivo CSV
df = pd.read_csv('/home/jl/Development/Ciencia Datos/Procesamiento de habla/ChatBot-Vinoteca/tipos_vinos.csv')

class Message(BaseModel):
    content: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Ollama para Vinotecas"}

@app.post("/chat")
async def chat(message: Message):
    # Agregar la pregunta al historial de mensajes
    messages.append({'role': 'user', 'content': message.content})

    # Buscar información relevante en el CSV
    respuesta_csv = buscar_informacion(message.content)

    # Realizar la consulta a Ollama
    stream = ollama.chat(
        model='llama3.2:3b',  # Modelo a usar
        messages=messages,  # Pasar todo el historial de mensajes
        stream=True,
    )

    # Recoger la respuesta del modelo
    respuesta = ''
    for chunk in stream:
        respuesta += chunk['message']['content']

    # Combinar la respuesta del modelo con la información del CSV
    respuesta_final = f"{respuesta}\n\nInformación adicional:\n{respuesta_csv}"

    # Agregar la respuesta del modelo al historial
    messages.append({'role': 'assistant', 'content': respuesta_final})

    # Devolver la respuesta al cliente
    return {"response": respuesta_final}
def buscar_informacion(query):
    # Implementar lógica para buscar información relevante en el DataFrame
    # Por ejemplo, buscar por tipo de vino, ubicación, etc.
    resultados = df[df.apply(lambda row: query.lower() in row.astype(str).str.lower().values, axis=1)]
    if not resultados.empty:
        return resultados.to_dict(orient='records')
    else:
        return "No se encontró información relevante en la base de datos."