import ollama
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

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
    {'role': 'system', 'content': 'Eres un asistente útil, resolviendo dudas.'},  # Configuración inicial
]

class Message(BaseModel):
    content: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Ollama"}

@app.post("/chat")
async def chat(message: Message):
    # Agregar la pregunta al historial de mensajes
    messages.append({'role': 'user', 'content': message.content})

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

    # Agregar la respuesta del modelo al historial
    messages.append({'role': 'assistant', 'content': respuesta})

    # Devolver la respuesta al cliente
    return {"response": respuesta}