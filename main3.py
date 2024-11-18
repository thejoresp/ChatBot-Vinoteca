import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

# Inicializar la aplicación de FastAPI
app = FastAPI()

# Historial de mensajes para la conversación
messages = [
    {'role': 'system', 'content': 'Eres un asistente útil.'},  # Configuración inicial
]

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Ollama"}

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Espera que el cliente envíe un mensaje
            data = await websocket.receive_text()

            # Si el mensaje es "salir", terminamos la conversación
            if data.lower() == "salir":
                await websocket.send_text("¡Adiós!")
                break

            # Agregar la pregunta al historial de mensajes
            messages.append({'role': 'user', 'content': data})

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

            # Enviar la respuesta al cliente
            await websocket.send_text(respuesta)
    except WebSocketDisconnect:
        print("Cliente desconectado")

