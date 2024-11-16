# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import ollama
# import requests  # Usar requests en lugar de httpx
#
# # Configuración de FastAPI
# app = FastAPI()
#
# # Clase para la estructura de mensaje
# class Message(BaseModel):
#     role: str  # Puede ser "user", "assistant", "system", etc.
#     content: str  # El contenido del mensaje
#
# # Variable global para almacenar el historial de conversación
# chat_history = []
#
# # Ruta POST para enviar un mensaje al chatbot
# @app.post("/chatbot")
# def chatbot_post(chat_request: List[Message]):
#     global chat_history
#
#     # Agregar los mensajes recibidos al historial
#     chat_history.extend([msg.dict() for msg in chat_request])
#
#     # URL de la API de Ollama
#     url = "http://127.0.0.1:11434/api/chat"
#
#     # Parámetros a enviar a Ollama
#     payload = {
#         "model": "llama3.2:latest",  # Modelo de la API de Ollama
#         "messages": chat_history,  # Usar el historial de la conversación
#         "stream": True  # No usar streaming
#     }
#
#     # Realizar la solicitud a Ollama con requests
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status()  # Levanta una excepción si el código de estado es 4xx o 5xx
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Error en la solicitud a la API de Ollama: {str(e)}")
#
#     # Obtener la respuesta y agregarla al historial
#     response_data = response.json()
#     assistant_message = {
#         "role": "assistant",
#         "content": response_data.get("text", "No response from model.")
#     }
#
#     # Agregar la respuesta del asistente al historial
#     chat_history.append(assistant_message)
#
#     # Devolver la respuesta al cliente
#     return response_data
#
# # Ruta GET para obtener el historial de mensajes
# @app.get("/chatbot")
# def get_chat_history():
#     return {"chat_history": chat_history}



# import ollama
#
# stream = ollama.chat(
#     model='llama3.2:latest',
#     messages=[{'role': 'user', 'content': 'que contiene el agua'}],
#     stream=True,
# )
#
# for chunk in stream:
#     print(chunk['message']['content'], end='', flush=True)


import ollama

# Iniciar la conversación con un modelo específico
stream = ollama.chat(
    model='llama3.2:latest',  # El modelo que deseas usar
    messages=[{'role': 'user', 'content': '¿Qué contiene el agua?'}],  # Mensajes enviados
    stream=True,  # Activar el streaming de la respuesta
)

# Recibir y mostrar la respuesta
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)