from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.responses import HTMLResponse

app = FastAPI()

# Estructura del mensaje del usuario
class Message(BaseModel):
    content: str  # Solo el contenido del mensaje del usuario

# Ruta POST para enviar un mensaje al chatbot
@app.post("/chatbot")
async def chatbot_post(chat_request: Message):
    # URL de la API de Ollama
    url = "http://127.0.0.1:11435/api/chat"  # Asegúrate de que este es el puerto correcto

    # Parámetros que se enviarán a la API de Ollama
    payload = {
        "model": "llama3.2:latest",
        "messages": [
            {"role": "system", "content": "Eres un chatbot útil y amigable."},  # Mensaje de sistema
            {"role": "user", "content": chat_request.content}  # Mensaje del usuario
        ]
    }

    # Envío de la solicitud POST a la API de Ollama
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Maneja errores HTTP
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error en la solicitud a la API de Ollama: {str(e)}")

    # Obtener la respuesta del modelo
    try:
        response_data = response.json()  # Parsear la respuesta JSON
        assistant_message = response_data.get("message", {}).get("content", "No response from model.")
    except ValueError:
        raise HTTPException(status_code=500, detail="Error al procesar la respuesta JSON de la API de Ollama")

    # Retornar la respuesta del asistente al cliente
    return {"response": assistant_message}

# Ruta GET para mostrar el formulario HTML (solo si deseas una interfaz sencilla)
@app.get("/chatbot", response_class=HTMLResponse)
async def get_chatbot_page():
    return """
<html>
    <body>
        <h2>Chatbot con Ollama</h2>
        <form id="chat-form">
            <label for="message">Tu mensaje:</label>
            <input type="text" id="message" name="content" required>
            <button type="submit">Enviar</button>
        </form>

        <h3>Respuesta:</h3>
        <pre id="response"></pre>

        <script>
            const form = document.getElementById("chat-form");
            const responseElement = document.getElementById("response");

            form.addEventListener("submit", async function(event) {
                event.preventDefault();  // Prevenir el comportamiento por defecto del formulario

                const message = document.getElementById("message").value;
                const response = await fetch("/chatbot", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        content: message
                    })
                });

                const data = await response.json();
                responseElement.textContent = JSON.stringify(data, null, 2);
            });
        </script>
    </body>
</html>
"""

