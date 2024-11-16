import ollama

# Inicializar el historial de mensajes
messages = [
    {'role': 'system', 'content': 'Eres un asistente útil.'},  # Configuración inicial
]

def interactuar():
    global messages

    print("Escribe 'salir' para terminar la conversación.")

    while True:
        # Leer entrada del usuario
        pregunta = input("Tú: ")

        # Salir del bucle si el usuario escribe 'salir'
        if pregunta.lower() == "salir":
            print("¡Adiós!")
            break

        # Agregar el nuevo mensaje del usuario al historial
        messages.append({'role': 'system', 'content': pregunta})

        # Realizar la petición al modelo
        stream = ollama.chat(
            model='llama3.2:latest',  # Modelo a usar
            messages=messages,  # Pasar todo el historial de mensajes
            stream=True,
        )

        # Obtener la respuesta
        respuesta = ''
        for chunk in stream:
            respuesta += chunk['message']['content']

        # Agregar la respuesta del modelo al historial
        messages.append({'role': 'assistant', 'content': respuesta})

        # Mostrar la respuesta del modelo
        print("Asistente:", respuesta)

# Iniciar la interacción
interactuar()

