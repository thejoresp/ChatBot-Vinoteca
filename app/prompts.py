"""Construcción del prompt que recibe el modelo."""

from app.rag.retriever import Resultado

SYSTEM_PROMPT = """Sos el asistente virtual de Enotek Vinos, una vinoteca argentina.

Reglas que tenés que seguir siempre:
1. Respondé únicamente con la información del listado de datos que acompaña a cada \
consulta, reproduciendo los valores exactos: precios, direcciones con su altura y \
horarios completos. Lo que no tenés que copiar es la forma del listado —nada de \
encabezados ni rótulos—: contestá con frases naturales, como le hablarías a un cliente \
en el mostrador.
2. El listado se arma por búsqueda automática y corresponde sólo a la última \
consulta, no a las anteriores. Puede traer datos que no tengan nada que ver con lo \
que se preguntó: antes de responder fijate si realmente la contestan, y si no, tratá \
el caso como si no hubiera datos. Nunca fuerces una respuesta con lo que haya a mano \
ni comentes qué información te falta sobre preguntas anteriores.
3. Cuando no tengas el dato, decilo con naturalidad y ofrecé ayudar con precios de \
productos, horarios o direcciones de las sucursales. Nunca inventes precios, marcas, \
direcciones ni horarios.
4. Si la pregunta no tiene nada que ver con la vinoteca (política, deportes, \
programación, cultura general), aclará amablemente que sólo podés ayudar con \
consultas sobre Enotek Vinos, y no la respondas aunque sepas la respuesta.
5. Contestá en español rioplatense, en tono cordial y breve.
6. Cuando enumeres varios productos o sucursales, usá una lista con guiones. \
Mostrá los precios en pesos con el símbolo $.
"""

SIN_RESULTADOS = (
    "No se encontró información relacionada con esta consulta en los datos de la vinoteca."
)


def construir_contexto(resultados: list[Resultado]) -> str:
    """Formatea los documentos recuperados como bloque de contexto para el modelo."""
    if not resultados:
        return SIN_RESULTADOS
    return "\n".join(f"- {r.document.text}" for r in resultados)


def construir_mensajes(
    pregunta: str, historial: list[dict[str, str]], resultados: list[Resultado]
) -> list[dict[str, str]]:
    """Arma la lista de mensajes para el modelo.

    Disposición, y por qué:

    1. El prompt de sistema se antepone en cada llamada en vez de vivir dentro
       del historial, así el recorte por antigüedad nunca puede eliminarlo.
    2. Los documentos recuperados van en un mensaje `system` propio, no dentro
       del mensaje del usuario. Mezclarlos ahí hacía que el historial quedara
       con mensajes de usuario de dos formas distintas (los viejos, crudos, y el
       actual, con encabezados), y llama3.2:3b reaccionaba inventando precios.
    3. El bloque de contexto no lleva ningún rótulo en mayúsculas. Con uno
       ("DATOS DE LA VINOTECA:"), el modelo lo copiaba tal cual al empezar
       algunas respuestas: un texto llamativo y separado del resto es
       justamente lo que un modelo chico tiende a imitar. Redactado como una
       instrucción corrida, deja de tener algo que copiar.
    """
    contexto = construir_contexto(resultados)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *historial,
        {
            "role": "system",
            "content": (
                "Para responder la próxima consulta sólo podés usar estos datos de "
                f"la vinoteca:\n{contexto}"
            ),
        },
        {"role": "user", "content": pregunta},
    ]
