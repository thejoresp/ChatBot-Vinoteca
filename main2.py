from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd

app = FastAPI()

# Configurar CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    # Agrega aquí otros orígenes permitidos
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Conectar a Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Crear el índice en Elasticsearch
index_name = 'documents'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

class Document(BaseModel):
    id: int
    content: str

class Query(BaseModel):
    query: str
    top_k: int = 5

class Message(BaseModel):
    content: str

@app.post("/index")
def index_document(doc: Document):
    embedding = model.encode([doc.content])[0]
    doc_body = {
        'id': doc.id,
        'content': doc.content,
        'embedding': embedding.tolist()
    }
    es.index(index=index_name, id=doc.id, body=doc_body)
    return {"message": "Documento indexado correctamente"}

@app.post("/index_csv")
def index_csv(file_path: str):
    # Leer el archivo CSV desde la ruta especificada
    df = pd.read_csv(file_path)
    for idx, row in df.iterrows():
        doc = Document(id=idx, content=row['content'])
        index_document(doc)
    return {"message": "Archivo CSV indexado correctamente"}

@app.post("/retrieve")
def retrieve_documents(query: Query):
    query_embedding = model.encode([query.query])[0]
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding.tolist()}
            }
        }
    }
    response = es.search(index=index_name, body={"query": script_query, "size": query.top_k})
    results = [hit['_source'] for hit in response['hits']['hits']]
    return {"results": results}

@app.post("/chat")
async def chat(message: Message):
    # Aquí iría la lógica para interactuar con el modelo de lenguaje
    respuesta = 'Esta es una respuesta simulada.'  # Simulación de respuesta
    return {"response": respuesta}

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API"}