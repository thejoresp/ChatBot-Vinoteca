from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por dominios específicos si es necesario.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Conectar a Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Crear el índice en Elasticsearch si no existe
index_name = 'documents'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Modelos de datos
class Document(BaseModel):
    id: int
    content: str

class Query(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
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

@app.post("/index_csv_fixed")
def index_csv_fixed():
    # Leer el archivo CSV desde una ruta fija
    file_path = '/home/jl/Descargas/tipos_vinos.csv'
    df = pd.read_csv(file_path)
    for idx, row in df.iterrows():
        doc = Document(id=idx, content=row['content'])
        index_document(doc)
    return {"message": "Archivo CSV indexado correctamente desde ruta fija"}

@app.post("/index_csv")
async def index_csv(file: UploadFile = File(...)):
    # Leer el archivo CSV subido
    df = pd.read_csv(file.file)
    for idx, row in df.iterrows():
        doc = Document(id=idx, content=row['content'])
        index_document(doc)
    return {"message": "Archivo CSV subido e indexado correctamente"}

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
def chat_endpoint(request: ChatRequest):
    query_embedding = model.encode([request.content])[0]
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding.tolist()}
            }
        }
    }
    response = es.search(index=index_name, body={"query": script_query, "size": 1})
    if response['hits']['hits']:
        best_match = response['hits']['hits'][0]['_source']['content']
        return {"response": f"El documento más relevante es: {best_match}"}
    else:
        return {"response": "No se encontraron documentos relevantes."}

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API"}
