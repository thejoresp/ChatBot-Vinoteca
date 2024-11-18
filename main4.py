from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch, ElasticsearchWarning
import warnings
warnings.filterwarnings('ignore', category=ElasticsearchWarning)
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

# Modelo de datos para las consultas
class Query(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    content: str

@app.post("/index_csv_fixed")
def index_csv_fixed():
    """
    Endpoint para indexar automáticamente el contenido de un archivo CSV desde una ruta fija.
    """
    file_path = '/home/jl/Descargas/tipos_vinos.csv'
    try:
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            embedding = model.encode([row['content']])[0]
            doc_body = {
                'id': idx,
                'content': row['content'],
                'embedding': embedding.tolist()
            }
            es.index(index=index_name, id=idx, body=doc_body)
        return {"message": "Archivo CSV indexado correctamente desde la ruta fija."}
    except Exception as e:
        return {"error": f"Error al indexar el archivo: {str(e)}"}

@app.post("/retrieve")
def retrieve_documents(query: Query):
    """
    Endpoint para buscar documentos relevantes en Elasticsearch usando embeddings.
    """
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
    """
    Endpoint de chat que encuentra el documento más relevante para una consulta dada.
    """
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
    """
    Endpoint raíz para probar la conectividad con la API.
    """
    return {"message": "Bienvenido a la API de Enotek Vinos"}
