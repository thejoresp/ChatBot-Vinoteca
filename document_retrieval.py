
from some_embedding_library import generate_embeddings
from some_database_library import DatabaseClient

db_client = DatabaseClient()

def index_documents(documents):
    for doc in documents:
        embedding = generate_embeddings(doc)
        db_client.index_document(doc, embedding)

def retrieve_documents(query):
    query_embedding = generate_embeddings(query)
    results = db_client.search(query_embedding)
    return [result['content'] for result in results]