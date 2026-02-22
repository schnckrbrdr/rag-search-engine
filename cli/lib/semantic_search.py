from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from constants import CACHE_DIR, DATA_PATH

def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_data()
    semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")

def load_data():
    movies_file = open(DATA_PATH)
    movies_json = json.load(movies_file)        
    return movies_json['movies']

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
        
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

class SemanticSearch:

    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2', token=True)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError(f"Error: generate_embedding(): text mus not by empty")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        document_list = []
        for document in documents:
            self.document_map[document['id']] = document
            document_list.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(document_list, show_progress_bar=True)
        self.save_embeddings()

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        
        self.documents = documents
        
        for document in documents:
            self.document_map[document['id']] = document

        if os.path.exists(self.embeddings_path):
            self.load_embeddings()
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def save_embeddings(self) -> None:
        if not os.path.isdir(CACHE_DIR):
            os.mkdir(CACHE_DIR)

        np.save(open(self.embeddings_path, "wb"), self.embeddings)

    def load_embeddings(self) -> None:
        self.embeddings = np.load(self.embeddings_path)