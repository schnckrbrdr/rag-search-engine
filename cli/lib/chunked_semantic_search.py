from lib.semantic_search import SemanticSearch, semantic_chunking, cosine_similarity
from utility import load_movies
from constants import CACHE_DIR
import os
import numpy as np
import json

CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def embed_chunks_command():
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents['movies'])
    print(f"Generated {len(chunked_semantic_search.chunk_embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int):
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents['movies'])
    search_results = chunked_semantic_search.search_chunks(query, limit)

    for idx, result in enumerate(search_results):
        print(f"\n{idx + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description']}...")

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None        

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}

        chunk_list = []
        chunk_dict = []

        # Iterate over all documents
        for i in range(0, len(self.documents)):
            doc = self.documents[i]

            # Skip document if there is no description
            if not doc['description'] or not doc['description'].strip():
                continue

            # Generate chunks fpr current document description and add all chunks to the overall chunks-list
            doc_chunks = semantic_chunking(doc['description'], 4, 1)
            chunk_list.extend(doc_chunks)

            if i == 0:
                print(f"Description: {doc['description']}")
                print(f"Chunks: {doc_chunks}")

            # Iterate over chunks of current document description and add a dictionary for each chunk to store movie-index, relative chunk-index and number of chunks of current document-description
            for j in range(0, len(doc_chunks)):
                chunk_dict.append({'movie_idx': i, 'chunk_idx': j, 'total_chunks': len(doc_chunks)})
        
        print(f"Length chunk list: {len(chunk_list)}")
        print(f"1: {chunk_list[0]}")

        # Set object-attributes 
        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_metadata = chunk_dict
        
        # Save Metadata and Embeddings to file
        if not os.path.isdir(CACHE_DIR):
            os.mkdir(CACHE_DIR)

        np.save(open(CHUNK_EMBEDDINGS_PATH, "wb"), self.chunk_embeddings)
        json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)}, open(CHUNK_METADATA_PATH, "w"), indent=2)

        # Return Embeddings
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        
        self.documents = documents
        
        for doc in documents:
            self.document_map[doc['id']] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)            
            data = json.load(open(CHUNK_METADATA_PATH, "r"))
            self.chunk_metadata = data["chunks"]
            return self.embeddings

        return self.build_chunk_embeddings(documents)
        
    def search_chunks(self, query: str, limit: int = 10):
        
        # Generate embedding of query
        query_embedding = super().generate_embedding(query)
        chunk_scores = []

        # Iterate over all chunks and calculate the cosine similarity with the embeddings of the query, store result in list (idx of movie in documents etc.)
        for i in range(0, len(self.chunk_embeddings)):            
            cosine_similarity_score = cosine_similarity(self.chunk_embeddings[i], query_embedding)
            chunk_scores.append({'chunk_idx': self.chunk_metadata[i]['chunk_idx'], 'movie_idx': self.chunk_metadata[i]['movie_idx'], 'score': cosine_similarity_score})            

        movie_index_score_dict = {}

        # Store highest chunk-score for any movie in a dictionary mapping the movie-idx to the chunk_score-dictionary
        for chunk_score in chunk_scores:
            if chunk_score['movie_idx'] not in movie_index_score_dict or chunk_score['score'] > movie_index_score_dict[chunk_score['movie_idx']]:
                movie_index_score_dict[chunk_score['movie_idx']] = chunk_score['score']

        # Sort mapped movie-idx-scores by score, limit results and convert to a list
        sorted_movie_chunk_scores = list(sorted(movie_index_score_dict.items(), key=lambda d: d[1], reverse=True))[0:limit]

        formatted_sorted_movies = []

        # Format results and retrieve data like movie-title etc. for search results
        for movie_idx, score in sorted_movie_chunk_scores:
            if movie_idx is None:
                continue
            movie = self.documents[movie_idx]
            formatted_sorted_movies.append({'id': movie['id'],
                                            'title': movie['title'],
                                            'description': movie['description'][:100],
                                            'score': round(score, 4)})
        
        return formatted_sorted_movies



