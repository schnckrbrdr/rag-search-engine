import os
import time

from lib.chunked_semantic_search import ChunkedSemanticSearch
from inverted_index import InvertedIndex
from utility import load_movies
from lib.llm import LLM

def normalize_scores(scores: list[float]) -> list[float]:
    
    if len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = []

    for score in scores:
        normalized_scores.append((score - min_score) / (max_score - min_score))

    return normalized_scores

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def normalize_command(scores: list[float]):
    normalized_scores = normalize_scores(scores)

    for score in normalized_scores:
        print(f"* {score:.4f}")

def weighted_search_command(query: str, alpha: float, limit: int):
    documents = load_movies()
    hybrid_search = HybridSearch(documents['movies'])
    results = hybrid_search.weighted_search(query, alpha, limit)

    for idx, result in enumerate(results):
        print(f"{idx + 1}. {result[1]['document']['title']}")
        print(f"Hybrid Score: {result[1]['hybrid_score']:.4f}")
        print(f"BM25: {result[1]['bm25score']:.4f}, Semantic: {result[1]['semantic_score']:.4f}")
        print(f"{result[1]['document']['description'][:100]}...\n")

def rrf_search_command(query: str, k: int, limit: int, enhance_choice: str, rerank_method: str):
    documents = load_movies()

    llm = LLM("gemma-3-27b-it")
    rerank_limit = limit

    if enhance_choice:
        
        new_query = llm.enhance_query(query, enhance_choice)
        if new_query != query:
           print(f"Enhanced query ({enhance_choice}): '{query}' -> '{new_query}'\n")
           query = new_query

    hybrid_search = HybridSearch(documents['movies'])

    if rerank_method == "individual":
        
        results = hybrid_search.rrf_search(query, k, limit * 5)

        for result in results:

            score = llm.rerank_request(query, result[1])

            result[1]['rerank_score'] = score
            time.sleep(3)

        sorted_scores = sorted(results, key=lambda d: d[1]['rerank_score'], reverse=True)[0:limit]

        for idx, result in enumerate(sorted_scores):
            print(f"{idx + 1}. {result[1]['document']['title']}")
            print(f"Re-rank Score: {result[1]['rerank_score']} / 10")
            print(f"RFF Score: {result[1]['rrf_score']:.4f}")
            print(f"BM25 Rank: {result[1]['bm25_rank']}, Semantic Rank: {result[1]['semantic_rank']}")
            print(f"{result[1]['document']['description'][:100]}...\n")
    
    elif rerank_method == "batch":
        
        results = hybrid_search.rrf_search(query, k, limit * 5)

        ranked_movie_ids = llm.rerank_batch(query, results)[:limit]

        results_dict = dict(results)

        counter = 0

        for id in ranked_movie_ids:
            if int(id) in results_dict:
                result = results_dict[int(id)]
                counter += 1
                print(f"{counter}. {result['document']['title']}")
                print(f"Re-rank Rank: {counter}")
                print(f"RFF Score: {result['rrf_score']:.4f}")
                print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                print(f"{result['document']['description'][:100]}...\n")
            else:
                print(f"Invalid Movie-ID returnes: {int(id)}")

    else:

        results = hybrid_search.rrf_search(query, k, rerank_limit)

        for idx, result in enumerate(results):
            print(f"{idx + 1}. {result[1]['document']['title']}")
            print(f"RFF Score: {result[1]['rrf_score']:.4f}")
            print(f"BM25 Rank: {result[1]['bm25_rank']}, Semantic Rank: {result[1]['semantic_rank']}")
            print(f"{result[1]['document']['description'][:100]}...\n") 


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit):
        bm25_results = self._bm25_search(query, 500 * limit)
        chunked_search_results = self.semantic_search.search_chunks(query, 500 * limit)

        weighted_scores = {}
        bm25_scores = []
        semantic_scores = []

        for result in bm25_results:
            bm25_scores.append(result['score'])

        for result in chunked_search_results:
            semantic_scores.append(result['score'])

        normalized_bm25_scores = normalize_scores(bm25_scores)
        normalized_semantic_scores = normalize_scores(semantic_scores)

        for idx, result in enumerate(bm25_results):
            normalized_score = normalized_bm25_scores[idx]
            weighted_scores[result['doc_id']] = {'bm25score': normalized_score, 'semantic_score': 0, 'hybrid_score': 0, 'document': result['movie']}

        for idx, result in enumerate(chunked_search_results):
            normalized_score = normalized_semantic_scores[idx]
            
            if result['id'] in weighted_scores:
                weighted_scores[result['id']]['semantic_score'] = normalized_score
            else:
                weighted_scores[result['id']] = {'bm25score': 0, 'semantic_score': normalized_score, 'hybrid_score':0, 'document': self.semantic_search.document_map[result['id']]}

        for entry in weighted_scores.items():
            entry[1]['hybrid_score'] = hybrid_score(entry[1]['bm25score'], entry[1]['semantic_score'], alpha)

        sorted_scores = list(sorted(weighted_scores.items(), key=lambda d: d[1]['hybrid_score'], reverse=True))[0:limit]

        return sorted_scores

    def rrf_search(self, query, k, limit):
        
        bm25_results = self._bm25_search(query, 500 * limit)
        chunked_search_results = self.semantic_search.search_chunks(query, 500 * limit)

        rrf_score_dict = {}

        for idx, result in enumerate(bm25_results):
            rrf_score_dict[result['doc_id']] = {'document': result['movie'], 'rrf_score': rrf_score(idx + 1, k), 'bm25_rank': idx + 1, 'semantic_rank': None}

        for idx, result in enumerate(chunked_search_results):
            if result['id'] in rrf_score_dict:
                rrf_score_dict[result['id']]['rrf_score'] += rrf_score(idx + 1, k)
                rrf_score_dict[result['id']]['semantic_rank'] = idx + 1
            else:
                rrf_score_dict[result['id']] = {'document': self.semantic_search.document_map[result['id']], 'rrf_score': rrf_score(idx + 1, k), 'bm25_rank': None, 'semantic_rank': idx + 1}
        
        sorted_scores = list(sorted(rrf_score_dict.items(), key=lambda d: d[1]['rrf_score'], reverse=True))[0:limit]

        return sorted_scores

