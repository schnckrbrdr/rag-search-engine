import os
import pickle
from utility import load_movies, load_stopwords, tokenize
from constants import DEFAULT_BM25_K1, DEFAULT_BM25_B
import collections
import math

class InvertedIndex:

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.stopwords = load_stopwords()   

    def __add_document(self, doc_id: int, text: str) -> None:
        
        tokens = tokenize(text, self.stopwords)
        self.doc_lengths[doc_id] = len(tokens)        
        self.term_frequencies[doc_id] = collections.Counter()
        for token in tokens:            
            if token in self.index:
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]
            self.term_frequencies[doc_id][token] += 1        
        for entry in self.index:
            self.index[entry] = sorted(self.index[entry])

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) > 0:
            total_token_count = 0
            for entry in self.doc_lengths:
                total_token_count += self.doc_lengths[entry]
            return total_token_count / len(self.doc_lengths)
        else:
            return 0.0


    def get_documents(self, term: str) -> list[int]:
        if term in self.index:
            return self.index[term]
        else:
            return []
        
    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id in self.term_frequencies:
            tokens = tokenize(term, self.stopwords)
            if len(tokens) != 1:
                raise Exception("Error: Must supply exactly one search term")
            else:                
                return self.term_frequencies[doc_id][tokens[0]] if tokens[0] in self.term_frequencies[doc_id] else 0
        else:
            return 0

    def get_idf(self, term: str) -> float:        
        tokens = tokenize(term, self.stopwords)
        if len(tokens) != 1:
            raise Exception("Error: Must supply exactly one serch term")
        else:
            term_match_doc_count = 0
            total_doc_count = 0
            for entry in self.docmap:
                if self.get_tf(entry, term) > 0:
                    term_match_doc_count += 1
                total_doc_count += 1
            
            return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tokens = tokenize(term, self.stopwords)
        if len(tokens) != 1:
            raise Exception("Error: Must supply exactly one search term")
        else:
            try:
                return self.get_tf(doc_id, term) * self.get_idf(term)
            except Exception as e:
                print("Error: TFIDF could not be calculated")

    def get_bm25_idf(self, term: str) -> float:
        try:
            # document frequency - how many documents in the whole dataset contain the search term
            df = 0
            # document count - how many documents are in the dataset
            n = 0
            for entry in self.docmap:
                if self.get_tf(entry, term) > 0:
                    df += 1
                n += 1                    
            return math.log((n - df + 0.5) / (df + 0.5) + 1)                
        except Exception as e:
            print("Error: BM25_IDF could not be calculated") 

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B) -> float:
        if doc_id in self.doc_lengths:
            doc_length = self.doc_lengths[doc_id]
        else:
            doc_length = 0
        
        # Length normalization factor        
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())        
        tf = self.get_tf(doc_id, term)
                         
        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def build(self) -> None:
        movies = load_movies()
        self.index.clear()
        self.docmap.clear()
        count = 0
        for movie in movies['movies']:
            count += 1
            self.docmap[movie['id']] = movie
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")            
            if count % 100 == 0:
                print(f"{(count * 100)/len(movies['movies'])}%")

    def save(self) -> None:
        try:
            if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")):
                os.mkdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))

            pickle.dump(self.index, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "index.pkl"), "wb"))
            pickle.dump(self.docmap, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "docmap.pkl"), "wb"))
            pickle.dump(self.term_frequencies, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "term_frequencies.pkl"), "wb"))
            pickle.dump(self.doc_lengths, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "doc_lengths.pkl"), "wb"))
        except Exception as e:
            print(f"Error: {e}")
    
    def load(self) -> bool:
        try:
            self.index = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "index.pkl"), "rb"))
            self.docmap = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "docmap.pkl"), "rb"))
            self.term_frequencies = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "term_frequencies.pkl"), "rb"))
            self.doc_lengths = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "doc_lengths.pkl"), "rb"))
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    

    