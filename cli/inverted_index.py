import os
import pickle
from utility import load_movies, load_stopwords, tokenize
import collections
import math

class InvertedIndex:

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.stopwords = load_stopwords()   

    def __add_document(self, doc_id: int, text: str) -> None:
        
        tokens = tokenize(text, self.stopwords)
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

    def get_documents(self, term: str) -> list[int]:
        if term in self.index:
            return self.index[term]
        else:
            return []
        
    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id in self.term_frequencies:
            tokens = tokenize(term, self.stopwords)
            if len(tokens) != 1:
                raise Exception("Error: Too many or no query tokens")
            else:                
                return self.term_frequencies[doc_id][tokens[0]] if tokens[0] in self.term_frequencies[doc_id] else 0
        else:
            return 0

    def get_idf(self, term: str) -> float:        
        tokens = tokenize(term, self.stopwords)
        if len(tokens) != 1:
            raise Exception("Error: Too many or no query tokens")
        else:
            term_match_doc_count = 0
            total_doc_count = 0
            for entry in self.docmap:
                if self.get_tf(entry, term) > 0:
                    term_match_doc_count += 1
                total_doc_count += 1
            
            return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


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
        except Exception as e:
            print(f"Error: {e}")
    
    def load(self) -> bool:
        try:
            self.index = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "index.pkl"), "rb"))
            self.docmap = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "docmap.pkl"), "rb"))
            self.term_frequencies = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "term_frequencies.pkl"), "rb"))
            return True
        except Exception as e:
            return False

    

    