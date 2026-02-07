import os
import pickle
from utility import load_movies, tokenize

class InvertedIndex:

    def __init__(self, stopwords: list[str]):
        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize(text, self.stopwords)
        for token in tokens:            
            if token in self.index:
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]
        for entry in self.index:
            self.index[entry] = sorted(self.index[entry])            

    def get_documents(self, term: str) -> list[int]:
        if term in self.index:
            return self.index[term]
        else:
            return []

    def build(self):
        
        movies = load_movies()
        self.index.clear()
        self.docmap.clear()
        for movie in movies['movies']:
            self.docmap[movie['id']] = movie
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")            

    def save(self):
        if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")):
            os.mkdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))

        pickle.dump(self.index, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "index.pkl"), "wb"))
        pickle.dump(self.docmap, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "docmap.pkl"), "wb"))
    
    

    

    