import os
import pickle
from utility import load_movies, load_stopwords, tokenize

class InvertedIndex:

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}        

    def __add_document(self, doc_id: int, text: str) -> None:
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

    def build(self) -> None:
        
        movies = load_movies()
        stopwords = load_stopwords()
        self.index.clear()
        self.docmap.clear()
        for movie in movies['movies']:
            self.docmap[movie['id']] = movie
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")            

    def save(self) -> None:
        try:
            if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")):
                os.mkdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))

            pickle.dump(self.index, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "index.pkl"), "wb"))
            pickle.dump(self.docmap, open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "docmap.pkl"), "wb"))
        except Exception as e:
            print(f"Error: {e}")
    
    def load(self) -> bool:
        try:
            self.index = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "index.pkl"), "rb"))
            self.docmap = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "docmap.pkl"), "rb"))
            return True
        except Exception as e:
            return False

    

    