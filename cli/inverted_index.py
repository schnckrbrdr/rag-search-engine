from utility import tokenize

class InvertedIndex:

    def __init__(self, stopwords: list[str]):
        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords

    def __add_document(self, doc_id: int, text: str):
        pass

    def get_documents(self, term: str) -> list[int]:
        pass

    def build(self):
        pass

    def save(self):
        pass
    
    

    

    