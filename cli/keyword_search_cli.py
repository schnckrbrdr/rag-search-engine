#!/usr/bin/env python3

import argparse
from utility import load_stopwords, tokenize
from inverted_index import InvertedIndex
from constants import DEFAULT_BM25_K1, DEFAULT_BM25_B

def keyword_search(query_string: str, inverted_index: InvertedIndex, limit: int = 5) -> list[dict]:
    
    matches = []
    stopwords = load_stopwords()
    
    if stopwords:
        query_tokens = tokenize(query_string, stopwords)
        doc_ids = []
        for query_token in query_tokens:
            doc_ids.extend(inverted_index.get_documents(query_token))
            doc_ids = list(set(doc_ids))
            if len(doc_ids) >= limit:
                break
        matches = []
        for doc_id in doc_ids:
            matches.append(inverted_index.docmap[doc_id])            
        sorted_matches = sorted(matches, key=lambda d: d['id'])
        return sorted_matches[0:min(len(sorted_matches), limit)]
    else:
        return []

def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save() 

def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    if not inverted_index.load():
        print("Index not found. Build index first!")
    else:
        try:
            return inverted_index.get_bm25_idf(term)
        except Exception as e:
            print(f"Error: {e}") 

def bm25_tf_command(doc_id: int, term: str, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B) -> float:
    inverted_index = InvertedIndex()
    if not inverted_index.load():
        print("Index not found. Build index first!")
    else:
        try:
            return inverted_index.get_bm25_tf(doc_id, term, k1, b)
        except Exception as e:
            print(f"Error: {e}")            
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="(Re)build index of movies")

    term_frequency_parser = subparsers.add_parser("tf", help="Determine occurences of search term in given DocumentID")
    term_frequency_parser.add_argument("doc_id", type=int, help="DocumentID to search")
    term_frequency_parser.add_argument("term", type=str, help="Term to determine occurences of")

    inverse_document_frequency_parser = subparsers.add_parser("idf", help="Calculate the inverse document frequency of given term")
    inverse_document_frequency_parser.add_argument("term", type=str, help="Term to get inverse frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TFIDF-Score of a term in context of a given DocID relative of the whole index of documents")
    tfidf_parser.add_argument("doc_id", type=int, help="DocumentID to search term in")
    tfidf_parser.add_argument("term", type=str, help="Term to get TDIDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=DEFAULT_BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=DEFAULT_BM25_B, help="Tunable BM25 B parameter")

    args = parser.parse_args()

    match args.command:
        case "search":

            inverted_index = InvertedIndex()
            if not inverted_index.load():
                print("Index not found. Build index first!")
            else:
                sorted_matches = keyword_search(args.query, inverted_index)
                for i in range(0, min(len(sorted_matches), 6)):
                    print(f"{i + 1}. {sorted_matches[i]['title']}")

        case "tf":
            inverted_index = InvertedIndex()
            if not inverted_index.load():
                print("Index not found. Build index first!")
            else:
                try:
                    print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {inverted_index.get_tf(args.doc_id, args.term)}")                    
                except Exception as e:
                    print(f"Error: {e}")

        case "idf":
            inverted_index = InvertedIndex()
            if not inverted_index.load():
                print("Index not found. Build index first!")
            else:
                try:
                    idf = inverted_index.get_idf(args.term)
                    print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
                except Exception as e:
                    print(f"Error: {e}")

        case "tfidf":
            inverted_index = InvertedIndex()
            if not inverted_index.load():
                print("Index not found. Build index first!")
            else:
                try:                    
                    print(f"The TFIDF-Score of '{args.term}' in DocID '{args.doc_id}' relative to the whole dataset is: {inverted_index.get_tfidf(args.doc_id, args.term):.2f}")
                except Exception as e:
                    print(f"Error: {e}")

        case "bm25idf":
            try:
                print(f"BM25 IDF score of '{args.term}': {bm25_idf_command(args.term):.2f}")
            except Exception as e:
                print(f"Error: {e}")

        case "bm25tf":
            try:
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf_command(args.doc_id, args.term, args.k1):.2f}")
            except Exception as e:
                print(f"Error: {e}")

        case "build":
            try:
                build_command()

            except Exception as e:
                print(f"Error: {e}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()