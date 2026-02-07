#!/usr/bin/env python3

import argparse
from utility import load_movies, load_stopwords, tokenize
from inverted_index import InvertedIndex

def keyword_search(query_string: str, movies_json: list[dict], stopwords: list[str], limit: int = 5) -> list[dict]:
    
    matches = []
    query_tokens = tokenize(query_string, stopwords)    
    # Iterate over all movies in list (via dict-entry)
    for movie in movies_json['movies']:        
        
        # tokenize and sanitize movie title
        movie_tokens = tokenize(movie["title"], stopwords)
        
        # set backward counter to end of movie-title tokens
        i = len(query_tokens) - 1        
        
        # iterate over all movie-title tokens
        while i >= 0:
            i -= 1
            
            # set backward counter to end of query tokens
            j = len(movie_tokens) - 1
            
            # iterate over all query tokens
            while j >= 0:
                j -= 1
                
                # if a qery token is contained in a movie token, add movie to list of matches and end loop
                if query_tokens[i] in movie_tokens[j]:
                    matches.append(movie)

                    # Force end of loop for current movie if match was found                   
                    i = -1
                    j = -1
        
        # End search when result limit of search is reached
        if len(matches) >= limit:            
            break

    sorted_matches = sorted(matches, key=lambda d: d['id'])
    return sorted_matches

def build_command():
    stopwords = load_stopwords()
    inverted_index = InvertedIndex(stopwords)
    inverted_index.build()
    inverted_index.save()
    docs = inverted_index.get_documents("merida")
    if len(docs) > 0:
        print(f"First document for token 'merida' = {docs[0]}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="(Re)build index of movies")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:  
                movies_json = load_movies()              
                stopwords = load_stopwords()
                if movies_json != None and stopwords != None:
                    sorted_matches = keyword_search(args.query, movies_json, stopwords)
                    for i in range(0, min(len(sorted_matches), 6)):
                        print(f"{i + 1}. {sorted_matches[i]['title']}")
                else:
                    print("Error reading data!")
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