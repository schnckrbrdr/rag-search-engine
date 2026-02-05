#!/usr/bin/env python3

import argparse
import json
import os

def keyword_search(search_string):
    print(os.path.dirname(os.path.dirname(__file__)))
    movies_file = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "movies.json"))
    movies_json = json.load(movies_file)
    matches = []
    for movie in movies_json['movies']:
        if search_string.lower() in movie["title"].lower():
            matches.append(movie)
        sorted_matches = sorted(matches, key=lambda d: d['id'])        
    return sorted_matches

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                sorted_matches = keyword_search(args.query)
                for i in range(0, min(len(sorted_matches), 5)):
                    print(f"{i + 1}. {sorted_matches[i]['title']}")
            except Exception as e:
                print(f"Error: {e}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()