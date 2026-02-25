#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_command
from constants import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify sentence transformer model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding of passed text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Load or generate embeddings and verify that all documents are processed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Generate embedding for passed query")
    embedquery_parser.add_argument("query", type=str, help="Query to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Performa a search with given query")
    search_parser.add_argument("query", type=str, help="Query th search for")
    search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit of returned search results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk input into chunks of the passed size")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNK_SIZE, help="Size of the chunks")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_CHUNK_OVERLAP, help="Overlap while chunking")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.text)            
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)        
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()