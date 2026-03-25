import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command
from constants import DEFAULT_SEARCH_LIMIT

def main():
    
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Search for query and summarize the results")
    summarize_parser.add_argument("query", type=str, help="Query")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit of search results")

    citation_parser = subparsers.add_parser("citations", help="Perform query and add citations to finds")
    citation_parser.add_argument("query", type=str, help="Query")
    citation_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit of search results")

    question_parser = subparsers.add_parser("question", help="Ask a question about movies")
    question_parser.add_argument("question", type=str, help="Question")
    question_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit of search results")

    args = parser.parse_args()

    match args.command:
        case "rag":            
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case "citations":
            citations_command(args.query, args.limit)
        case "question":
            question_command(args.question, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()