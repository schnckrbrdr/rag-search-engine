import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from constants import DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA, DEFAULT_RRF_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_score_parser = subparsers.add_parser("normalize", help="Normalize list of scores")
    normalize_score_parser.add_argument("scores", type=float, nargs='+', help="List of scores")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Search-Query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA, help="Alpha-constant to weight search")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned search results")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion Search")
    rrf_search_parser.add_argument("query", type=str, help="Search-Query")
    rrf_search_parser.add_argument("-k", type=int, nargs='?', default=DEFAULT_RRF_K, help="Weighting of top-ranked vs. low-ranked results")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned search results")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()