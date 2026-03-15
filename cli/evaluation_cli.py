import argparse

from lib.hybrid_search import HybridSearch
from utility import load_json, load_movies
from constants import GOLDEN_DATASET_PATH, DEFAULT_RRF_K

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_json(GOLDEN_DATASET_PATH)

    movies = load_movies()

    hybrid_search = HybridSearch(movies['movies'])

    for test_case in golden_dataset['test_cases']:
        relevant_titles = test_case['relevant_docs']
        results = hybrid_search.rrf_search(test_case['query'], DEFAULT_RRF_K, limit)

        found_titles = []
        found_relevant = 0

        for result in results[:limit]:
            found_title = result[1]['document']['title']
            found_titles.append(found_title)
            if found_title in relevant_titles:
                found_relevant += 1

        precision_at_k = found_relevant / len(results)

        print(f"- Query: {test_case['query']}")
        print(f"  - Precision@{limit}: {precision_at_k:.4f}")
        print(f"  - Retrieved: {', '.join(found_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}\n")

if __name__ == "__main__":
    main()