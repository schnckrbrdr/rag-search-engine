from utility import load_movies, format_documents
from constants import DEFAULT_RRF_K, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search import HybridSearch
from lib.llm import LLM

def rag_command(query: str):

    movies = load_movies()

    hybrid_search = HybridSearch(movies['movies'])

    results = hybrid_search.rrf_search(query, DEFAULT_RRF_K, DEFAULT_SEARCH_LIMIT)

    formatted_documents = format_documents(results)

    llm  = LLM()

    rag_response = llm.rag_request(query, formatted_documents)

    print("Search Results:")

    for result in results:
        print(f"- {result[1]['document']['title']}")

    print("RAG Response:")
    print(rag_response)

def summarize_command(query: str, limit: int):
    movies = load_movies()

    hybrid_search = HybridSearch(movies['movies'])

    results = hybrid_search.rrf_search(query, DEFAULT_RRF_K, limit)

    formatted_documents = format_documents(results)

    llm = LLM()

    rag_summary = llm.rag_summarize(query, formatted_documents)

    print("Search Results:")

    for result in results:
        print(f"- {result[1]['document']['title']}")

    print("LLM Summary:")
    print(rag_summary)

def citations_command(query: str, limit: int):

    movies = load_movies()

    hybrid_search = HybridSearch(movies['movies'])

    results = hybrid_search.rrf_search(query, DEFAULT_RRF_K, limit)

    formatted_documents = format_documents(results)

    llm = LLM()

    rag_citations = llm.rag_citations(query, formatted_documents)

    print("Search Results:")

    for result in results:
        print(f"- {result[1]['document']['title']}")

    print("LLM Answer:")
    print(rag_citations)

def question_command(question: str, limit: int):

    movies = load_movies()

    hybrid_search = HybridSearch(movies['movies'])

    results = hybrid_search.rrf_search(question, DEFAULT_RRF_K, limit)

    formatted_documents = format_documents(results)

    llm = LLM()

    rag_answer = llm.rag_question(question, formatted_documents)

    print("Search Results:")

    for result in results:
        print(f"- {result[1]['document']['title']}")

    print("Answer:")
    print(rag_answer)