import os

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 5
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_ALPHA = 0.5
DEFAULT_RRF_K = 60

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")