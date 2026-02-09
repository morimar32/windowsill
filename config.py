MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_PREFIX = "classification: "
MATRYOSHKA_DIM = 768
TRUST_REMOTE_CODE = True
BATCH_SIZE = 256
INTERMEDIATE_SAVE_INTERVAL = 50
DB_PATH = "vector_distillery.duckdb"
ZSCORE_THRESHOLD = 2.0
PAIR_OVERLAP_THRESHOLD = 3
COMMIT_INTERVAL = 50
INTERMEDIATE_DIR = "intermediates"
SENSE_EMBEDDING_PREFIX = "classification: "
SENSE_BATCH_SIZE = 256
SENSE_INTERMEDIATE_DIR = "intermediates/senses"
COMPOSITIONALITY_THRESHOLD = 0.20
CONTAMINATION_ZSCORE_MIN = 2.0

# Island detection (Phase 9)
ISLAND_JACCARD_ZSCORE = 3.0           # Min hypergeometric z-score to include edge in graph
ISLAND_LEIDEN_RESOLUTION = 1.0       # Leiden resolution (higher = more/smaller islands)
ISLAND_CHARACTERISTIC_WORDS_N = 100  # Top N PMI-ranked words stored per island
ISLAND_MIN_COMMUNITY_SIZE = 2        # Communities smaller than this become noise (island_id = -1)
ISLAND_SUB_LEIDEN_RESOLUTION = 1.5   # Leiden resolution for sub-island detection (higher = more splitting)
ISLAND_MIN_DIMS_FOR_SUBDIVISION = 10 # Don't subdivide islands with fewer dims than this
REEF_MIN_DEPTH = 2                   # Min dims a word must activate in a reef/island/archipelago to be encoded
