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
REEF_MIN_DEPTH = 2                   # Min dims a word must activate in a reef to count as meaningfully present

# Reef refinement (Phase 10)
REEF_REFINE_MIN_DIMS = 4              # Min dims for a reef to be analyzed for misplaced dims
REEF_REFINE_LOYALTY_THRESHOLD = 1.0   # Dims with loyalty_ratio below this are considered misplaced
REEF_REFINE_MAX_ITERATIONS = 5        # Safety valve: max refinement rounds before stopping

# Universal word analytics
SENSE_SPREAD_INFLATED_THRESHOLD = 15   # Min sense_spread to flag as polysemy-inflated
DOMAIN_GENERAL_THRESHOLD = 0.75        # Min arch_concentration for v_domain_generals
ABSTRACT_DIM_THRESHOLD = 0.30          # Min universal_pct for v_abstract_dims
CONCRETE_DIM_THRESHOLD = 0.15          # Max universal_pct for v_concrete_dims

# FNV-1a u64 hashing
FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME = 1099511628211

# Valence analytics
NEGATION_PREFIXES = ['un', 'non', 'in', 'im', 'il', 'ir', 'dis', 'mal', 'mis']
POSITIVE_DIM_VALENCE_THRESHOLD = -0.15   # valence below this = positive-pole dim
NEGATIVE_DIM_VALENCE_THRESHOLD = 0.15    # valence above this = negative-pole dim

# Reef scoring constants
N_REEFS = 207
N_ISLANDS = 52
N_ARCHS = 4
BM25_K1 = 1.2
BM25_B = 0.75

# Composite weight formula coefficients
# weight = (containment * lift) * pos_similarity^ALPHA_POS
#          * exp(-ALPHA_VAL * |valence_gap|)
#          * exp(-ALPHA_SPEC * |specificity_gap|)
COMPOSITE_ALPHA_POS = 2.0    # POS similarity exponent (higher = stronger gating)
COMPOSITE_ALPHA_VAL = 2.92   # Valence gap decay rate (higher = more suppression)
COMPOSITE_ALPHA_SPEC = 0.5   # Specificity gap decay rate (mild effect)

# Export thresholds
EXPORT_WEIGHT_THRESHOLD = 0.01
