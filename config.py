# Embedding model
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_PREFIX = "classification: "
MATRYOSHKA_DIM = 768
TRUST_REMOTE_CODE = True
BATCH_SIZE = 256
INTERMEDIATE_SAVE_INTERVAL = 50
INTERMEDIATE_DIR = "intermediates"
SENSE_EMBEDDING_PREFIX = "classification: "
SENSE_BATCH_SIZE = 256
SENSE_INTERMEDIATE_DIR = "intermediates/senses"
ZSCORE_THRESHOLD = 2.0

# FNV-1a u64 hashing
FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME = 1099511628211

# Domain augmentation (Claude-assisted domain discovery + XGBoost classifiers)
AUGMENT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
AUGMENT_BATCH_SIZE = 5           # Domains per Claude API call
AUGMENT_API_DELAY = 0.5          # Seconds between API calls
AUGMENT_MIN_DOMAIN_WORDS = 20    # Min matched words for XGBoost training
AUGMENT_NEG_RATIO = 5            # Negative:positive sampling ratio

# XGBoost inference threshold
XGBOOST_SCORE_THRESHOLD = 0.4

# Polysemy filter: words appearing in >= this many curated domains are flagged
# and excluded from XGBoost training to prevent generic-word contamination.
POLYSEMY_DOMAIN_THRESHOLD = 20

# Words explicitly flagged as polysemous regardless of domain count.
# These are XGBoost contaminants that appear normal in curated sources
# but get picked up by nearly every classifier due to embedding proximity.
POLYSEMY_EXPLICIT_WORDS = frozenset({
    "classification",
    "outside",
})

# Domains excluded from XGBoost training/inference.
# These are stylistic, pragmatic, or meta-linguistic categories whose membership
# cannot be determined from embedding proximity. They keep only their curated
# seeds (WordNet + Claude).
XGBOOST_EXCLUDE_DOMAINS = frozenset({
    "african american vernacular english",
    "descriptive linguistics",
    "euphemism",
    "formality",
    "phonology",
    "regionalism",
    "slang",
    "trope",
    "dialect",
})

# Domain reef subdivision (v2 domain word clustering)
REEF_SCORE_THRESHOLD = 0.6       # Min xgboost score for "core" clustering words
REEF_ALPHA = 0.7                 # Hybrid weight: α * emb_cos + (1-α) * pmi_cos
REEF_KNN_K = 15                  # kNN neighbors for graph construction
REEF_LEIDEN_RESOLUTION = 1.0     # Leiden resolution parameter
REEF_MIN_COMMUNITY_SIZE = 3      # Smaller communities become noise
REEF_MIN_DOMAIN_SIZE = 10        # Skip domains with fewer core words
REEF_CHARACTERISTIC_WORDS_N = 10 # Top centroid-closest words per reef

# Archipelago clustering (domain-level grouping)
ARCH_ALPHA = 0.7                    # Hybrid weight: α * emb_cos + (1-α) * pmi_norm
ARCH_KNN_K = 10                     # kNN neighbors (444 domains, ~2% connectivity)
ARCH_LEIDEN_RESOLUTION = 1.0        # Leiden resolution parameter
ARCH_MIN_COMMUNITY_SIZE = 2         # Smaller communities become noise
ARCH_CHARACTERISTIC_DOMAINS_N = 10  # Top centroid-closest domains per archipelago

# Background model
BG_STD_FLOOR = 1.0              # Floor on adjusted bg_std to cap z-score sensitivity

# Ubiquity pruning (post-XGBoost: words appearing in many domains)
# Words in POLYSEMY_DOMAIN_THRESHOLD+ domains get cleaned up:
UBIQUITY_SCORE_FLOOR = 0.80      # Below this: DELETE
UBIQUITY_SCORE_CEILING = 0.95    # Floor to ceiling: penalize
UBIQUITY_PENALTY = 0.5           # Score multiplier for penalized range

# Domain-name cosine blending
DOMAIN_NAME_COS_ALPHA = 0.3   # effective_sim = (1-α)*centroid_sim + α*domain_name_cos

# Export thresholds
EXPORT_WEIGHT_THRESHOLD = 0.01
