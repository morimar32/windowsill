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

# Export thresholds
EXPORT_WEIGHT_THRESHOLD = 0.01
