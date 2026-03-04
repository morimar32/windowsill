#!/usr/bin/env bash
#
# Full v3 pipeline rebuild — from empty database to test battery.
# Run from the windowsill project root: bash v3/load.sh
#
set -euo pipefail

DB="v3/windowsill.db"
SECONDS=0

step() { echo -e "\n\033[1;36m=== Step $1: $2 ===\033[0m"; }
ok()   { echo -e "\033[1;32m  done (${SECONDS}s elapsed)\033[0m"; }

# ------------------------------------------------------------------
step 1 "Schema"
sqlite3 "$DB" < v3/schema.sql
ok

# ------------------------------------------------------------------
step 2 "Populate archipelagos & islands"
sqlite3 "$DB" < v3/populate_archipelagos_islands.sql
ok

# ------------------------------------------------------------------
step 3 "Populate towns"
sqlite3 "$DB" < v3/populate_towns.sql
ok

# ------------------------------------------------------------------
step 4 "Populate bucket islands"
sqlite3 "$DB" < v3/populate_bucket_islands.sql
ok

# ------------------------------------------------------------------
step 5 "Load WordNet vocabulary"
python v3/load_wordnet_vocab.py
ok

# ------------------------------------------------------------------
step 6 "Embed words (GPU)"
python v3/reembed_words.py
ok

# ------------------------------------------------------------------
step 7 "Compute dimension stats"
python v3/compute_dimstats.py
ok

# ------------------------------------------------------------------
step 8 "Import WDH seeds"
python v3/import_wdh.py --apply
ok

# ------------------------------------------------------------------
step 9 "Load Claude single-word seeds"
sqlite3 "$DB" < v3/populate_claude_seeds.sql
ok

# ------------------------------------------------------------------
step 9b "Load Claude compound seeds"
sqlite3 "$DB" < v3/populate_compound_seeds.sql
ok

# ------------------------------------------------------------------
step 9c "Link seed words to vocabulary"
python v3/link_seed_words.py --apply
ok

# ------------------------------------------------------------------
step 9d "Embed newly-inserted words"
python v3/reembed_words.py --missing-only
ok

# ------------------------------------------------------------------
step 10 "Sanity-check seeds"
python v3/sanity_check_seeds.py --apply
ok

# ------------------------------------------------------------------
step 11 "Detect island words"
python v3/detect_island_words.py --apply
ok

# ------------------------------------------------------------------
step 11b "Flag baseline stop words"
python v3/flag_stop_words.py --baseline
ok

# ------------------------------------------------------------------
step 12 "Train XGBoost models (per island)"
# Get all non-bucket island names from the database
ISLANDS=$(sqlite3 "$DB" "SELECT name FROM Islands WHERE is_bucket = 0 ORDER BY island_id")
ISLAND_COUNT=$(echo "$ISLANDS" | wc -l)
I=0
while IFS= read -r island; do
    I=$((I + 1))
    echo -e "\n  \033[1;33m[$I/$ISLAND_COUNT] Training: $island\033[0m"
    python v3/train_town_xgboost.py --island "$island"
done <<< "$ISLANDS"
ok

# ------------------------------------------------------------------
step 13 "Post-process XGBoost predictions"
python v3/post_process_xgb.py --apply
ok

# ------------------------------------------------------------------
step 14 "Flag ubiquitous stop words"
python v3/flag_stop_words.py --ubiquity
ok

# ------------------------------------------------------------------
step 15 "Cluster reefs (per island)"
I=0
while IFS= read -r island; do
    I=$((I + 1))
    echo -e "\n  \033[1;33m[$I/$ISLAND_COUNT] Clustering: $island\033[0m"
    python v3/cluster_reefs.py --island "$island"
done <<< "$ISLANDS"
ok

# ------------------------------------------------------------------
step 16 "Cluster bucket reefs"
python v3/cluster_bucket_reefs.py
ok

# ------------------------------------------------------------------
step 17 "Compute word stats"
python v3/compute_word_stats.py
ok

# ------------------------------------------------------------------
step 18 "Compute hierarchy stats"
python v3/compute_hierarchy_stats.py
ok

# ------------------------------------------------------------------
step 19 "Populate exports"
python v3/populate_exports.py
ok

# ------------------------------------------------------------------
step 20 "Test battery"
python v3/test_battery.py
ok

# ------------------------------------------------------------------
TOTAL=$SECONDS
MINS=$((TOTAL / 60))
SECS=$((TOTAL % 60))
echo -e "\n\033[1;32m=== Pipeline complete: ${MINS}m ${SECS}s ===\033[0m"
