import os

import duckdb
import numpy as np
import pandas as pd

import config


def get_connection(db_path=None, read_only=False):
    if db_path is None:
        db_path = config.DB_PATH
    return duckdb.connect(db_path, read_only=read_only)


def create_schema(con):
    dim = config.MATRYOSHKA_DIM
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS words (
            word_id INTEGER PRIMARY KEY,
            word TEXT NOT NULL,
            total_dims INTEGER DEFAULT 0,
            embedding FLOAT[{dim}]
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_stats (
            dim_id INTEGER PRIMARY KEY,
            mean DOUBLE,
            std DOUBLE,
            min_val DOUBLE,
            max_val DOUBLE,
            median DOUBLE,
            skewness DOUBLE,
            kurtosis DOUBLE,
            threshold DOUBLE,
            threshold_method TEXT,
            n_members INTEGER,
            selectivity DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_memberships (
            dim_id INTEGER NOT NULL,
            word_id INTEGER NOT NULL,
            value DOUBLE,
            z_score DOUBLE,
            PRIMARY KEY (dim_id, word_id)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS word_pair_overlap (
            word_id_a INTEGER NOT NULL,
            word_id_b INTEGER NOT NULL,
            shared_dims INTEGER,
            PRIMARY KEY (word_id_a, word_id_b)
        )
    """)


def create_indexes(con):
    con.execute("CREATE INDEX IF NOT EXISTS idx_dm_word ON dim_memberships(word_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dm_dim ON dim_memberships(dim_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_words_total ON words(total_dims DESC)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ds_selectivity ON dim_stats(selectivity)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wpo_a ON word_pair_overlap(word_id_a)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wpo_b ON word_pair_overlap(word_id_b)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wpo_shared ON word_pair_overlap(shared_dims DESC)")


def insert_words(con, words, embeddings):
    word_ids = [w[0] for w in words]
    word_texts = [w[1] for w in words]
    embedding_lists = [emb.tolist() for emb in embeddings]

    df = pd.DataFrame({
        "word_id": word_ids,
        "word": word_texts,
        "total_dims": [0] * len(words),
        "embedding": embedding_lists,
    })

    con.execute("DELETE FROM words")
    con.execute("INSERT INTO words SELECT * FROM df")
    print(f"  Inserted {len(words)} words into database")


def insert_dim_stats(con, dim_stats_list):
    if not dim_stats_list:
        return
    df = pd.DataFrame(dim_stats_list)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO dim_stats ({cols}) SELECT * FROM df")


def insert_dim_memberships(con, memberships):
    if not memberships:
        return
    df = pd.DataFrame(memberships, columns=["dim_id", "word_id", "value", "z_score"])
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO dim_memberships ({cols}) SELECT * FROM df")


def load_embedding_matrix(con):
    result = con.execute("SELECT word_id, embedding FROM words ORDER BY word_id").fetchall()
    word_ids = [r[0] for r in result]
    matrix = np.array([r[1] for r in result], dtype=np.float32)
    return matrix, word_ids


def migrate_schema(con):
    dim = config.MATRYOSHKA_DIM

    # Add columns to words table
    for col, typedef in [("pos", "TEXT"), ("category", "TEXT"), ("word_count", "INTEGER DEFAULT 1")]:
        try:
            con.execute(f"ALTER TABLE words ADD COLUMN {col} {typedef}")
        except duckdb.CatalogException:
            pass

    # Add POS enrichment columns to dim_stats
    for col in ["verb_enrichment", "adj_enrichment", "adv_enrichment", "noun_pct"]:
        try:
            con.execute(f"ALTER TABLE dim_stats ADD COLUMN {col} DOUBLE")
        except duckdb.CatalogException:
            pass

    # Add compound_support to dim_memberships
    try:
        con.execute("ALTER TABLE dim_memberships ADD COLUMN compound_support INTEGER DEFAULT 0")
    except duckdb.CatalogException:
        pass

    # Add hypergeometric z-score columns to dim_jaccard
    for col, typedef in [("expected_intersection", "DOUBLE"), ("z_score", "DOUBLE")]:
        try:
            con.execute(f"ALTER TABLE dim_jaccard ADD COLUMN {col} {typedef}")
        except duckdb.CatalogException:
            pass

    # Add island_name to island_stats
    try:
        con.execute("ALTER TABLE island_stats ADD COLUMN island_name TEXT")
    except duckdb.CatalogException:
        pass

    # Add depth-based metrics to island_stats
    for col, typedef in [("n_core_words", "INTEGER"), ("median_word_depth", "DOUBLE")]:
        try:
            con.execute(f"ALTER TABLE island_stats ADD COLUMN {col} {typedef}")
        except duckdb.CatalogException:
            pass

    # Add specificity band to words
    try:
        con.execute("ALTER TABLE words ADD COLUMN specificity INTEGER DEFAULT 0")
    except duckdb.CatalogException:
        pass

    # Add archipelago encoding bit-position columns to island_stats
    for col, typedef in [("arch_column", "TEXT"), ("arch_bit", "INTEGER")]:
        try:
            con.execute(f"ALTER TABLE island_stats ADD COLUMN {col} {typedef}")
        except duckdb.CatalogException:
            pass

    # Add archipelago bitmask columns to words
    for col in ["archipelago", "archipelago_ext", "reef_0", "reef_1", "reef_2", "reef_3", "reef_4", "reef_5"]:
        try:
            con.execute(f"ALTER TABLE words ADD COLUMN {col} BIGINT DEFAULT 0")
        except duckdb.CatalogException:
            pass

    # word_pos table
    con.execute("""
        CREATE TABLE IF NOT EXISTS word_pos (
            word_id INTEGER NOT NULL,
            pos TEXT NOT NULL,
            synset_count INTEGER,
            PRIMARY KEY (word_id, pos)
        )
    """)

    # word_components table
    con.execute("""
        CREATE TABLE IF NOT EXISTS word_components (
            compound_word_id INTEGER NOT NULL,
            component_word_id INTEGER,
            component_text TEXT NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (compound_word_id, position)
        )
    """)

    # word_senses table
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS word_senses (
            sense_id INTEGER PRIMARY KEY,
            word_id INTEGER NOT NULL,
            pos TEXT NOT NULL,
            synset_name TEXT NOT NULL,
            gloss TEXT NOT NULL,
            sense_embedding FLOAT[{dim}],
            total_dims INTEGER DEFAULT 0
        )
    """)

    # sense_dim_memberships table
    con.execute("""
        CREATE TABLE IF NOT EXISTS sense_dim_memberships (
            dim_id INTEGER NOT NULL,
            sense_id INTEGER NOT NULL,
            value DOUBLE,
            z_score DOUBLE,
            PRIMARY KEY (dim_id, sense_id)
        )
    """)

    # New indexes
    con.execute("CREATE INDEX IF NOT EXISTS idx_wp_word ON word_pos(word_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wc_compound ON word_components(compound_word_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wc_component ON word_components(component_word_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ws_word ON word_senses(word_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sdm_sense ON sense_dim_memberships(sense_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sdm_dim ON sense_dim_memberships(dim_id)")

    # Island detection tables (Phase 9)
    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_jaccard (
            dim_id_a INTEGER NOT NULL,
            dim_id_b INTEGER NOT NULL,
            intersection_size INTEGER NOT NULL,
            union_size INTEGER NOT NULL,
            jaccard DOUBLE NOT NULL,
            expected_intersection DOUBLE,
            z_score DOUBLE,
            PRIMARY KEY (dim_id_a, dim_id_b)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_islands (
            dim_id INTEGER NOT NULL,
            island_id INTEGER NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0,
            parent_island_id INTEGER,
            PRIMARY KEY (dim_id, generation)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS island_stats (
            island_id INTEGER NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0,
            n_dims INTEGER NOT NULL,
            n_words INTEGER NOT NULL,
            avg_internal_jaccard DOUBLE,
            max_internal_jaccard DOUBLE,
            min_internal_jaccard DOUBLE,
            modularity_contribution DOUBLE,
            parent_island_id INTEGER,
            island_name TEXT,
            n_core_words INTEGER,
            median_word_depth DOUBLE,
            PRIMARY KEY (island_id, generation)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS island_characteristic_words (
            island_id INTEGER NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0,
            word_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            pmi DOUBLE NOT NULL,
            island_freq DOUBLE NOT NULL,
            corpus_freq DOUBLE NOT NULL,
            n_dims_in_island INTEGER NOT NULL,
            PRIMARY KEY (island_id, generation, word_id)
        )
    """)

    con.execute("CREATE INDEX IF NOT EXISTS idx_dj_a ON dim_jaccard(dim_id_a)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dj_b ON dim_jaccard(dim_id_b)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_di_island ON dim_islands(island_id, generation)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_is_gen ON island_stats(generation)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_icw_island ON island_characteristic_words(island_id, generation)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_icw_word ON island_characteristic_words(word_id)")

    print("  Schema migration complete")


def insert_word_pos(con, rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["word_id", "pos", "synset_count"])
    con.execute("INSERT INTO word_pos SELECT * FROM df")


def insert_word_components(con, rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["compound_word_id", "component_word_id", "component_text", "position"])
    con.execute("INSERT INTO word_components SELECT * FROM df")


def insert_word_senses(con, rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["sense_id", "word_id", "pos", "synset_name", "gloss", "sense_embedding", "total_dims"])
    con.execute("INSERT INTO word_senses SELECT * FROM df")


def insert_sense_dim_memberships(con, memberships):
    if not memberships:
        return
    df = pd.DataFrame(memberships, columns=["dim_id", "sense_id", "value", "z_score"])
    con.execute("INSERT INTO sense_dim_memberships SELECT * FROM df")


def load_sense_embedding_matrix(con):
    result = con.execute(
        "SELECT sense_id, sense_embedding FROM word_senses WHERE sense_embedding IS NOT NULL ORDER BY sense_id"
    ).fetchall()
    sense_ids = [r[0] for r in result]
    matrix = np.array([r[1] for r in result], dtype=np.float32)
    return matrix, sense_ids


def get_word_count(con):
    return con.execute("SELECT COUNT(*) FROM words").fetchone()[0]


def get_word_id_map(con):
    rows = con.execute("SELECT word, word_id FROM words").fetchall()
    return {r[0]: r[1] for r in rows}


def get_id_word_map(con):
    rows = con.execute("SELECT word_id, word FROM words").fetchall()
    return {r[0]: r[1] for r in rows}


def _table_exists(con, table_name):
    result = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result[0] > 0


def run_integrity_checks(con):
    print("  Running integrity checks...")
    passed = 0
    failed = 0
    skipped = 0

    # FK integrity checks
    fk_checks = [
        ("dim_memberships", "word_id", "dim_memberships", "words", "word_id", None),
        ("word_pos", "word_id", "word_pos", "words", "word_id", None),
        ("word_components", "compound_word_id", "word_components", "words", "word_id", None),
        ("word_components", "component_word_id", "word_components", "words", "word_id", "c.component_word_id IS NOT NULL"),
        ("word_senses", "word_id", "word_senses", "words", "word_id", None),
        ("sense_dim_memberships", "sense_id", "sense_dim_memberships", "word_senses", "sense_id", None),
        ("compositionality", "word_id", "compositionality", "words", "word_id", None),
        ("dim_islands", "dim_id", "dim_islands", "dim_stats", "dim_id", None),
        ("island_characteristic_words", "word_id", "island_characteristic_words", "words", "word_id", None),
    ]

    for child_table, child_col, guard_table, parent_table, parent_col, where_clause in fk_checks:
        if not _table_exists(con, guard_table):
            print(f"  [SKIP] {child_table}.{child_col} -> {parent_table}: table not found")
            skipped += 1
            continue
        where = f"AND {where_clause}" if where_clause else ""
        count = con.execute(f"""
            SELECT COUNT(*) FROM {child_table} c
            LEFT JOIN {parent_table} p ON c.{child_col} = p.{parent_col}
            WHERE p.{parent_col} IS NULL {where}
        """).fetchone()[0]
        if count == 0:
            print(f"  [PASS] {child_table}.{child_col} -> {parent_table}: 0 orphans")
            passed += 1
        else:
            print(f"  [FAIL] {child_table}.{child_col} -> {parent_table}: {count} orphans")
            failed += 1

    # word_pair_overlap: both columns
    if _table_exists(con, "word_pair_overlap"):
        for col in ("word_id_a", "word_id_b"):
            count = con.execute(f"""
                SELECT COUNT(*) FROM word_pair_overlap c
                LEFT JOIN words p ON c.{col} = p.word_id
                WHERE p.word_id IS NULL
            """).fetchone()[0]
            if count == 0:
                print(f"  [PASS] word_pair_overlap.{col} -> words: 0 orphans")
                passed += 1
            else:
                print(f"  [FAIL] word_pair_overlap.{col} -> words: {count} orphans")
                failed += 1
    else:
        print("  [SKIP] word_pair_overlap.word_id_a -> words: table not found")
        print("  [SKIP] word_pair_overlap.word_id_b -> words: table not found")
        skipped += 2

    # Sanity checks
    if _table_exists(con, "dim_stats"):
        dim_count = con.execute("SELECT COUNT(*) FROM dim_stats").fetchone()[0]
        expected = config.MATRYOSHKA_DIM
        if dim_count == expected:
            print(f"  [PASS] dim_stats has exactly {expected} rows")
            passed += 1
        else:
            print(f"  [FAIL] dim_stats has {dim_count} rows (expected {expected})")
            failed += 1
    else:
        print("  [SKIP] dim_stats row count: table not found")
        skipped += 1

    if _table_exists(con, "words"):
        word_count = con.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        if word_count > 100_000:
            print(f"  [PASS] words has {word_count:,} rows (> 100,000)")
            passed += 1
        else:
            print(f"  [FAIL] words has {word_count:,} rows (expected > 100,000)")
            failed += 1
    else:
        print("  [SKIP] words row count: table not found")
        skipped += 1

    if _table_exists(con, "dim_memberships"):
        dm_count = con.execute("SELECT COUNT(*) FROM dim_memberships").fetchone()[0]
        if dm_count > 0:
            print(f"  [PASS] dim_memberships is non-empty ({dm_count:,} rows)")
            passed += 1
        else:
            print(f"  [FAIL] dim_memberships is empty")
            failed += 1
    else:
        print("  [SKIP] dim_memberships non-empty: table not found")
        skipped += 1

    # Archipelago encoding: bit position uniqueness
    if _table_exists(con, "island_stats"):
        try:
            dup_count = con.execute("""
                SELECT COUNT(*) FROM (
                    SELECT arch_column, arch_bit, COUNT(*) as cnt
                    FROM island_stats
                    WHERE arch_column IS NOT NULL
                    GROUP BY arch_column, arch_bit
                    HAVING cnt > 1
                )
            """).fetchone()[0]
            if dup_count == 0:
                has_encoding = con.execute(
                    "SELECT COUNT(*) FROM island_stats WHERE arch_column IS NOT NULL"
                ).fetchone()[0]
                if has_encoding > 0:
                    print(f"  [PASS] archipelago encoding: {has_encoding} bit positions, 0 duplicates")
                    passed += 1
                else:
                    print(f"  [SKIP] archipelago encoding: no bit positions assigned yet")
                    skipped += 1
            else:
                print(f"  [FAIL] archipelago encoding: {dup_count} duplicate bit positions")
                failed += 1
        except Exception:
            print("  [SKIP] archipelago encoding: columns not present")
            skipped += 1
    else:
        print("  [SKIP] archipelago encoding: island_stats not found")
        skipped += 1

    print(f"  Integrity: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


def rebuild_indexes(con):
    indexes = [
        ("idx_dm_word", "CREATE INDEX idx_dm_word ON dim_memberships(word_id)", None),
        ("idx_dm_dim", "CREATE INDEX idx_dm_dim ON dim_memberships(dim_id)", None),
        ("idx_words_total", "CREATE INDEX idx_words_total ON words(total_dims DESC)", None),
        ("idx_ds_selectivity", "CREATE INDEX idx_ds_selectivity ON dim_stats(selectivity)", None),
        ("idx_wpo_a", "CREATE INDEX idx_wpo_a ON word_pair_overlap(word_id_a)", None),
        ("idx_wpo_b", "CREATE INDEX idx_wpo_b ON word_pair_overlap(word_id_b)", None),
        ("idx_wpo_shared", "CREATE INDEX idx_wpo_shared ON word_pair_overlap(shared_dims DESC)", None),
        ("idx_wp_word", "CREATE INDEX idx_wp_word ON word_pos(word_id)", "word_pos"),
        ("idx_wc_compound", "CREATE INDEX idx_wc_compound ON word_components(compound_word_id)", "word_components"),
        ("idx_wc_component", "CREATE INDEX idx_wc_component ON word_components(component_word_id)", "word_components"),
        ("idx_ws_word", "CREATE INDEX idx_ws_word ON word_senses(word_id)", "word_senses"),
        ("idx_sdm_sense", "CREATE INDEX idx_sdm_sense ON sense_dim_memberships(sense_id)", "sense_dim_memberships"),
        ("idx_sdm_dim", "CREATE INDEX idx_sdm_dim ON sense_dim_memberships(dim_id)", "sense_dim_memberships"),
        ("idx_dj_a", "CREATE INDEX idx_dj_a ON dim_jaccard(dim_id_a)", "dim_jaccard"),
        ("idx_dj_b", "CREATE INDEX idx_dj_b ON dim_jaccard(dim_id_b)", "dim_jaccard"),
        ("idx_di_island", "CREATE INDEX idx_di_island ON dim_islands(island_id, generation)", "dim_islands"),
        ("idx_is_gen", "CREATE INDEX idx_is_gen ON island_stats(generation)", "island_stats"),
        ("idx_icw_island", "CREATE INDEX idx_icw_island ON island_characteristic_words(island_id, generation)", "island_characteristic_words"),
        ("idx_icw_word", "CREATE INDEX idx_icw_word ON island_characteristic_words(word_id)", "island_characteristic_words"),
    ]

    dropped = 0
    created = 0
    idx_skipped = 0

    for name, create_sql, required_table in indexes:
        if required_table and not _table_exists(con, required_table):
            idx_skipped += 1
            continue
        con.execute(f"DROP INDEX IF EXISTS {name}")
        dropped += 1
        con.execute(create_sql)
        created += 1

    print(f"  Indexes: {dropped} dropped, {created} created, {idx_skipped} skipped")


def run_storage_optimization(con):
    print("  Running ANALYZE...")
    con.execute("ANALYZE")
    print("  Running CHECKPOINT...")
    con.execute("FORCE CHECKPOINT")
    print("  Storage optimization complete")


def print_database_report(con, db_path):
    print("  ========== DATABASE MAINTENANCE REPORT ==========")

    # File size
    if os.path.exists(db_path):
        size_bytes = os.path.getsize(db_path)
        if size_bytes >= 1_073_741_824:
            size_str = f"{size_bytes / 1_073_741_824:.2f} GB"
        elif size_bytes >= 1_048_576:
            size_str = f"{size_bytes / 1_048_576:.1f} MB"
        elif size_bytes >= 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes} bytes"
        print(f"  Database: {db_path} ({size_str})")
    else:
        print(f"  Database: {db_path} (file not found)")

    # Row counts
    tables = [
        "words", "dim_stats", "dim_memberships", "word_pair_overlap",
        "word_pos", "word_components", "word_senses",
        "sense_dim_memberships", "compositionality",
        "dim_jaccard", "dim_islands", "island_stats", "island_characteristic_words",
    ]
    total_rows = 0
    for table in tables:
        if _table_exists(con, table):
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            total_rows += count
            print(f"  {table:30s} {count:>12,} rows")
        else:
            print(f"  {table:30s} (not found)")
    print(f"  {'TOTAL':30s} {total_rows:>12,} rows")

    # Index count
    try:
        idx_count = con.execute("SELECT COUNT(*) FROM duckdb_indexes()").fetchone()[0]
        print(f"  Indexes: {idx_count}")
    except Exception:
        print("  Indexes: (unable to query)")

    # Storage info
    try:
        row = con.execute("SELECT * FROM pragma_database_size()").fetchone()
        if row:
            print(f"  DuckDB storage: database_size={row[1]}, block_size={row[2]}, "
                  f"total_blocks={row[3]}, used_blocks={row[4]}, free_blocks={row[5]}")
    except Exception:
        pass

    print("  =================================================")
