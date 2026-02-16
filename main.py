import argparse
import os
import sys
import numpy as np

import config


# ---------------------------------------------------------------------------
# Phase 1: Vocabulary
# ---------------------------------------------------------------------------

def run_phase1():
    """Word list curation (extract + clean WordNet lemmas)."""
    print("\n=== Phase 1: Vocabulary ===")
    import word_list
    words = word_list.get_word_list()
    print(f"  Word list: {len(words)} words")
    print(f"  First 5: {[w[1] for w in words[:5]]}")
    print(f"  Last 5: {[w[1] for w in words[-5:]]}")
    return words


# ---------------------------------------------------------------------------
# Phase 2: Embeddings
# ---------------------------------------------------------------------------

def run_phase2(words=None, no_resume=False):
    """Word embedding generation (nomic-embed-text-v1.5, CPU, ~30 min)."""
    print("\n=== Phase 2: Embeddings ===")
    import embedder

    if words is None:
        import word_list
        words = word_list.get_word_list()
        print(f"  Regenerated word list: {len(words)} words")

    word_texts = [w[1] for w in words]
    model = embedder.load_model()
    embeddings = embedder.embed_words(word_texts, model, resume=not no_resume)
    embedder.cleanup_intermediates()
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Phase 3: Database
# ---------------------------------------------------------------------------

def run_phase3(words=None, embeddings=None, db_path=None):
    """Schema + bulk insert + POS/components backfill + word hashes."""
    print("\n=== Phase 3: Database ===")
    import database
    import post_process
    import word_list

    if db_path is None:
        db_path = config.DB_PATH

    con = database.get_connection(db_path)
    database.create_schema(con)

    if words is None or embeddings is None:
        if words is None:
            words = word_list.get_word_list()
            print(f"  Regenerated word list: {len(words)} words")
        if embeddings is None:
            final_path = os.path.join(config.INTERMEDIATE_DIR, "embeddings_final.npy")
            if os.path.exists(final_path):
                embeddings = np.load(final_path)
                print(f"  Loaded embeddings from {final_path}: {embeddings.shape}")
            else:
                raise RuntimeError(
                    "No embeddings available. Run phase 2 first or provide embeddings_final.npy"
                )

    database.insert_words(con, words, embeddings)
    database.create_indexes(con)
    print(f"  Database ready at {db_path}")

    # Schema migration + POS/category/components backfill (was phase 4b)
    print("  Running schema migration + POS backfill...")
    database.migrate_schema(con)
    post_process.backfill_pos_and_category(con)
    post_process.populate_word_components(con)

    # Word hash computation (was phase 4c)
    print("  Computing word hashes...")
    word_list.compute_word_hashes(con)

    print("  Phase 3 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 4: Analysis
# ---------------------------------------------------------------------------

def run_phase4(con=None, embeddings=None, word_ids=None, db_path=None,
               skip_pair_overlap=False):
    """Dimension thresholds + word counts + specificity + pair overlap."""
    print("\n=== Phase 4: Analysis ===")
    import database
    import analyzer
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    # Statistical analysis (was phase 5)
    analyzer.run_analysis(con, embedding_matrix=embeddings, word_ids=word_ids)

    # Post-processing: dim counts, specificity, views (was phase 6)
    post_process.update_word_dim_counts(con)
    post_process.compute_word_specificity(con)
    post_process.create_views(con)

    if not skip_pair_overlap:
        post_process.materialize_word_pair_overlap(con)
    else:
        print("  Skipping word pair overlap materialization")

    post_process.print_summary(con)
    print("  Phase 4 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 5: Senses
# ---------------------------------------------------------------------------

def run_phase5(con=None, db_path=None, no_resume=False):
    """Sense embeddings + sense dim analysis + domain-anchored compounds."""
    print("\n=== Phase 5: Senses ===")
    import database
    import embedder
    import analyzer

    if con is None:
        con = database.get_connection(db_path)

    # Full-gloss sense embeddings (was phase 5b)
    sense_texts = embedder.build_sense_texts(con)
    if not sense_texts:
        print("  No ambiguous words to embed")
        return con

    model = embedder.load_model()
    sense_embeddings = embedder.embed_senses(sense_texts, model, resume=not no_resume)
    embedder.cleanup_intermediates(config.SENSE_INTERMEDIATE_DIR)

    # Store senses and embeddings in DB
    print("  Inserting senses into database...")
    con.execute("DELETE FROM word_senses WHERE is_domain_anchored = FALSE")
    sense_rows = []
    for i, s in enumerate(sense_texts):
        emb = sense_embeddings[i].tolist()
        sense_rows.append((
            s["sense_id"], s["word_id"], s["pos"],
            s["synset_name"], s["gloss"], emb, 0, False
        ))
    database.insert_word_senses(con, sense_rows)
    print(f"  Inserted {len(sense_rows)} full-gloss senses")

    # Sense analysis against existing thresholds (was phase 5c)
    sense_embs, sense_ids = database.load_sense_embedding_matrix(con)
    if len(sense_ids) > 0:
        analyzer.run_sense_analysis(con, sense_embs, sense_ids)

    # Domain-anchored compound embeddings (NEW)
    print("  Building domain-anchored compound embeddings...")
    domain_texts = embedder.build_domain_compound_texts(con)
    if domain_texts:
        domain_embeddings = embedder.embed_domain_compounds(
            domain_texts, model, resume=not no_resume
        )
        embedder.cleanup_intermediates(embedder.DOMAIN_COMPOUND_INTERMEDIATE_DIR)

        # Store domain-anchored senses
        print("  Inserting domain-anchored senses into database...")
        con.execute("DELETE FROM word_senses WHERE is_domain_anchored = TRUE")
        domain_rows = []
        for i, s in enumerate(domain_texts):
            emb = domain_embeddings[i].tolist()
            domain_rows.append((
                s["sense_id"], s["word_id"], s["pos"],
                s["synset_name"], s["gloss"], emb, 0, True
            ))
        database.insert_word_senses(con, domain_rows)
        print(f"  Inserted {len(domain_rows)} domain-anchored senses")

        # Analyze domain-anchored sense embeddings against thresholds
        domain_emb_rows = con.execute("""
            SELECT sense_id, sense_embedding FROM word_senses
            WHERE is_domain_anchored = TRUE AND sense_embedding IS NOT NULL
            ORDER BY sense_id
        """).fetchall()
        if domain_emb_rows:
            d_ids = [r[0] for r in domain_emb_rows]
            d_matrix = np.array([r[1] for r in domain_emb_rows], dtype=np.float32)
            analyzer.run_sense_analysis(con, d_matrix, d_ids)

    print("  Phase 5 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 6: Enrichment
# ---------------------------------------------------------------------------

def run_phase6(con=None, db_path=None):
    """POS enrichment, compound contamination, compositionality, dim abstractness/specificity, sense spread, negation, valence."""
    print("\n=== Phase 6: Enrichment ===")
    import database
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    post_process.compute_pos_enrichment(con)
    post_process.score_compound_contamination(con)
    post_process.compute_compositionality(con)
    post_process.compute_dimension_abstractness(con)
    post_process.compute_dimension_specificity(con)
    post_process.compute_sense_spread(con)
    post_process.compute_negation_vector(con)
    post_process.compute_dimension_valence(con)
    print("  Phase 6 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 7: Islands
# ---------------------------------------------------------------------------

def run_phase7(con=None, db_path=None):
    """Jaccard matrix, 3-generation Leiden detection, noise recovery."""
    print("\n=== Phase 7: Islands ===")
    import database, islands

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    islands.run_island_detection(con)
    print("  Phase 7 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 8: Refinement
# ---------------------------------------------------------------------------

def run_phase8(con=None, db_path=None):
    """Reef loyalty refinement (iterative dim reassignment)."""
    print("\n=== Phase 8: Refinement ===")
    import database, reef_refine

    if con is None:
        con = database.get_connection(db_path)

    reef_refine.run_reef_refinement(con)
    print("  Phase 8 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 9: Reef Analytics
# ---------------------------------------------------------------------------

def run_phase9(con=None, db_path=None):
    """Backfill hierarchy, domain-aware affinity, reef IDF, arch concentration, reef valence, POS composition, reef edges, composite weight."""
    print("\n=== Phase 9: Reef Analytics ===")
    import database, islands
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)

    # Backfill denormalized hierarchy columns (was 9b)
    islands.backfill_membership_islands(con)

    # Domain-aware word-reef affinity (was 9b, now with domain-anchored filter)
    islands.compute_word_reef_affinity(con)

    # Reef IDF (was 9e)
    post_process.compute_reef_idf(con)

    # Universal word analytics: arch concentration (was 9d)
    post_process.compute_arch_concentration(con)

    # Reef valence (was 9d)
    post_process.compute_reef_valence(con)

    # Sense-aware POS composition at all hierarchy levels (was 9f)
    post_process.compute_dim_pos_composition(con)
    post_process.compute_hierarchy_pos_composition(con)

    # Hierarchy specificity + reef edges + composite weight (was 9g)
    post_process.compute_hierarchy_specificity(con)
    post_process.compute_reef_edges(con)
    post_process.compute_composite_weight(con)

    # Recreate views with updated data
    post_process.create_views(con)

    print("  Phase 9 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 10: Naming
# ---------------------------------------------------------------------------

def run_phase10(con=None, db_path=None):
    """LLM-based naming (reefs -> islands -> archipelagos)."""
    print("\n=== Phase 10: Naming ===")
    import database, islands

    if con is None:
        con = database.get_connection(db_path)

    islands.generate_island_names(con)
    print("  Phase 10 complete")
    return con


# ---------------------------------------------------------------------------
# Phase 11: Finalization
# ---------------------------------------------------------------------------

def run_phase11(con=None, db_path=None):
    """Morphy variant expansion + DB maintenance (integrity, indexes, optimization)."""
    print("\n=== Phase 11: Finalization ===")
    import database
    import word_list

    if db_path is None:
        db_path = config.DB_PATH

    if con is None:
        con = database.get_connection(db_path)

    # Morphy variant expansion (was phase 11)
    database.migrate_schema(con)
    word_list.expand_morphy_variants(con)

    # DB maintenance (was phase 7)
    database.run_integrity_checks(con)
    database.rebuild_indexes(con)
    database.run_storage_optimization(con)
    database.print_database_report(con, db_path)

    print("  Phase 11 complete")
    return con


# ---------------------------------------------------------------------------
# Explorer (standalone command, not a pipeline phase)
# ---------------------------------------------------------------------------

def run_explore(con=None, db_path=None):
    print("\n=== Interactive Explorer ===")
    import database
    import explore

    if con is None:
        con = database.get_connection(db_path, read_only=True)

    print("Commands:")
    print("  what_is <word>              — dimensions a word belongs to")
    print("  words_like <word> [n]       — similar words")
    print("  dim <id> [n]                — dimension members")
    print("  compare <word1> <word2>     — compare two words")
    print("  disambiguate <word>         — find distinct senses")
    print("  bridges <word1> <word2> [n] — bridge words")
    print("  dim_info <id>               — full dimension stats")
    print("  search <pattern>            — search words (SQL LIKE)")
    print("  senses <word>               — show WordNet senses with dims")
    print("  compositionality <compound> — compositionality analysis")
    print("  contamination <word>        — compound contamination report")
    print("  pos_dims <pos>              — dimensions enriched for POS")
    print("  archipelago <word>          — island hierarchy position")
    print("  relationship <w1> <w2>      — classify word relationship")
    print("  exclusion <w1> <w2>         — shared reef exclusions (universal words)")
    print("  bridge_profile <word>       — reef distribution & cross-arch bridges")
    print("  affinity <word>             — reef affinity profile (all reefs by weighted z)")
    print("  synonyms <word> [n]        — synonym candidates via Jaccard overlap")
    print("  antonyms <word> [n]        — antonym prediction via negation vector")
    print("  evaluate                    — run standard semantic evaluation battery")
    print("  quit                        — exit")

    while True:
        try:
            line = input("\nexplore> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "quit" or cmd == "exit":
                break
            elif cmd == "what_is" and len(parts) >= 2:
                explore.what_is(con, " ".join(parts[1:]))
            elif cmd == "words_like" and len(parts) >= 2:
                n = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 20
                word = " ".join(parts[1:-1]) if len(parts) > 2 and parts[-1].isdigit() else " ".join(parts[1:])
                explore.words_like(con, word, top_n=n)
            elif cmd == "dim" and len(parts) >= 2:
                dim_id = int(parts[1])
                n = int(parts[2]) if len(parts) > 2 else 50
                explore.dimension_members(con, dim_id, top_n=n)
            elif cmd == "compare" and len(parts) >= 3:
                explore.compare_words(con, parts[1], parts[2])
            elif cmd == "disambiguate" and len(parts) >= 2:
                explore.disambiguate(con, " ".join(parts[1:]))
            elif cmd == "bridges" and len(parts) >= 3:
                n = int(parts[-1]) if len(parts) > 3 and parts[-1].isdigit() else 10
                explore.find_bridges(con, parts[1], parts[2], top_n=n)
            elif cmd == "dim_info" and len(parts) >= 2:
                explore.dim_info(con, int(parts[1]))
            elif cmd == "search" and len(parts) >= 2:
                explore.search_words(con, " ".join(parts[1:]))
            elif cmd == "senses" and len(parts) >= 2:
                explore.senses(con, " ".join(parts[1:]))
            elif cmd == "compositionality" and len(parts) >= 2:
                explore.compositionality(con, " ".join(parts[1:]))
            elif cmd == "contamination" and len(parts) >= 2:
                explore.contamination(con, " ".join(parts[1:]))
            elif cmd == "pos_dims" and len(parts) >= 2:
                explore.pos_dims(con, parts[1])
            elif cmd == "archipelago" and len(parts) >= 2:
                explore.archipelago(con, " ".join(parts[1:]))
            elif cmd == "relationship" and len(parts) >= 3:
                explore.relationship(con, parts[1], parts[2])
            elif cmd == "exclusion" and len(parts) >= 3:
                explore.exclusion(con, parts[1], parts[2])
            elif cmd == "bridge_profile" and len(parts) >= 2:
                explore.bridge_profile(con, " ".join(parts[1:]))
            elif cmd == "affinity" and len(parts) >= 2:
                explore.affinity(con, " ".join(parts[1:]))
            elif cmd == "synonyms" and len(parts) >= 2:
                n = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 20
                word = " ".join(parts[1:-1]) if len(parts) > 2 and parts[-1].isdigit() else " ".join(parts[1:])
                explore.synonyms(con, word, top_n=n)
            elif cmd == "antonyms" and len(parts) >= 2:
                n = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 10
                word = " ".join(parts[1:-1]) if len(parts) > 2 and parts[-1].isdigit() else " ".join(parts[1:])
                explore.antonyms(con, word, top_n=n)
            elif cmd == "evaluate":
                explore.evaluate(con)
            else:
                print(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

PHASE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

PHASE_NAMES = {
    "1": "Vocabulary",
    "2": "Embeddings",
    "3": "Database",
    "4": "Analysis",
    "5": "Senses",
    "6": "Enrichment",
    "7": "Islands",
    "8": "Refinement",
    "9": "Reef Analytics",
    "10": "Naming",
    "11": "Finalization",
}


def main():
    parser = argparse.ArgumentParser(description="Vector Space Distillation Engine")
    parser.add_argument("--phase", type=str,
                        help="Run only this phase (1-11 or explore)")
    parser.add_argument("--from", dest="from_phase", type=str,
                        help="Run from this phase onward")
    parser.add_argument("--skip-pair-overlap", action="store_true",
                        help="Skip expensive pair overlap materialization")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from intermediate .npy files")
    parser.add_argument("--db", default=config.DB_PATH,
                        help=f"Database path (default: {config.DB_PATH})")
    args = parser.parse_args()

    if args.phase:
        if args.phase == "explore":
            run_explore(db_path=args.db)
            return
        phases = [args.phase]
    elif args.from_phase:
        start_idx = PHASE_ORDER.index(args.from_phase)
        phases = PHASE_ORDER[start_idx:]
    else:
        phases = PHASE_ORDER[:]

    words = None
    embeddings = None
    con = None

    for phase in phases:
        if phase == "1":
            words = run_phase1()
        elif phase == "2":
            embeddings = run_phase2(words=words, no_resume=args.no_resume)
        elif phase == "3":
            con = run_phase3(words=words, embeddings=embeddings, db_path=args.db)
        elif phase == "4":
            word_ids = [w[0] for w in words] if words else None
            con = run_phase4(
                con=con, embeddings=embeddings, word_ids=word_ids,
                db_path=args.db, skip_pair_overlap=args.skip_pair_overlap,
            )
        elif phase == "5":
            con = run_phase5(con=con, db_path=args.db, no_resume=args.no_resume)
        elif phase == "6":
            con = run_phase6(con=con, db_path=args.db)
        elif phase == "7":
            con = run_phase7(con=con, db_path=args.db)
        elif phase == "8":
            con = run_phase8(con=con, db_path=args.db)
        elif phase == "9":
            con = run_phase9(con=con, db_path=args.db)
        elif phase == "10":
            con = run_phase10(con=con, db_path=args.db)
        elif phase == "11":
            con = run_phase11(con=con, db_path=args.db)

    if con is not None:
        con.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
