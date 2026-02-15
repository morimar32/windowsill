import argparse
import os
import sys
import numpy as np

import config


def run_phase2():
    print("\n=== Phase 2: Word List Curation ===")
    import word_list
    words = word_list.get_word_list()
    print(f"  Word list: {len(words)} words")
    print(f"  First 5: {[w[1] for w in words[:5]]}")
    print(f"  Last 5: {[w[1] for w in words[-5:]]}")
    return words


def run_phase3(words=None, no_resume=False):
    print("\n=== Phase 3: Embedding Generation ===")
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


def run_phase4(words=None, embeddings=None, db_path=None):
    print("\n=== Phase 4: Database Schema & Bulk Insert ===")
    import database

    if db_path is None:
        db_path = config.DB_PATH

    con = database.get_connection(db_path)
    database.create_schema(con)

    if words is None or embeddings is None:
        if words is None:
            import word_list
            words = word_list.get_word_list()
            print(f"  Regenerated word list: {len(words)} words")
        if embeddings is None:
            final_path = os.path.join(config.INTERMEDIATE_DIR, "embeddings_final.npy")
            if os.path.exists(final_path):
                embeddings = np.load(final_path)
                print(f"  Loaded embeddings from {final_path}: {embeddings.shape}")
            else:
                raise RuntimeError(
                    "No embeddings available. Run phase 3 first or provide embeddings_final.npy"
                )

    database.insert_words(con, words, embeddings)
    database.create_indexes(con)
    print(f"  Database ready at {db_path}")
    return con


def run_phase5(con=None, embeddings=None, word_ids=None, db_path=None):
    print("\n=== Phase 5: Statistical Analysis ===")
    import database
    import analyzer

    if con is None:
        con = database.get_connection(db_path)

    analyzer.run_analysis(con, embedding_matrix=embeddings, word_ids=word_ids)
    return con


def run_phase6(con=None, db_path=None, skip_pair_overlap=False):
    print("\n=== Phase 6: Post-Processing ===")
    import database
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    post_process.update_word_dim_counts(con)
    post_process.compute_word_specificity(con)
    post_process.create_views(con)

    if not skip_pair_overlap:
        post_process.materialize_word_pair_overlap(con)
    else:
        print("  Skipping word pair overlap materialization")

    post_process.print_summary(con)
    return con


def run_phase4b(con=None, db_path=None):
    print("\n=== Phase 4b: Schema Migration + POS/Category/Components Backfill ===")
    import database
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    post_process.backfill_pos_and_category(con)
    post_process.populate_word_components(con)
    print("  Phase 4b complete")
    return con


def run_phase5b(con=None, db_path=None, no_resume=False):
    print("\n=== Phase 5b: Sense-Specific Embedding Generation ===")
    import database
    import embedder

    if con is None:
        con = database.get_connection(db_path)

    sense_texts = embedder.build_sense_texts(con)
    if not sense_texts:
        print("  No ambiguous words to embed")
        return con

    model = embedder.load_model()
    sense_embeddings = embedder.embed_senses(sense_texts, model, resume=not no_resume)
    embedder.cleanup_intermediates(config.SENSE_INTERMEDIATE_DIR)

    # Store senses and embeddings in DB
    print("  Inserting senses into database...")
    con.execute("DELETE FROM word_senses")
    sense_rows = []
    for i, s in enumerate(sense_texts):
        emb = sense_embeddings[i].tolist()
        sense_rows.append((
            s["sense_id"], s["word_id"], s["pos"],
            s["synset_name"], s["gloss"], emb, 0
        ))
    database.insert_word_senses(con, sense_rows)
    print(f"  Inserted {len(sense_rows)} senses")
    return con


def run_phase5c(con=None, db_path=None):
    print("\n=== Phase 5c: Sense Analysis Against Existing Thresholds ===")
    import database
    import analyzer

    if con is None:
        con = database.get_connection(db_path)

    sense_embeddings, sense_ids = database.load_sense_embedding_matrix(con)
    if len(sense_ids) == 0:
        print("  No sense embeddings found. Run phase 5b first.")
        return con

    analyzer.run_sense_analysis(con, sense_embeddings, sense_ids)
    print("  Phase 5c complete")
    return con


def run_phase6b(con=None, db_path=None):
    print("\n=== Phase 6b: POS Enrichment + Compound Contamination + Compositionality ===")
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
    print("  Phase 6b complete")
    return con


def run_phase9(con=None, db_path=None):
    print("\n=== Phase 9: Island Detection ===")
    import database, islands

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    islands.run_island_detection(con)
    return con


def run_phase9b(con=None, db_path=None):
    print("\n=== Phase 9b: Backfill + Affinity ===")
    import database, islands

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    islands.backfill_membership_islands(con)
    islands.compute_word_reef_affinity(con)
    return con


def run_phase9f(con=None, db_path=None):
    print("\n=== Phase 9f: POS Composition ===")
    import database
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    post_process.compute_dim_pos_composition(con)
    post_process.compute_hierarchy_pos_composition(con)
    print("  Phase 9f complete")
    return con


def run_phase9g(con=None, db_path=None):
    print("\n=== Phase 9g: Reef Edges ===")
    import database
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    post_process.compute_hierarchy_specificity(con)
    post_process.compute_reef_edges(con)
    print("  Phase 9g complete")
    return con


def run_phase9c(con=None, db_path=None):
    print("\n=== Phase 9c: Island & Reef Naming ===")
    import database, islands

    if con is None:
        con = database.get_connection(db_path)

    islands.generate_island_names(con)
    return con


def run_phase9d(con=None, db_path=None):
    print("\n=== Phase 9d: Universal Word Analytics (post-island) ===")
    import database, post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    post_process.compute_arch_concentration(con)
    post_process.compute_reef_valence(con)
    post_process.create_views(con)
    print("  Phase 9d complete")
    return con


def run_phase4c(con=None, db_path=None):
    print("\n=== Phase 4c: Word Hash Computation ===")
    import database
    import word_list

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    word_list.compute_word_hashes(con)
    print("  Phase 4c complete")
    return con


def run_phase9e(con=None, db_path=None):
    print("\n=== Phase 9e: Reef IDF Computation ===")
    import database
    import post_process

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    post_process.compute_reef_idf(con)
    print("  Phase 9e complete")
    return con


def run_phase11(con=None, db_path=None):
    print("\n=== Phase 11: Morphy Variant Expansion ===")
    import database
    import word_list

    if con is None:
        con = database.get_connection(db_path)

    database.migrate_schema(con)
    word_list.expand_morphy_variants(con)
    print("  Phase 11 complete")
    return con


def run_phase7(con=None, db_path=None):
    print("\n=== Phase 7: Database Maintenance ===")
    import database

    if db_path is None:
        db_path = config.DB_PATH

    if con is None:
        con = database.get_connection(db_path)

    database.run_integrity_checks(con)
    database.rebuild_indexes(con)
    database.run_storage_optimization(con)
    database.print_database_report(con, db_path)
    return con


def run_phase10(con=None, db_path=None):
    print("\n=== Phase 10: Reef Refinement ===")
    import database, reef_refine

    if con is None:
        con = database.get_connection(db_path)

    reef_refine.run_reef_refinement(con)
    return con


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


PHASE_ORDER = ["2", "3", "4", "4b", "4c", "5", "5b", "5c", "6", "6b", "9", "9b", "9d", "9e", "10", "9f", "9g", "9c", "11", "7", "explore"]


def main():
    parser = argparse.ArgumentParser(description="Vector Space Distillation Engine")
    parser.add_argument("--phase", type=str, help="Run only this phase (2-9, 4b, 4c, 5b, 5c, 6b, 9b, 9c, 9d, 9e, 9f, 9g, 10, 11, 7, explore)")
    parser.add_argument("--from", dest="from_phase", type=str, help="Run from this phase onward")
    parser.add_argument("--skip-pair-overlap", action="store_true",
                        help="Skip expensive pair overlap materialization")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from intermediate .npy files")
    parser.add_argument("--db", default=config.DB_PATH, help=f"Database path (default: {config.DB_PATH})")
    args = parser.parse_args()

    if args.phase:
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
        if phase == "2":
            words = run_phase2()
        elif phase == "3":
            embeddings = run_phase3(words=words, no_resume=args.no_resume)
        elif phase == "4":
            con = run_phase4(words=words, embeddings=embeddings, db_path=args.db)
        elif phase == "4b":
            con = run_phase4b(con=con, db_path=args.db)
        elif phase == "4c":
            con = run_phase4c(con=con, db_path=args.db)
        elif phase == "5":
            word_ids = [w[0] for w in words] if words else None
            con = run_phase5(con=con, embeddings=embeddings, word_ids=word_ids, db_path=args.db)
        elif phase == "5b":
            con = run_phase5b(con=con, db_path=args.db, no_resume=args.no_resume)
        elif phase == "5c":
            con = run_phase5c(con=con, db_path=args.db)
        elif phase == "6":
            con = run_phase6(
                con=con, db_path=args.db,
                skip_pair_overlap=args.skip_pair_overlap,
            )
        elif phase == "6b":
            con = run_phase6b(con=con, db_path=args.db)
        elif phase == "9":
            con = run_phase9(con=con, db_path=args.db)
        elif phase == "9b":
            con = run_phase9b(con=con, db_path=args.db)
        elif phase == "9c":
            con = run_phase9c(con=con, db_path=args.db)
        elif phase == "9d":
            con = run_phase9d(con=con, db_path=args.db)
        elif phase == "9e":
            con = run_phase9e(con=con, db_path=args.db)
        elif phase == "9f":
            con = run_phase9f(con=con, db_path=args.db)
        elif phase == "9g":
            con = run_phase9g(con=con, db_path=args.db)
        elif phase == "10":
            con = run_phase10(con=con, db_path=args.db)
        elif phase == "11":
            con = run_phase11(con=con, db_path=args.db)
        elif phase == "7":
            con = run_phase7(con=con, db_path=args.db)
        elif phase == "explore":
            run_explore(con=con, db_path=args.db)

    if con is not None:
        con.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
