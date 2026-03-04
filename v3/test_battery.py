"""Test battery for export weight quality.

Runs a suite of multi-word queries against the export tables and checks
whether the correct island wins by accumulated export_weight.  Used for
rapid iteration on the weight formula in populate_exports.py.

Usage:
    python v3/test_battery.py                # run all tests
    python v3/test_battery.py -v             # verbose: show top-5 per query
    python v3/test_battery.py -k astronomy   # run only queries matching name
    python v3/test_battery.py --diagnostics  # run per-word diagnostics on failures
"""

import os
import sqlite3
import sys
from dataclasses import dataclass

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")

# ── Test definitions ──────────────────────────────────────────────────

@dataclass
class Query:
    name: str
    expected_island: str
    words: list[str]
    source: str = ""  # where the test came from

QUERIES = [
    # ── Shoal xfail tests (all 5 should PASS) ────────────────────────
    Query("xfail_crane_bird",
          "Biology",
          ["crane", "bird", "migration", "wetland"],
          source="shoal xfail: crane bird activates aircraft"),
    Query("xfail_crane_machine",
          "Games",
          ["crane", "machine", "claw", "arcade", "prize"],
          source="shoal xfail: claw arcade activates esports"),
    Query("xfail_apple_pie",
          "Gastronomy",
          ["apple", "pie", "recipe"],
          source="shoal xfail: apple pie activates french"),
    Query("xfail_un_security",
          "Politics",
          ["united", "nation", "security", "council"],
          source="shoal xfail: all words unknown to scorer"),
    Query("xfail_lunar_crater",
          "Astronomy",
          ["lunar", "crater", "moon", "surface"],
          source="shoal xfail: drowned by Mercury/Jupiter"),

    # ── Core domain convergence (5 specific words per domain) ─────────
    Query("military",
          "Military",
          ["soldier", "infantry", "regiment", "battalion", "trench"]),
    Query("visual_arts",
          "Art",
          ["fresco", "mural", "watercolor", "pigment", "sketch"]),
    Query("cooking",
          "Gastronomy",
          ["broth", "stew", "seasoning", "flour", "bake"]),
    Query("comp_science",
          "Computer Science",
          ["compiler", "algorithm", "database", "recursion", "syntax"]),
    Query("chess",
          "Games",
          ["checkmate", "pawn", "knight", "rook", "bishop"]),
    Query("astronomy_core",
          "Astronomy",
          ["telescope", "star", "comet", "galaxy", "nebula"]),
    Query("medicine",
          "Medicine",
          ["diagnosis", "symptom", "therapy", "patient", "prescription"]),
    Query("economics",
          "Economics",
          ["inflation", "recession", "monetary", "fiscal", "unemployment"]),
    Query("music_performance",
          "Music",
          ["violin", "piano", "melody", "symphony", "orchestra"]),
    Query("law_trial",
          "Law",
          ["verdict", "defendant", "plaintiff", "jury", "trial"]),
    Query("chemistry",
          "Chemistry",
          ["molecule", "atom", "reaction", "catalyst", "compound"]),

    # ── Broader domain coverage ───────────────────────────────────────
    Query("football",
          "Sport",
          ["tackle", "touchdown", "quarterback", "punt", "scrimmage"]),
    Query("religion",
          "Religion",
          ["prayer", "worship", "temple", "sermon", "congregation"]),
    Query("geology",
          "Earth Science",
          ["erosion", "sediment", "tectonic", "mineral", "fault"]),
    Query("psychology",
          "Psychology",
          ["cognition", "emotion", "behavior", "perception", "motivation"]),
    Query("architecture",
          "Architecture",
          ["dome", "arch", "column", "facade", "vault"]),
    Query("oceanography",
          "Earth Science",
          ["ocean", "tide", "current", "coral", "submarine"]),
    Query("genetics",
          "Biology",
          ["gene", "chromosome", "mutation", "allele", "genome"]),
    Query("photography",
          "Art",
          ["camera", "lens", "exposure", "aperture", "shutter"]),
    Query("agriculture",
          "Agriculture",
          ["harvest", "crop", "irrigation", "soil", "fertilizer"]),
    Query("finance",
          "Finance",
          ["stock", "bond", "dividend", "portfolio", "equity"]),
    Query("transportation",
          "Transportation",
          ["railroad", "locomotive", "freight", "depot", "track"]),
    Query("linguistics",
          "Linguistics",
          ["phoneme", "morpheme", "syntax", "vowel", "consonant"]),
    Query("philosophy",
          "Philosophy",
          ["ontology", "epistemology", "metaphysics", "existentialism", "phenomenology"]),
    Query("sociology",
          "Sociology",
          ["norm", "institution", "stratification", "migration", "inequality"]),

    # ── Known dangerous patterns (from Shoal data_prob reports) ───────
    Query("military_vs_sports",
          "Military",
          ["attack", "defense", "offensive", "retreat", "charge", "advance"],
          source="shoal: military words went to sports reefs"),
    Query("math_vs_devops",
          "Mathematics",
          ["derivative", "logarithm", "integral", "function", "variable"],
          source="shoal: math words absorbed by ML/devops"),
    Query("ornithology",
          "Biology",
          ["avian", "falcon", "sparrow", "parrot", "eagle"],
          source="shoal: 100% ornithology failure in v2"),
    Query("visual_arts_broad",
          "Art",
          ["painting", "sculpture", "canvas", "gallery", "museum"],
          source="shoal: 95% visual arts failure in v2"),
    Query("cooking_basics",
          "Gastronomy",
          ["cuisine", "chef", "kitchen", "recipe", "stew"],
          source="shoal: cooking words went to bacteriology"),

    # ── Bottom-9 stress tests (low-health islands, tricky compound-ish) ──
    #    Written blind — no DB lookups beforehand.
    Query("ideology_propaganda",
          "Ideology",
          ["propaganda", "manifesto", "doctrine", "indoctrination", "dogma"],
          source="bottom-9: Ideology, health=62, worst exclusivity"),
    Query("engineering_robotics",
          "Engineering",
          ["actuator", "servo", "torque", "hydraulic", "pneumatic"],
          source="bottom-9: Engineering, health=85"),
    Query("compsci_machine_learning",
          "Computer Science",
          ["neural", "gradient", "tensor", "epoch", "backpropagation"],
          source="bottom-9: CompSci, health=87, cross-domain w/ math"),
    Query("performing_arts_theater",
          "Performing Arts",
          ["rehearsal", "audition", "curtain", "monologue", "playwright"],
          source="bottom-9: Performing Arts, health=87"),
    Query("hobbies_knitting",
          "Hobbies & Crafts",
          ["yarn", "needle", "stitch", "crochet", "bobbin"],
          source="bottom-9: Hobbies & Crafts, health=96"),
    Query("philosophy_ethics",
          "Philosophy",
          ["utilitarian", "deontology", "virtue", "categorical", "imperative"],
          source="bottom-9: Philosophy, health=99, only 138 excl compounds"),
    Query("media_journalism",
          "Media & Communications",
          ["journalist", "headline", "editorial", "newsroom", "deadline"],
          source="bottom-9: Media & Comms, health=103"),
    Query("astronomy_planets",
          "Astronomy",
          ["orbit", "planet", "asteroid", "eclipse", "equinox"],
          source="bottom-9: Astronomy, health=103, tiny island"),
    Query("games_tabletop",
          "Games",
          ["dice", "token", "gambit", "bluff", "wager"],
          source="bottom-9: Games, health=108"),

    # ── Top-9 stress tests (high-health islands, equally difficult) ───────
    #    Written blind — no DB lookups beforehand.
    Query("biology_ecology",
          "Biology",
          ["predator", "prey", "habitat", "symbiosis", "ecosystem"],
          source="top-9: Biology, health=260, highest exclusivity"),
    Query("chemistry_organic",
          "Chemistry",
          ["benzene", "polymer", "solvent", "reagent", "titration"],
          source="top-9: Chemistry, health=218"),
    Query("medicine_surgery",
          "Medicine",
          ["scalpel", "incision", "suture", "anesthesia", "biopsy"],
          source="top-9: Medicine, health=209"),
    Query("gastronomy_baking",
          "Gastronomy",
          ["dough", "yeast", "knead", "oven", "crust"],
          source="top-9: Gastronomy, health=197"),
    Query("psychology_disorders",
          "Psychology",
          ["anxiety", "phobia", "trauma", "delusion", "obsession"],
          source="top-9: Psychology, health=176, cross-domain bleed"),
    Query("mythology_greek",
          "Mythology",
          ["olympus", "zeus", "titan", "oracle", "minotaur"],
          source="top-9: Mythology, health=168"),
    Query("architecture_gothic",
          "Architecture",
          ["buttress", "nave", "spire", "gargoyle", "cloister"],
          source="top-9: Architecture, health=161"),
    Query("art_renaissance",
          "Art",
          ["palette", "easel", "portrait", "mosaic", "engraving"],
          source="top-9: Art, health=160"),
    Query("physics_quantum",
          "Physics",
          ["photon", "quark", "boson", "fermion", "lepton"],
          source="top-9: Physics, health=160"),

    # ── Disambiguation stress tests (IQF/TQF evaluation) ────────────
    #    Each query has an ambiguous anchor word that belongs to multiple
    #    islands; surrounding context should resolve to the correct one.
    #    Written blind — no DB lookups.
    Query("ambiguous_cell",
          "Biology",
          ["cell", "membrane", "cytoplasm", "nucleus", "organelle"],
          source="IQF: cell ambiguous (CS/prison/bio), context is biology"),
    Query("ambiguous_culture",
          "Biology",
          ["culture", "bacteria", "incubation", "petri", "agar"],
          source="IQF: culture ambiguous (sociology/art/bio), context is microbiology"),
    Query("ambiguous_depression",
          "Economics",
          ["depression", "recession", "unemployment", "downturn", "market"],
          source="IQF: depression ambiguous (psych/econ/geo), context is economics"),
    Query("ambiguous_movement",
          "Music",
          ["movement", "allegro", "tempo", "adagio", "sonata"],
          source="IQF: movement ambiguous (politics/music/dance), context is musical"),
    Query("ambiguous_revolution",
          "Astronomy",
          ["revolution", "orbit", "perihelion", "aphelion", "celestial"],
          source="IQF: revolution ambiguous (politics/astro), context is orbital"),
    Query("ambiguous_field",
          "Agriculture",
          ["field", "crop", "harvest", "plow", "sow"],
          source="IQF: field ambiguous (physics/math/ag), context is farming"),
    Query("ambiguous_power",
          "Physics",
          ["power", "watt", "voltage", "ampere", "resistance"],
          source="IQF: power ambiguous (politics/physics), context is electrical"),
    Query("ambiguous_network",
          "Computer Science",
          ["network", "router", "protocol", "bandwidth", "server"],
          source="IQF: network ambiguous (CS/sociology/media), context is networking"),
    Query("noisy_should_win_sociology",
          "Sociology",
          ["stratification", "proletariat", "bourgeoisie", "caste", "inequality"],
          source="IQF-reverse: noisy island should still win with strong context"),
    Query("noisy_should_win_psychology",
          "Psychology",
          ["schizophrenia", "hallucination", "paranoia", "psychosis", "delusion"],
          source="IQF-reverse: noisy island should win with specific clinical terms"),
]


# ── Scoring ───────────────────────────────────────────────────────────

def score_query(con, query):
    """Score a query against export tables. Returns list of (island, total_weight, n_words, word_list).

    Uses MAX(export_weight) per (word, island) to avoid double-counting when
    a word appears in multiple towns/reefs within the same island.
    """
    placeholders = ",".join("?" for _ in query.words)
    rows = con.execute(f"""
        SELECT sub.island_id, i.name AS island,
            SUM(sub.max_weight) AS total_weight,
            COUNT(*) AS n_words,
            GROUP_CONCAT(sub.word) AS words
        FROM (
            SELECT ei.island_id, w.word_id, w.word,
                   MAX(ei.export_weight) AS max_weight
            FROM ExportIndex ei
            JOIN Words w ON ei.word_id = w.word_id
            WHERE w.word IN ({placeholders})
            GROUP BY ei.island_id, w.word_id
        ) sub
        JOIN Islands i ON sub.island_id = i.island_id
        GROUP BY sub.island_id
        ORDER BY total_weight DESC
    """, query.words).fetchall()
    return rows


def diagnose_query(con, query):
    """Per-word breakdown for a failing query."""
    placeholders = ",".join("?" for _ in query.words)

    print(f"\n    Per-word breakdown for '{query.name}':")
    for word in query.words:
        rows = con.execute("""
            SELECT ei.export_level, i.name AS island, t.name AS town, r.name AS reef,
                   ei.export_weight
            FROM ExportIndex ei
            JOIN Islands i ON ei.island_id = i.island_id
            LEFT JOIN Towns t ON ei.town_id = t.town_id
            LEFT JOIN Reefs r ON ei.reef_id = r.reef_id
            JOIN Words w ON ei.word_id = w.word_id
            WHERE w.word = ?
            ORDER BY ei.export_weight DESC
            LIMIT 5
        """, (word,)).fetchall()
        if not rows:
            # Check if word exists at all
            exists = con.execute(
                "SELECT word_id, reef_count FROM Words WHERE word = ?", (word,)
            ).fetchone()
            if exists:
                print(f"      {word:20s} IN VOCAB (reef_count={exists[1]}) but NOT EXPORTED")
            else:
                print(f"      {word:20s} NOT IN VOCABULARY")
            continue

        top = rows[0]
        expected_hit = any(r[1] == query.expected_island for r in rows)
        marker = "" if expected_hit else " ← MISSING expected island!"
        print(f"      {word:20s} top: {top[0]:6s} {top[1]:25s} w={top[4]:3d}"
              f"  ({len(rows)} entries){marker}")

    # Show weight component breakdown for island-level words in expected vs actual
    actual_rows = score_query(con, query)
    if actual_rows:
        actual_island = actual_rows[0][1]
        if actual_island != query.expected_island:
            print(f"\n    Weight components (expected={query.expected_island} vs actual={actual_island}):")
            for word in query.words:
                row = con.execute("""
                    SELECT ie.island_id, i.name,
                        ROUND(ie.idf, 2) AS island_idf,
                        ROUND(ie.centroid_sim, 3) AS csim,
                        ROUND(ie.name_cos, 3) AS ncos,
                        ROUND(ie.effective_sim, 3) AS esim,
                        ROUND(ie.source_quality, 2) AS sq,
                        ie.export_weight
                    FROM IslandWordExports ie
                    JOIN Islands i USING(island_id)
                    JOIN Words w USING(word_id)
                    WHERE w.word = ? AND i.name IN (?, ?)
                    ORDER BY i.name
                """, (word, query.expected_island, actual_island)).fetchall()
                if row:
                    for r in row:
                        print(f"      {word:15s} {r[1]:25s}  idf={r[2]}  csim={r[3]}  "
                              f"ncos={r[4]}  esim={r[5]}  sq={r[6]}  w={r[7]}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export weight test battery")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show top-5 islands per query")
    parser.add_argument("-k", "--filter", type=str, default="",
                        help="Only run queries whose name contains this string")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Show per-word diagnostics for failures")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Flag passes with margin below this ratio (e.g. 1.2)")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Verify export tables are populated
    n_exports = con.execute("""
        SELECT (SELECT COUNT(*) FROM ReefWordExports) +
               (SELECT COUNT(*) FROM TownWordExports) +
               (SELECT COUNT(*) FROM IslandWordExports)
    """).fetchone()[0]
    if n_exports == 0:
        print("ERROR: Export tables are empty. Run populate_exports.py first.")
        sys.exit(1)

    queries = QUERIES
    if args.filter:
        queries = [q for q in queries if args.filter.lower() in q.name.lower()]
        if not queries:
            print(f"No queries match filter '{args.filter}'")
            sys.exit(1)

    # Run tests
    n_pass = 0
    n_fail = 0
    n_thin = 0
    failures = []
    thin_margins = []

    for q in queries:
        results = score_query(con, q)

        if not results:
            status = "FAIL"
            winner = "(no results)"
            margin = 0.0
            n_fail += 1
            failures.append(q)
        else:
            winner = results[0][1]
            w1 = results[0][2]
            nw = results[0][3]
            w2 = results[1][2] if len(results) > 1 else 0
            runner_up = results[1][1] if len(results) > 1 else "-"
            margin = w1 / w2 if w2 > 0 else 99.0

            if winner == q.expected_island:
                status = "PASS"
                n_pass += 1
                if args.min_margin and margin < args.min_margin:
                    n_thin += 1
                    thin_margins.append((q, margin, runner_up))
            else:
                status = "FAIL"
                n_fail += 1
                failures.append(q)

        # Output
        margin_str = f"{margin:.2f}x" if margin < 99 else "∞"
        flag = ""
        if status == "FAIL":
            flag = f" ← expected {q.expected_island}"
        elif args.min_margin and margin < args.min_margin:
            flag = f" ← thin margin ({margin_str})"

        print(f"  {'PASS' if status == 'PASS' else 'FAIL':4s}  {q.name:25s}  "
              f"→ {winner:25s}  margin={margin_str:>6s}  "
              f"words={nw if results else 0}/{len(q.words)}{flag}")

        if args.verbose and results:
            for r in results[:5]:
                marker = " ◄" if r[1] == q.expected_island else ""
                print(f"        {r[1]:25s}  weight={r[2]:>5}  "
                      f"words={r[3]}/{len(q.words)}  [{r[4]}]{marker}")

        if args.diagnostics and status == "FAIL":
            diagnose_query(con, q)

    # Summary
    total = n_pass + n_fail
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {n_pass}/{total} pass ({100*n_pass/total:.0f}%)")

    if failures:
        print(f"\nFAILURES ({n_fail}):")
        for q in failures:
            results = score_query(con, q)
            actual = results[0][1] if results else "(none)"
            print(f"  {q.name:25s}  expected={q.expected_island:25s}  "
                  f"actual={actual}")
            if q.source:
                print(f"  {'':25s}  source: {q.source}")

    if thin_margins:
        print(f"\nTHIN MARGINS ({n_thin}, below {args.min_margin}x):")
        for q, margin, runner_up in thin_margins:
            print(f"  {q.name:25s}  {q.expected_island:25s}  "
                  f"margin={margin:.2f}x  runner_up={runner_up}")

    # Per-word coverage check
    all_words = set()
    for q in queries:
        all_words.update(q.words)
    missing = []
    for word in sorted(all_words):
        row = con.execute("""
            SELECT COUNT(*) FROM ExportIndex ei
            JOIN Words w ON ei.word_id = w.word_id WHERE w.word = ?
        """, (word,)).fetchone()
        if row[0] == 0:
            in_vocab = con.execute(
                "SELECT reef_count FROM Words WHERE word = ?", (word,)
            ).fetchone()
            if in_vocab:
                missing.append(f"{word} (in vocab, reef_count={in_vocab[0]})")
            else:
                missing.append(f"{word} (NOT in vocab)")

    if missing:
        print(f"\nWORDS NOT IN EXPORTS ({len(missing)}):")
        for m in missing:
            print(f"  {m}")

    con.close()


if __name__ == "__main__":
    main()
