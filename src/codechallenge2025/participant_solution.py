# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
from typing import List, Dict, Any


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.

    Args:
        query_profile: dict with 'PersonID' and locus columns (e.g. {'PersonID': 'Q001', 'TH01': '9,9.3', ...})
        database_df: Full database as pandas DataFrame (500k rows)

    Returns:
        List of up to 10 candidate dicts, sorted by strength (best first):
        [
            {
                "person_id": "P000123",
                "clr": 1e15,                    # Combined Likelihood Ratio
                "posterior": 0.99999,           # Optional: posterior probability
                "consistent_loci": 20,
                "mutated_loci": 1,
                "inconclusive_loci": 0
            },
            ...
        ]
    """
    # ------------------------------
    # Helper: parse allele strings into a set of floats
    # ------------------------------
    def parse_alleles(val):
        if pd.isna(val):
            return None
        if val in ("-", "", None):
            return None
        if "," in str(val):
            return {float(x) for x in str(val).split(",") if x}
        try:
            return {float(val)}
        except ValueError:
            return None

    # Identify loci columns
    loci = [col for col in database_df.columns if col != "PersonID"]

    # Pre-parse query alleles for quick lookup
    query_alleles = {locus: parse_alleles(query_profile.get(locus, "-")) for locus in loci}
    query_id = query_profile.get("PersonID")

    candidates: List[Dict[str, Any]] = []

    # Iterate through database profiles
    for row in database_df.itertuples(index=False):
        candidate_id = getattr(row, "PersonID")
        if candidate_id == query_id:
            continue  # skip self-comparison

        consistent_loci = 0
        mutated_loci = 0
        inconclusive_loci = 0
        score = 0.0

        for locus in loci:
            q_alleles = query_alleles[locus]
            c_val = getattr(row, locus)
            c_alleles = parse_alleles(c_val)

            if q_alleles is None or c_alleles is None:
                inconclusive_loci += 1
                continue

            shared = q_alleles.intersection(c_alleles)
            if shared:
                consistent_loci += 1
                score += 2.0 + 0.1 * len(shared)
                continue

            # Allow ±1 repeat mutations
            mutated_here = False
            for qa in q_alleles:
                for ca in c_alleles:
                    if abs(qa - ca) <= 1.0:
                        mutated_here = True
                        break
                if mutated_here:
                    break

            if mutated_here:
                mutated_loci += 1
                score += 1.0
            else:
                score -= 0.5  # penalize complete mismatch

        total_informative = consistent_loci + mutated_loci
        if total_informative == 0:
            continue  # no evidence of relatedness

        # Convert simple score to pseudo-CLR and posterior
        clr = max(score, 0.01)
        posterior = clr / (clr + 1.0)

        candidates.append(
            {
                "person_id": candidate_id,
                "clr": clr,
                "posterior": posterior,
                "consistent_loci": consistent_loci,
                "mutated_loci": mutated_loci,
                "inconclusive_loci": inconclusive_loci,
            }
        )

    candidates.sort(key=lambda x: x["clr"], reverse=True)
    return candidates[:10]


# ============================================================
# DO NOT MODIFY BELOW THIS LINE — This runs your function!
# ============================================================


def find_matches(database_path: str, queries_path: str) -> List[Dict]:
    """
    Main entry point — automatically tested by CI.
    Loads data and calls your match_single for each query.
    """
    print("Loading database and queries...")
    database_df = pd.read_csv(database_path)
    queries_df = pd.read_csv(queries_path)

    results = []

    print(f"Processing {len(queries_df)} queries...")
    for _, query_row in queries_df.iterrows():
        query_id = query_row["PersonID"]
        query_profile = query_row.to_dict()

        print(f"  Matching query {query_id}...")
        top_candidates = match_single(query_profile, database_df)

        results.append(
            {
                "query_id": query_id,
                "top_candidates": top_candidates[:10],  # Ensure max 10
            }
        )

    print("All queries processed.")
    return results
