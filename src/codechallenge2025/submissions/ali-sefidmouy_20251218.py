# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
from typing import List, Dict, Any


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame) -> List[Dict]:
    query_id = query_profile["PersonID"]

    loci = [
        col
        for col in database_df.columns
        if col != "PersonID"
    ]

    def parse_alleles(value: Any) -> set:
        if pd.isna(value):
            return set()
        if isinstance(value, (int, float)):
            return {str(value)}
        text = str(value).strip()
        if not text:
            return set()
        return {a.strip() for a in text.split(",") if a.strip()}

    query_alleles = {
        locus: parse_alleles(query_profile.get(locus, ""))
        for locus in loci
    }

    candidates: List[Dict] = []

    for _, row in database_df.iterrows():
        person_id = row["PersonID"]
        if person_id == query_id:
            continue

        consistent_loci = 0
        inconclusive_loci = 0

        for locus in loci:
            q_set = query_alleles.get(locus, set())
            d_set = parse_alleles(row.get(locus, ""))

            if not q_set or not d_set:
                inconclusive_loci += 1
                continue

            if q_set & d_set:
                consistent_loci += 1
            else:
                inconclusive_loci += 1

        if consistent_loci == 0:
            continue

        clr = float(consistent_loci)
        posterior = min(0.999999, 1.0 - 1e-6 / (1.0 + clr))

        candidates.append(
            {
                "person_id": person_id,
                "clr": clr,
                "posterior": posterior,
                "consistent_loci": consistent_loci,
                "mutated_loci": 0,
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
