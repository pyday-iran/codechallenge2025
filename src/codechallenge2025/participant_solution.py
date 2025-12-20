# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""
import pandas as pd
import math
from typing import List, Dict, Any

MUTATION_RATE = 0.002
MIN_LR = 1e-6

def parse_alleles(val):
    if pd.isna(val) or val == "-" or val == "":
        return None
    return set(float(x) for x in str(val).split(","))


def allele_relation(a, b):
    if a is None or b is None:
        return "missing"
    for x in a:
        for y in b:
            if x == y:
                return "match"
            if abs(x - y) == 1:
                return "mutation"
    return "mismatch"


def locus_lr(status, allele_freq=0.05):
    if status == "match":
        return 1.0 / allele_freq
    if status == "mutation":
        return MUTATION_RATE / allele_freq
    if status == "missing":
        return 1.0
    return MIN_LR


# ---------- ONLY FUNCTION YOU IMPLEMENT ----------

def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:

    # 1) Extract loci and query id
    loci = [c for c in database_df.columns if c != "PersonID"]
    query_id = query_profile["PersonID"]

    # 2) Parse query profile once
    query_alleles = {l: parse_alleles(query_profile[l]) for l in loci}

    # 3) Pre-filter candidates (performance critical)
    informative_loci = [l for l in loci if query_alleles[l] is not None][:5]

    candidates = []
    for _, row in database_df.iterrows():
        if row["PersonID"] == query_id:
            continue

        shared = 0
        for l in informative_loci:
            rel = allele_relation(query_alleles[l], parse_alleles(row[l]))
            if rel in ("match", "mutation"):
                shared += 1

        if shared >= 2:
            candidates.append(row)

    # 4) Score candidates using LR in log-space
    results = []

    for row in candidates:
        log_lr = 0.0
        consistent = 0
        mutated = 0
        inconclusive = 0

        for l in loci:
            qa = query_alleles[l]
            da = parse_alleles(row[l])
            status = allele_relation(qa, da)

            if status == "match":
                consistent += 1
            elif status == "mutation":
                mutated += 1
            elif status == "missing":
                inconclusive += 1

            log_lr += math.log(locus_lr(status))

        clr = math.exp(log_lr)
        posterior = clr / (clr + 1)

        results.append(
            {
                "person_id": row["PersonID"],
                "clr": clr,
                "posterior": posterior,
                "consistent_loci": consistent,
                "mutated_loci": mutated,
                "inconclusive_loci": inconclusive,
            }
        )

    # 5) Sort and return top 10
    results.sort(key=lambda x: x["clr"], reverse=True)
    return results[:10]



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
