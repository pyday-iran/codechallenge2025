# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
from typing import List, Dict, Any


# ============================================================
# TUNABLE PARAMETERS
# ============================================================
PARTIAL_MIN = 1
MISMATCH_MAX = 3
PARTIAL_BONUS = 3.0
PREFILTER_MIN = 5
IDENTICAL_RATIO_MAX = 0.9
PARTIAL_THRESHOLD = 3

DEFAULT_FREQ = 0.01


def normalize_str(value) -> str:
    """Normalize allele string for comparison."""
    if pd.isna(value) or str(value).strip() in ("-", ""):
        return "-"
    return str(value).strip()


def parse_alleles(value) -> frozenset:
    """Parse allele string into a frozenset of float values."""
    if pd.isna(value) or str(value).strip() in ("-", ""):
        return frozenset()
    
    parts = str(value).strip().split(",")
    alleles = set()
    for p in parts:
        p = p.strip()
        if p and p != "-":
            try:
                alleles.add(float(p))
            except ValueError:
                pass
    return frozenset(alleles)


def build_cache(database_df: pd.DataFrame) -> dict:
    """Build cache with index and frequencies (runs once)."""
    loci = [col for col in database_df.columns if col != "PersonID"]
    total = len(database_df)
    
    allele_index = {}
    allele_counts = {}
    profiles = {}
    
    for _, row in database_df.iterrows():
        pid = row["PersonID"]
        profile = {}
        
        for locus in loci:
            raw_val = row[locus]
            alleles = parse_alleles(raw_val)
            raw_str = normalize_str(raw_val)
            profile[locus] = (alleles, raw_str)
            
            for a in alleles:
                key = (locus, a)
                if key not in allele_index:
                    allele_index[key] = set()
                    allele_counts[key] = 0
                allele_index[key].add(pid)
                allele_counts[key] += 1
        
        profiles[pid] = profile
    
    allele_freqs = {}
    for (locus, allele), count in allele_counts.items():
        if locus not in allele_freqs:
            allele_freqs[locus] = {}
        allele_freqs[locus][allele] = count / total
    
    return {
        "loci": loci,
        "allele_index": allele_index,
        "allele_freqs": allele_freqs,
        "profiles": profiles,
    }


def score_candidate(query_parsed, cand_profile, loci, allele_freqs, 
                    partial_min, mismatch_max, partial_bonus):
    """Score a single candidate with given parameters."""
    def get_freq(locus, allele):
        if locus in allele_freqs and allele in allele_freqs[locus]:
            return allele_freqs[locus][allele]
        return DEFAULT_FREQ
    
    total_lr = 1.0
    identical_count = 0
    partial_count = 0
    mismatch_count = 0
    mutation_count = 0
    missing_count = 0
    
    for locus in loci:
        q_alleles, q_raw = query_parsed[locus]
        c_alleles, c_raw = cand_profile[locus]
        
        if q_raw == "-" or c_raw == "-" or not q_alleles or not c_alleles:
            missing_count += 1
            continue
        
        shared = q_alleles & c_alleles
        
        if not shared:
            is_mutation = any(0 < abs(q - c) <= 1.0 for q in q_alleles for c in c_alleles)
            if is_mutation:
                mutation_count += 1
                total_lr *= 0.1
            else:
                mismatch_count += 1
                total_lr *= 0.0001
            continue
        
        min_freq = min(get_freq(locus, a) for a in shared)
        total_lr *= 1.0 / min_freq
        
        if q_raw == c_raw:
            identical_count += 1
        else:
            partial_count += 1
    
    # Apply filters
    if mismatch_count > mismatch_max:
        return None
    if partial_count < partial_min:
        return None
    
    # Clever filter: reject self-matches or identical twins
    comparable = identical_count + partial_count
    if comparable > 0:
        identical_ratio = identical_count / comparable
        if identical_ratio > IDENTICAL_RATIO_MAX and partial_count < PARTIAL_THRESHOLD:
            return None
    
    score = total_lr * (1 + partial_count * partial_bonus)
    
    # Bayesian posterior with 50% prior
    posterior = score / (score + 1.0) if score > 0 else 0.0
    
    return {
        "clr": score,
        "posterior": posterior,
        "consistent_loci": identical_count + partial_count,
        "mutated_loci": mutation_count,
        "inconclusive_loci": missing_count,
    }


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.
    """
    db_id = id(database_df)
    if not hasattr(match_single, "_cache") or match_single._cache.get("db_id") != db_id:
        match_single._cache = build_cache(database_df)
        match_single._cache["db_id"] = db_id
    
    cache = match_single._cache
    loci = cache["loci"]
    allele_index = cache["allele_index"]
    allele_freqs = cache["allele_freqs"]
    profiles = cache["profiles"]
    
    query_id = query_profile["PersonID"]
    
    query_parsed = {}
    for locus in loci:
        val = query_profile.get(locus, "-")
        query_parsed[locus] = (parse_alleles(val), normalize_str(val))
    
    candidate_shared = {}
    for locus in loci:
        q_alleles, _ = query_parsed[locus]
        for allele in q_alleles:
            key = (locus, allele)
            if key in allele_index:
                for pid in allele_index[key]:
                    if pid != query_id:
                        candidate_shared[pid] = candidate_shared.get(pid, 0) + 1
    
    promising = [pid for pid, cnt in candidate_shared.items() if cnt >= PREFILTER_MIN]
    
    candidates = []
    for pid in promising:
        result = score_candidate(
            query_parsed, profiles[pid], loci, allele_freqs,
            PARTIAL_MIN, MISMATCH_MAX, PARTIAL_BONUS
        )
        if result:
            result["person_id"] = pid
            candidates.append(result)
    
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
