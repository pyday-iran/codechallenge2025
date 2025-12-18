# src/codechallenge2025/participant_solution.py
"""
Forensic STR Parent-Child Relationship Detector for #codechallenge2025

Implements efficient parent-child matching using:
- Inverted index for fast candidate filtering based on shared alleles
- Combined Likelihood Ratio (CLR) calculation for accurate scoring
- Support for mutations (±1 step), microvariants, and missing data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict

# ============================================================
# Population allele frequencies (from forensic reference data)
# ============================================================

ALLELE_FREQS = {
    "D3S1358": {14: 0.15, 15: 0.25, 16: 0.22, 17: 0.20, 18: 0.13, 19: 0.05},
    "vWA": {14: 0.10, 15: 0.12, 16: 0.20, 17: 0.25, 18: 0.20, 19: 0.10, 20: 0.03},
    "FGA": {19: 0.05, 20: 0.10, 21: 0.15, 22: 0.20, 23: 0.18, 24: 0.15, 25: 0.10, 26: 0.07},
    "D8S1179": {10: 0.05, 11: 0.08, 12: 0.10, 13: 0.30, 14: 0.25, 15: 0.15, 16: 0.07},
    "D21S11": {27: 0.05, 28: 0.15, 29: 0.20, 30: 0.25, 31: 0.15, 32: 0.10, 30.2: 0.08, 31.2: 0.02},
    "D18S51": {12: 0.08, 13: 0.15, 14: 0.20, 15: 0.18, 16: 0.12, 17: 0.10, 18: 0.08, 19: 0.06, 20: 0.03},
    "D5S818": {9: 0.05, 10: 0.08, 11: 0.25, 12: 0.30, 13: 0.20, 14: 0.10, 15: 0.02},
    "D13S317": {8: 0.05, 9: 0.08, 10: 0.10, 11: 0.25, 12: 0.20, 13: 0.18, 14: 0.12, 15: 0.02},
    "D7S820": {8: 0.10, 9: 0.12, 10: 0.25, 11: 0.28, 12: 0.15, 13: 0.08, 14: 0.02},
    "D16S539": {8: 0.05, 9: 0.20, 10: 0.15, 11: 0.25, 12: 0.20, 13: 0.10, 14: 0.05},
    "TH01": {6: 0.20, 7: 0.15, 8: 0.18, 9: 0.22, 9.3: 0.15, 10: 0.08, 11: 0.02},
    "TPOX": {8: 0.40, 9: 0.10, 10: 0.12, 11: 0.25, 12: 0.10, 13: 0.03},
    "CSF1PO": {9: 0.05, 10: 0.20, 11: 0.25, 12: 0.30, 13: 0.12, 14: 0.08},
    "D2S1338": {17: 0.08, 18: 0.05, 19: 0.10, 20: 0.15, 21: 0.08, 22: 0.07, 23: 0.12, 24: 0.15, 25: 0.15},
    "D19S433": {13: 0.15, 14: 0.30, 14.2: 0.05, 15: 0.20, 15.2: 0.05, 16: 0.15, 17: 0.10},
    "D22S1045": {11: 0.10, 14: 0.08, 15: 0.30, 16: 0.35, 17: 0.12, 18: 0.05},
    "D10S1248": {11: 0.05, 12: 0.08, 13: 0.25, 14: 0.30, 15: 0.20, 16: 0.10, 17: 0.02},
    "D1S1656": {12: 0.10, 13: 0.08, 14: 0.05, 15: 0.12, 16: 0.15, 17: 0.20, 17.3: 0.10, 18: 0.10, 18.3: 0.05},
    "D12S391": {17: 0.05, 18: 0.15, 19: 0.12, 20: 0.20, 21: 0.18, 22: 0.15, 23: 0.10, 24: 0.05},
    "D2S441": {10: 0.10, 11: 0.20, 11.3: 0.05, 12: 0.08, 13: 0.10, 14: 0.25, 15: 0.15, 16: 0.07},
    "SE33": {19: 0.05, 20: 0.08, 21: 0.10, 22: 0.12, 23: 0.10, 24: 0.08, 25: 0.12, 26: 0.10, 27: 0.10, 28: 0.08, 29: 0.07},
}

# Normalize frequencies
for locus in ALLELE_FREQS:
    total = sum(ALLELE_FREQS[locus].values())
    for allele in ALLELE_FREQS[locus]:
        ALLELE_FREQS[locus][allele] /= total

# Known loci list
LOCI = list(ALLELE_FREQS.keys())

# Mutation rate per locus per generation
MUTATION_RATE = 0.002
DEFAULT_FREQ = 0.01  # For unknown alleles

# ============================================================
# Module-level cache for database preprocessing
# ============================================================

_db_cache = {
    "hash": None,
    "profiles": {},      # pid -> {locus: set of alleles}
    "allele_index": {},  # (locus, allele) -> set of pids
    "loci": [],
}


def parse_alleles(allele_str: Any) -> Set[float]:
    """Parse allele string into set of float values."""
    if pd.isna(allele_str) or str(allele_str).strip() in ("-", ""):
        return set()
    s = str(allele_str).strip()
    if "," in s:
        return {float(x.strip()) for x in s.split(",")}
    return {float(s)}


def get_allele_freq(locus: str, allele: float) -> float:
    """Get population frequency for an allele at a locus."""
    freqs = ALLELE_FREQS.get(locus, {})
    return freqs.get(allele, DEFAULT_FREQ)


def _build_database_cache(database_df: pd.DataFrame) -> None:
    """Build inverted index and profile cache from database."""
    global _db_cache

    profiles = {}
    allele_index = defaultdict(set)
    loci = [c for c in database_df.columns if c != "PersonID"]

    for _, row in database_df.iterrows():
        pid = row["PersonID"]
        profile = {}
        for locus in loci:
            alleles = parse_alleles(row[locus])
            profile[locus] = alleles
            for allele in alleles:
                allele_index[(locus, allele)].add(pid)
        profiles[pid] = profile

    _db_cache["profiles"] = profiles
    _db_cache["allele_index"] = dict(allele_index)
    _db_cache["loci"] = loci
    _db_cache["hash"] = id(database_df)


def compute_locus_lr(
    query_alleles: Set[float],
    candidate_alleles: Set[float],
    locus: str
) -> Tuple[float, str]:
    """
    Compute likelihood ratio for a single locus.

    Returns:
        (lr, status) where status is one of:
        - 'consistent': direct allele match
        - 'mutated': match via ±1 step mutation
        - 'inconclusive': missing data or possible dropout match
        - 'excluded': no possible match
    """
    # Handle missing data
    if not query_alleles or not candidate_alleles:
        return 1.0, "inconclusive"

    # Find direct shared alleles
    shared = query_alleles & candidate_alleles

    if shared:
        # Direct match - compute LR using Paternity Index formula
        # LR = transmission_prob / allele_frequency
        best_lr = 0.0
        for allele in shared:
            # Transmission probability: 1.0 if homozygous, 0.5 if heterozygous
            trans_prob = 1.0 if len(candidate_alleles) == 1 else 0.5
            freq = get_allele_freq(locus, allele)
            lr = trans_prob / max(freq, 0.001)
            best_lr = max(best_lr, lr)
        return best_lr, "consistent"

    # Check for mutation (±1 step difference)
    for qa in query_alleles:
        for ca in candidate_alleles:
            diff = abs(qa - ca)
            # Allow ±1 step for integers, or small diff for microvariants
            if 0 < diff <= 1.0:
                trans_prob = 1.0 if len(candidate_alleles) == 1 else 0.5
                freq = get_allele_freq(locus, qa)
                # Penalize by mutation rate
                lr = (trans_prob * MUTATION_RATE) / max(freq, 0.001)
                return max(lr, 0.001), "mutated"

    # Special case: single allele in both (possible dropout masking match)
    # If both show only 1 allele with no match, the dropped alleles might match
    if len(query_alleles) == 1 and len(candidate_alleles) == 1:
        # Treat as inconclusive with slight penalty (dropout probability ~5%)
        return 0.5, "inconclusive"

    # Check for possible dropout scenario with larger difference
    # If one side has single allele, the dropped allele might have been
    # the transmitted one
    if len(query_alleles) == 1 or len(candidate_alleles) == 1:
        for qa in query_alleles:
            for ca in candidate_alleles:
                diff = abs(qa - ca)
                # ±2 step could be mutation + dropout combination
                if 1.0 < diff <= 2.0:
                    # Very rare: double-step mutation
                    lr = MUTATION_RATE * MUTATION_RATE * 0.5
                    return max(lr, 0.0001), "mutated"

    # Complete mismatch - exclusion
    return 0.0, "excluded"


def score_candidate(
    query_profile: Dict[str, Set[float]],
    candidate_profile: Dict[str, Set[float]],
    loci: List[str]
) -> Optional[Dict]:
    """
    Compute full CLR score for a candidate.

    Returns:
        Candidate dict with scores, or None if excluded.
    """
    clr = 1.0
    consistent_loci = 0
    mutated_loci = 0
    inconclusive_loci = 0
    exclusions = 0

    # Track identity matches (both alleles identical) to detect same-person
    identity_matches = 0
    compared_loci = 0

    for locus in loci:
        q_alleles = query_profile.get(locus, set())
        c_alleles = candidate_profile.get(locus, set())

        # Skip if either has missing data for identity check
        if q_alleles and c_alleles:
            compared_loci += 1
            # Check for identical genotype (same-person indicator)
            if q_alleles == c_alleles:
                identity_matches += 1

        lr, status = compute_locus_lr(q_alleles, c_alleles, locus)

        if status == "excluded":
            exclusions += 1
            # Apply progressive penalty: each exclusion gets worse
            # 1st exclusion: 0.01, 2nd: 0.001, 3rd: 0.0001, etc.
            penalty = 10 ** (-2 - exclusions)
            clr *= penalty
        elif status == "consistent":
            consistent_loci += 1
            clr *= lr
        elif status == "mutated":
            mutated_loci += 1
            clr *= lr
        else:  # inconclusive
            inconclusive_loci += 1
            # No change to CLR for missing data

    # Hard cutoff: too many exclusions means definitely not related
    if exclusions > 4:
        return None

    # Must have reasonable number of consistent loci
    if consistent_loci < 5:
        return None

    # Filter out near-identical profiles (same person, not parent-child)
    # In true parent-child, expect ~50% identity at each locus on average
    # If >80% of compared loci have identical genotypes, likely same person
    if compared_loci > 0:
        identity_ratio = identity_matches / compared_loci
        if identity_ratio > 0.80:
            return None  # Same person/twin, not parent-child

    # Compute posterior probability with 50% prior
    posterior = clr / (clr + 1.0) if clr > 0 else 0.0

    return {
        "clr": clr,
        "posterior": posterior,
        "consistent_loci": consistent_loci,
        "mutated_loci": mutated_loci,
        "inconclusive_loci": inconclusive_loci,
    }


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.

    Args:
        query_profile: dict with 'PersonID' and locus columns
        database_df: Full database as pandas DataFrame

    Returns:
        List of up to 10 candidate dicts, sorted by CLR (best first)
    """
    global _db_cache

    # Build/update cache if needed
    if _db_cache["hash"] != id(database_df):
        _build_database_cache(database_df)

    profiles = _db_cache["profiles"]
    allele_index = _db_cache["allele_index"]
    loci = _db_cache["loci"]
    query_id = query_profile["PersonID"]

    # Parse query alleles
    query_parsed = {}
    for locus in loci:
        query_parsed[locus] = parse_alleles(query_profile.get(locus, "-"))

    # Step 1: Fast candidate filtering using inverted index
    # Score candidates by weighted allele overlap (weight by rarity)
    candidate_scores = defaultdict(float)

    for locus in loci:
        for allele in query_parsed[locus]:
            key = (locus, allele)
            if key in allele_index:
                freq = get_allele_freq(locus, allele)
                # Weight by inverse frequency (rare alleles score higher)
                weight = 1.0 / max(freq, 0.01)
                for pid in allele_index[key]:
                    if pid != query_id:
                        candidate_scores[pid] += weight

    # Get top candidates by preliminary score
    if not candidate_scores:
        return []

    top_candidates = sorted(
        candidate_scores.keys(),
        key=lambda x: -candidate_scores[x]
    )[:1000]  # Consider top 1000 for detailed scoring

    # Step 2: Detailed CLR scoring for top candidates
    results = []

    for pid in top_candidates:
        candidate_profile = profiles[pid]
        score_result = score_candidate(query_parsed, candidate_profile, loci)

        if score_result is not None:
            results.append({
                "person_id": pid,
                "clr": score_result["clr"],
                "posterior": score_result["posterior"],
                "consistent_loci": score_result["consistent_loci"],
                "mutated_loci": score_result["mutated_loci"],
                "inconclusive_loci": score_result["inconclusive_loci"],
            })

    # Sort by CLR descending and return top 10
    results.sort(key=lambda x: -x["clr"])
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
