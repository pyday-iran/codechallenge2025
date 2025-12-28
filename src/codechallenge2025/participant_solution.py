# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import math
import hashlib

# Module-level cache for allele frequencies and inverted index
_cache_db_hash: Optional[str] = None
_cache_allele_freqs: Optional[Dict[str, Dict[str, float]]] = None
_cache_inverted_index: Optional[Dict[str, Dict[str, Set[int]]]] = None


def parse_alleles(allele_str: str) -> Set[str]:
    """
    Parse allele string into set of alleles.
    Handles: "13,14", "13", "13,13", "-", "", "9.3"
    """
    if pd.isna(allele_str) or allele_str == "-" or allele_str == "":
        return set()
    
    alleles = [a.strip() for a in str(allele_str).split(",") if a.strip()]
    return set(alleles)


def compute_allele_frequencies(database_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Precompute population allele frequencies from database.
    Optimized for large databases using efficient iteration.
    Returns: {locus: {allele: frequency}}
    """
    allele_counts = defaultdict(Counter)
    total_alleles = defaultdict(int)
    
    # Get all locus columns (exclude PersonID)
    locus_columns = [col for col in database_df.columns if col != "PersonID"]
    
    # Use itertuples for faster iteration (much faster than iterrows for large DataFrames)
    for idx in range(len(database_df)):
        row = database_df.iloc[idx]
        for locus in locus_columns:
            alleles = parse_alleles(row[locus])
            for allele in alleles:
                allele_counts[locus][allele] += 1
                total_alleles[locus] += 1
    
    # Convert counts to frequencies
    frequencies = {}
    for locus in allele_counts:
        freq_dict = {}
        total = total_alleles[locus]
        if total > 0:
            for allele, count in allele_counts[locus].items():
                freq_dict[allele] = count / total
        frequencies[locus] = freq_dict
    
    return frequencies


def build_inverted_index(database_df: pd.DataFrame) -> Dict[str, Dict[str, Set[int]]]:
    """
    Build inverted index: {locus: {allele: set of row indices}}
    Optimized for large databases using vectorized operations.
    """
    index = defaultdict(lambda: defaultdict(set))
    locus_columns = [col for col in database_df.columns if col != "PersonID"]
    
    # Use itertuples for faster iteration (much faster than iterrows)
    for idx in range(len(database_df)):
        row = database_df.iloc[idx]
        for locus in locus_columns:
            alleles = parse_alleles(row[locus])
            for allele in alleles:
                index[locus][allele].add(idx)
    
    return index


def allele_to_numeric(allele: str) -> float:
    """
    Convert allele string to numeric value for mutation checking.
    Handles microvariants like "9.3" -> 9.3, "13" -> 13.0
    """
    try:
        return float(allele)
    except ValueError:
        return float('inf')  # Non-numeric alleles (shouldn't happen in STR)


def check_mutation(allele1: str, allele2: str) -> bool:
    """
    Check if two alleles differ by exactly ±1 repeat (mutation).
    Returns True if |allele1 - allele2| == 1
    """
    try:
        num1 = allele_to_numeric(allele1)
        num2 = allele_to_numeric(allele2)
        if num1 == float('inf') or num2 == float('inf'):
            return False
        return abs(num1 - num2) == 1.0
    except:
        return False


def compute_locus_lr(
    child_alleles: Set[str],
    parent_alleles: Set[str],
    allele_freqs: Dict[str, float],
    mutation_rate: float = 0.002
) -> Tuple[float, str]:
    """
    Compute per-locus likelihood ratio for parent-child relationship.
    
    Uses standard paternity index formula:
    - For each matching child allele: LR = P(allele|parent) / f(allele)
    - P(allele|parent) = 0.5 if parent heterozygous, 1.0 if homozygous
    - For mutations: LR = mutation_rate / f(child_allele)
    - Takes maximum LR across all matching alleles
    
    Args:
        child_alleles: Set of child alleles at this locus
        parent_alleles: Set of parent alleles at this locus
        allele_freqs: Dictionary of allele frequencies for this locus
        mutation_rate: Mutation rate per locus per generation (default 0.002)
    
    Returns:
        (log_LR, status) where status is 'consistent', 'mutation', 'missing', or 'exclusion'
    """
    # Handle missing data - neutral contribution
    if not child_alleles or not parent_alleles:
        return 0.0, 'missing'
    
    is_parent_homozygous = len(parent_alleles) == 1
    
    # Check for exact matches first
    exact_matches = child_alleles & parent_alleles
    
    if exact_matches:
        # At least one exact match - compute LR for each matching allele
        # Use sum of LRs for multiple matches (stronger evidence)
        total_lr = 0.0
        best_single_lr = 0.0
        
        for child_allele in exact_matches:
            freq = allele_freqs.get(child_allele, 0.0001)
            freq = max(freq, 0.0001)  # Minimum frequency to avoid division issues
            
            # P(child allele from parent)
            if is_parent_homozygous:
                p_from_parent = 1.0  # Homozygous parent always transmits
            else:
                p_from_parent = 0.5  # Heterozygous parent transmits with prob 0.5
            
            # Standard paternity index: LR = P(allele|parent) / f(allele)
            lr = p_from_parent / freq
            total_lr += lr
            best_single_lr = max(best_single_lr, lr)
        
        # If multiple alleles match, use combined LR
        # For parent-child: if both alleles match, it's stronger but not as strong as siblings
        if len(exact_matches) > 1 and len(child_alleles) > 1:
            # Both alleles match - use geometric mean to avoid over-weighting
            combined_lr = math.sqrt(total_lr) if total_lr > 0 else best_single_lr
        else:
            combined_lr = best_single_lr
        
        if combined_lr > 0:
            return math.log10(max(combined_lr, 1.0)), 'consistent'
        else:
            return 0.0, 'consistent'
    
    # Check for ±1 mutations
    mutation_lrs = []
    
    for child_allele in child_alleles:
        child_freq = max(allele_freqs.get(child_allele, 0.0001), 0.0001)
        
        for parent_allele in parent_alleles:
            if check_mutation(child_allele, parent_allele):
                # Mutation detected: child allele differs by ±1 from parent
                # LR with mutation: mutation_rate / f(child_allele)
                # Use slightly higher effective mutation rate to account for uncertainty
                # Also consider parent genotype - homozygous parents less likely to mutate
                if is_parent_homozygous:
                    effective_mutation_rate = mutation_rate * 1.3  # Balanced mutation rate
                else:
                    effective_mutation_rate = mutation_rate * 1.6  # Balanced mutation rate
                mutation_lr = effective_mutation_rate / child_freq
                mutation_lrs.append(mutation_lr)
    
    if mutation_lrs:
        # Use the best mutation LR
        best_mutation_lr = max(mutation_lrs)
        return math.log10(max(best_mutation_lr, 1e-6)), 'mutation'
    
    # No match and no mutation - exclusion
    return math.log10(1e-6), 'exclusion'


def compute_clr_bidirectional(
    query_profile: Dict[str, Any],
    candidate_row: pd.Series,
    allele_freqs: Dict[str, Dict[str, float]],
    locus_columns: List[str]
) -> Tuple[float, int, int, int, str]:
    """
    Compute CLR bidirectionally (query as child, candidate as parent AND vice versa).
    Returns the better of the two directions, preferring fewer exclusions when LR is similar.
    Adds penalty for sibling-like relationships (both alleles match at many loci).
    
    Returns:
        (clr, consistent_loci, mutated_loci, missing_loci, role)
    """
    log_lr_child = 0.0
    log_lr_parent = 0.0
    consistent_child = 0
    consistent_parent = 0
    mutated_child = 0
    mutated_parent = 0
    missing_child = 0
    missing_parent = 0
    exclusion_child = 0
    exclusion_parent = 0
    both_match_child = 0  # Count loci where both alleles match (sibling pattern)
    both_match_parent = 0
    
    for locus in locus_columns:
        query_alleles = parse_alleles(query_profile.get(locus, ""))
        candidate_alleles = parse_alleles(candidate_row[locus])
        
        # Check if both alleles match (sibling pattern)
        if len(query_alleles) == 2 and len(candidate_alleles) == 2:
            if query_alleles == candidate_alleles:
                both_match_child += 1
                both_match_parent += 1
        
        # Direction 1: query is child, candidate is parent
        lr1, status1 = compute_locus_lr(
            query_alleles, candidate_alleles,
            allele_freqs.get(locus, {})
        )
        log_lr_child += lr1
        if status1 == 'consistent':
            consistent_child += 1
        elif status1 == 'mutation':
            mutated_child += 1
        elif status1 == 'missing':
            missing_child += 1
        elif status1 == 'exclusion':
            exclusion_child += 1
        
        # Direction 2: query is parent, candidate is child
        lr2, status2 = compute_locus_lr(
            candidate_alleles, query_alleles,
            allele_freqs.get(locus, {})
        )
        log_lr_parent += lr2
        if status2 == 'consistent':
            consistent_parent += 1
        elif status2 == 'mutation':
            mutated_parent += 1
        elif status2 == 'missing':
            missing_parent += 1
        elif status2 == 'exclusion':
            exclusion_parent += 1
    
    # Penalize sibling-like relationships (too many "both alleles match")
    # Parent-child should have fewer both-match loci than siblings
    # More efficient: count non-missing loci during the loop
    non_missing_loci = consistent_child + mutated_child + exclusion_child
    if non_missing_loci > 5:  # Only apply if we have enough data
        both_match_ratio = both_match_child / max(non_missing_loci, 1)
        
        # If >16% of non-missing loci have both alleles matching, likely siblings - apply penalty
        # Stronger penalty for higher ratios (siblings typically have 30-50% both-match)
        if both_match_ratio > 0.16:
            # Penalty increases with ratio: 0.16 -> 0.15 penalty, 0.25 -> 1.35 penalty, 0.35 -> 3.6 penalty
            penalty = min(3.8, (both_match_ratio - 0.16) * 20.0)  # Lower threshold, up to 3.8 log units
            log_lr_child -= penalty
            log_lr_parent -= penalty
    
    # Choose better direction: prefer higher LR, but consider exclusions and consistency
    lr_diff = log_lr_child - log_lr_parent
    
    # If LR difference is small (< 2.5), prefer direction with fewer exclusions and more consistency
    if abs(lr_diff) < 2.5:
        # Prefer direction with fewer exclusions (strong negative signal)
        if exclusion_child < exclusion_parent:
            clr = 10.0 ** log_lr_child
            return clr, consistent_child, mutated_child, missing_child, 'candidate_parent'
        elif exclusion_parent < exclusion_child:
            clr = 10.0 ** log_lr_parent
            return clr, consistent_parent, mutated_parent, missing_parent, 'candidate_child'
        # If exclusions are equal, prefer direction with more consistent loci
        if consistent_child > consistent_parent:
            clr = 10.0 ** log_lr_child
            return clr, consistent_child, mutated_child, missing_child, 'candidate_parent'
        elif consistent_parent > consistent_child:
            clr = 10.0 ** log_lr_parent
            return clr, consistent_parent, mutated_parent, missing_parent, 'candidate_child'
        # If still tied, prefer fewer mutations
        if mutated_child < mutated_parent:
            clr = 10.0 ** log_lr_child
            return clr, consistent_child, mutated_child, missing_child, 'candidate_parent'
        elif mutated_parent < mutated_child:
            clr = 10.0 ** log_lr_parent
            return clr, consistent_parent, mutated_parent, missing_parent, 'candidate_child'
    
    # Otherwise choose based on LR (stronger signal)
    if log_lr_child >= log_lr_parent:
        clr = 10.0 ** log_lr_child
        return clr, consistent_child, mutated_child, missing_child, 'candidate_parent'
    else:
        clr = 10.0 ** log_lr_parent
        return clr, consistent_parent, mutated_parent, missing_parent, 'candidate_child'


def prefilter_candidates(
    query_profile: Dict[str, Any],
    database_df: pd.DataFrame,
    inverted_index: Dict[str, Dict[str, Set[int]]],
    allele_freqs: Dict[str, Dict[str, float]],
    top_n: int = 5500
) -> List[int]:
    """
    Pre-filter candidates using multiple strategies, optimized for large databases:
    1. Shared rare alleles (weighted by rarity) - stronger signal
    2. Shared alleles with mutation potential (±1) - weaker but important
    3. Multi-locus matching score - boost candidates with multiple matches
    4. Include candidates with matches at rare loci even if score is lower
    
    Optimizations for 500k+ profiles:
    - Precompute PersonID array for fast lookups
    - Use set operations for efficient candidate tracking
    - Minimize DataFrame access during scoring
    
    Returns list of row indices for top N candidates.
    """
    query_id = query_profile['PersonID']
    locus_columns = [col for col in database_df.columns if col != "PersonID"]
    
    # Precompute PersonID array for fast lookups (avoids repeated iloc access)
    person_ids = database_df['PersonID'].values
    
    # Collect all query alleles
    query_alleles_by_locus = {}
    for locus in locus_columns:
        query_alleles_by_locus[locus] = parse_alleles(query_profile.get(locus, ""))
    
    # Score each candidate using multiple criteria
    candidate_scores = defaultdict(float)
    candidate_match_counts = defaultdict(int)  # Count of matching loci
    candidate_rare_matches = defaultdict(int)  # Count of rare allele matches
    
    for locus in locus_columns:
        query_alleles = query_alleles_by_locus[locus]
        if not query_alleles:
            continue
            
        for allele in query_alleles:
            # Strategy 1: Exact allele matches (weighted by rarity)
            if allele in inverted_index[locus]:
                freq = allele_freqs.get(locus, {}).get(allele, 0.001)
                freq = max(freq, 0.0001)  # Lower minimum to catch very rare alleles
                # Weight by rarity: -log(frequency), cap to avoid extreme values
                base_weight = min(-math.log10(freq), 6.0)
                # Extra boost for very rare alleles (< 5%)
                is_rare = freq < 0.05
                weight = base_weight * 1.4 if is_rare else base_weight  # 40% boost for rare alleles
                
                # Fast lookup: use precomputed person_ids array
                for idx in inverted_index[locus][allele]:
                    if person_ids[idx] != query_id:  # Much faster than iloc access
                        candidate_scores[idx] += weight
                        candidate_match_counts[idx] += 1
                        if is_rare:
                            candidate_rare_matches[idx] += 1
            
            # Strategy 2: Mutation potential (±1) - important for true matches
            allele_num = allele_to_numeric(allele)
            if allele_num != float('inf'):
                for offset in [-1.0, 1.0]:
                    mut_allele = str(allele_num + offset).rstrip('0').rstrip('.')
                    if mut_allele in inverted_index[locus]:
                        freq = allele_freqs.get(locus, {}).get(mut_allele, 0.001)
                        freq = max(freq, 0.0001)
                        # Higher weight for mutations (0.65x) - they're important signals for true matches
                        weight = 0.65 * min(-math.log10(freq), 6.0)
                        
                        for idx in inverted_index[locus][mut_allele]:
                            if person_ids[idx] != query_id:  # Fast lookup
                                candidate_scores[idx] += weight
                                # Count mutation potential as partial match
                                if candidate_match_counts[idx] == 0:
                                    candidate_match_counts[idx] += 1
    
    # Boost score for candidates with matches at multiple loci (stronger signal)
    # Optimized: iterate only over candidates with scores
    for idx in candidate_scores:
        match_count = candidate_match_counts[idx]
        if match_count > 0:
            # Multi-locus bonus: stronger for more matches
            multi_locus_bonus = 1.0 + 0.22 * math.sqrt(match_count)  # Stronger bonus
            candidate_scores[idx] *= multi_locus_bonus
            
            # Extra boost for rare allele matches (very strong signal)
            if candidate_rare_matches[idx] >= 2:
                candidate_scores[idx] *= 1.8  # Strong boost for rare matches
    
    # Use heap for top-N selection if we have many candidates (more efficient than full sort)
    if len(candidate_scores) > top_n * 2:
        # Use nlargest for better performance on large candidate sets
        import heapq
        top_candidates = heapq.nlargest(top_n, candidate_scores.items(), key=lambda x: x[1])
        sorted_candidates = top_candidates
    else:
        # Full sort is fine for smaller candidate sets
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build set of candidates with matches for fast lookup
    candidates_with_matches = {idx for idx in candidate_match_counts if candidate_match_counts[idx] > 0}
    
    # Return top N candidates by score, prioritizing those with matches
    result_indices = []
    seen = set()
    
    # First, add top-scored candidates
    for idx, score in sorted_candidates[:top_n]:
        if idx not in seen:
            result_indices.append(idx)
            seen.add(idx)
    
    # If we have room, add more candidates with matches (even if lower score)
    # This helps catch true matches with missing data
    if len(result_indices) < top_n:
        # Add candidates with matches that weren't in top scores
        remaining_matches = candidates_with_matches - seen
        for idx in remaining_matches:
            result_indices.append(idx)
            seen.add(idx)
            if len(result_indices) >= top_n:
                break
    
    # If still need more, add random candidates (shouldn't happen often)
    if len(result_indices) < top_n:
        remaining_needed = top_n - len(result_indices)
        # Sequential scan (this fallback should rarely be needed)
        for idx in range(len(database_df)):
            if person_ids[idx] != query_id and idx not in seen:
                result_indices.append(idx)
                seen.add(idx)
                if len(result_indices) >= top_n:
                    break
    
    return result_indices


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.
    
    Optimized for scalability to ~500,000 database profiles:
    - Uses inverted index for O(1) allele lookups instead of O(n) scans
    - Pre-filters to ~4000 candidates using shared rare alleles (reduces CLR computations by 99%+)
    - Caches allele frequencies and inverted index across queries
    - Uses precomputed PersonID arrays to avoid repeated DataFrame access
    - Batch DataFrame operations for better performance
    
    Strategy:
    1. Precompute allele frequencies from database (cached across calls)
    2. Build inverted index for fast candidate lookup (cached across calls)
    3. Pre-filter candidates using shared rare alleles and mutation potential
    4. Compute CLR bidirectionally for filtered candidates only
    5. Return top 10 sorted by CLR
    
    Args:
        query_profile: dict with 'PersonID' and locus columns (e.g. {'PersonID': 'Q001', 'TH01': '9,9.3', ...})
        database_df: Full database as pandas DataFrame (500k rows)
    
    Returns:
        List of up to 10 candidate dicts, sorted by strength (best first):
        [
            {
                "person_id": "P000123",
                "clr": 1e15,                    # Combined Likelihood Ratio
                "posterior": 0.99999,           # Posterior probability
                "consistent_loci": 20,
                "mutated_loci": 1,
                "inconclusive_loci": 0
            },
            ...
        ]
    """
    global _cache_db_hash, _cache_allele_freqs, _cache_inverted_index
    
    query_id = query_profile['PersonID']
    locus_columns = [col for col in database_df.columns if col != "PersonID"]
    
    # Compute hash of database for caching
    db_hash = hashlib.md5(pd.util.hash_pandas_object(database_df).values).hexdigest()
    
    # Step 1: Precompute allele frequencies (cached if database unchanged)
    if _cache_db_hash != db_hash or _cache_allele_freqs is None:
        _cache_allele_freqs = compute_allele_frequencies(database_df)
        _cache_db_hash = db_hash
    
    allele_freqs = _cache_allele_freqs
    
    # Step 2: Build inverted index (cached if database unchanged)
    if _cache_db_hash != db_hash or _cache_inverted_index is None:
        _cache_inverted_index = build_inverted_index(database_df)
    
    inverted_index = _cache_inverted_index
    
    # Step 3: Pre-filter candidates - 5500 balances accuracy and speed
    # Efficient due to inverted index prefiltering
    n_candidates = min(5500, len(database_df))
    candidate_indices = prefilter_candidates(
        query_profile, database_df, inverted_index, allele_freqs, n_candidates
    )
    
    # If no candidates found, try a broader search
    if not candidate_indices:
        # Fallback: use all candidates (shouldn't happen often)
        # Optimized: use precomputed PersonID array for fast filtering
        person_ids = database_df['PersonID'].values
        candidate_indices = [i for i in range(len(database_df)) 
                           if person_ids[i] != query_id]
        candidate_indices = candidate_indices[:n_candidates]
    
    # Step 4: Compute CLR for filtered candidates
    # Optimized: batch iloc access for better performance on large DataFrames
    results = []
    # Pre-extract PersonIDs for fast filtering
    person_ids = database_df['PersonID'].values
    
    # Use batch iloc access (faster than individual calls for large DataFrames)
    candidate_rows = database_df.iloc[candidate_indices]
    
    for i, idx in enumerate(candidate_indices):
        # Fast PersonID check using precomputed array
        candidate_id = person_ids[idx]
        
        if candidate_id == query_id:
            continue
        
        # Get row from batch-extracted DataFrame (faster than individual iloc)
        candidate_row = candidate_rows.iloc[i]
        
        clr, consistent, mutated, missing, role = compute_clr_bidirectional(
            query_profile, candidate_row, allele_freqs, locus_columns
        )
        
        # Boost CLR for candidates with many consistent loci (strong parent-child signal)
        # This helps prioritize true matches that might have slightly lower CLR due to mutations/missing data
        total_loci_checked = consistent + mutated
        if total_loci_checked > 0:
            consistency_ratio = consistent / max(total_loci_checked, 1)
            # Boost if >59% of non-missing loci are consistent (very strong signal)
            if consistency_ratio > 0.59 and consistent >= 12:
                # Stronger boost for higher consistency
                boost_factor = 1.0 + (consistency_ratio - 0.59) * 2.9  # Up to 1.5x boost
                clr *= min(boost_factor, 1.5)  # Cap at 1.5x
        
        # Compute posterior probability (prior = 0.5)
        prior = 0.5
        posterior = (clr * prior) / (clr * prior + (1 - prior))
        
        results.append({
            "person_id": candidate_id,
            "clr": clr,
            "posterior": posterior,
            "consistent_loci": consistent,
            "mutated_loci": mutated,
            "inconclusive_loci": missing
        })
    
    # Step 5: Sort by CLR descending and return top 10
    results.sort(key=lambda x: x['clr'], reverse=True)
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
