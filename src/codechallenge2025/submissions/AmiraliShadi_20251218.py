# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# We use a global/static cache approach to avoid re-processing
# the 500k database rows for every single query.
_DB_CACHE = {}


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.

    Implements a vectorized NumPy approach for high-performance forensic matching.
    Includes logic for:
    1. Allele sharing (Mendelian inheritance).
    2. Stepwise Mutations (±1 repeat).
    3. Null/Dropout handling.
    4. Likelihood Ratio (CLR) calculation based on population frequencies.
    """

    # ---------------------------------------------------------
    # 1. DATABASE PRE-PROCESSING (Run Once & Cache)
    # ---------------------------------------------------------
    # The database object ID is used as a cache key. If the dataframe
    # matches what we saw last time, we skip parsing.
    db_id = id(database_df)

    if _DB_CACHE.get("id") != db_id:
        # Identify Locus Columns (exclude ID)
        loci_cols = [c for c in database_df.columns if c != "PersonID"]

        # Initialize 3D Array: (Rows, Loci, 2_Alleles)
        # We use floats to handle microvariants (9.3) and NaNs.
        n_rows = len(database_df)
        n_loci = len(loci_cols)
        db_tensor = np.full((n_rows, n_loci, 2), np.nan, dtype=np.float32)

        # Frequency dictionary for CLR calculation
        # shape: {locus_index: {allele_value: frequency}}
        allele_freqs = []

        print(f"  [One-time setup] Vectorizing database ({n_rows} profiles)...")

        for i, col in enumerate(loci_cols):
            # Extract column data as strings
            series = database_df[col].astype(str).str.strip()

            # Handle Dropouts ("-" or empty) -> NaN
            series = series.replace({"-": np.nan, "": np.nan, "nan": np.nan})

            # Split "13,14" into two columns.
            # Optimized approach using pandas vectorized split
            split_df = series.str.split(",", expand=True)

            # Convert to numeric
            a1 = pd.to_numeric(split_df[0], errors="coerce").values

            if split_df.shape[1] > 1:
                a2 = pd.to_numeric(split_df[1], errors="coerce").values
                # Fill missing second allele with first for Homozygotes IF original wasn't NaN
                # logic: if a1 is valid but a2 is NaN, assume homozygous (e.g. "13" -> 13,13)
                mask_homo = ~np.isnan(a1) & np.isnan(a2)
                a2[mask_homo] = a1[mask_homo]
            else:
                # Column only had single values
                a2 = a1.copy()

            db_tensor[:, i, 0] = a1
            db_tensor[:, i, 1] = a2

            # Calculate Frequencies for this locus (ignoring NaNs)
            # Combine both alleles to get counts for statistics
            all_alleles = np.concatenate([a1[~np.isnan(a1)], a2[~np.isnan(a2)]])
            vals, counts = np.unique(all_alleles, return_counts=True)
            total_alleles = counts.sum() if counts.sum() > 0 else 1
            # Default min freq 0.001 to avoid div/0 or huge LRs for rare alleles
            freq_map = {k: max(v / total_alleles, 0.001) for k, v in zip(vals, counts)}
            allele_freqs.append(freq_map)

        # Store in cache
        _DB_CACHE["id"] = db_id
        _DB_CACHE["tensor"] = db_tensor
        _DB_CACHE["loci"] = loci_cols
        _DB_CACHE["ids"] = database_df["PersonID"].values
        _DB_CACHE["freqs"] = allele_freqs
        print("  [One-time setup] Complete.")

    # Retrieve from cache
    db_tensor = _DB_CACHE["tensor"]  # Shape: (N, L, 2)
    loci_cols = _DB_CACHE["loci"]
    db_ids = _DB_CACHE["ids"]
    allele_freqs = _DB_CACHE["freqs"]

    # ---------------------------------------------------------
    # 2. QUERY PARSING
    # ---------------------------------------------------------
    # Parse query into (L, 2) array
    n_loci = len(loci_cols)
    query_matrix = np.full((n_loci, 2), np.nan, dtype=np.float32)

    for i, col in enumerate(loci_cols):
        val = str(query_profile.get(col, "-")).strip()
        if val in ["-", "", "nan", "None"]:
            continue
        parts = val.split(",")
        try:
            v1 = float(parts[0])
            v2 = float(parts[1]) if len(parts) > 1 else v1  # Assume homo if 1 val
            query_matrix[i, 0] = v1
            query_matrix[i, 1] = v2
        except ValueError:
            pass  # Keep as NaNs

    # ---------------------------------------------------------
    # 3. VECTORIZED MATCHING LOGIC
    # ---------------------------------------------------------

    # DB: (N, L, 2)
    # Q:  (1, L, 2)

    # Extract columns for cleaner syntax
    # Shape: (N, L)
    db_a1 = db_tensor[:, :, 0]
    db_a2 = db_tensor[:, :, 1]

    # Shape: (1, L) - reshaped for broadcasting
    q_a1 = query_matrix[:, 0].reshape(1, n_loci)
    q_a2 = query_matrix[:, 1].reshape(1, n_loci)

    # Boolean Masks for EXACT MATCHES
    # (DB_1 == Q_1) | (DB_1 == Q_2) | (DB_2 == Q_1) | (DB_2 == Q_2)
    m1 = np.abs(db_a1 - q_a1) < 0.001
    m2 = np.abs(db_a1 - q_a2) < 0.001
    m3 = np.abs(db_a2 - q_a1) < 0.001
    m4 = np.abs(db_a2 - q_a2) < 0.001

    has_shared_allele = m1 | m2 | m3 | m4

    # Handle MISSING DATA (Inconclusive)
    db_missing = np.isnan(db_a1)  # If first allele is NaN, locus is missing
    q_missing = np.isnan(q_a1)
    is_inconclusive = db_missing | q_missing

    # MUTATION HANDLING
    # If not shared, check if within +/- 1 step (allow small float margin)
    def is_mut(arr1, arr2):
        diff = np.abs(arr1 - arr2)
        return (diff > 0.9) & (diff < 1.1)

    mut_1 = is_mut(db_a1, q_a1)
    mut_2 = is_mut(db_a1, q_a2)
    mut_3 = is_mut(db_a2, q_a1)
    mut_4 = is_mut(db_a2, q_a2)

    has_mutation = (
        (mut_1 | mut_2 | mut_3 | mut_4) & (~has_shared_allele) & (~is_inconclusive)
    )

    # EXCLUSION
    is_excluded = (~has_shared_allele) & (~has_mutation) & (~is_inconclusive)

    # ---------------------------------------------------------
    # 4. SCORING & FILTERING
    # ---------------------------------------------------------

    # Count exclusions per person
    exclusion_counts = is_excluded.sum(axis=1)

    # Filter candidates: Allow max 2 exclusions
    candidate_indices = np.where(exclusion_counts <= 2)[0]

    results = []
    query_id = query_profile["PersonID"]

    for idx in candidate_indices:
        pid = db_ids[idx]
        if pid == query_id:
            continue  # Skip self match

        # Get boolean vectors for this candidate
        shared_vec = has_shared_allele[idx]
        mut_vec = has_mutation[idx]
        inc_vec = is_inconclusive[idx]

        # Calculate CLR (Combined Likelihood Ratio)
        log_clr = 0.0

        for loc_i in range(n_loci):
            if inc_vec[loc_i]:
                continue

            if shared_vec[loc_i]:
                # Find WHICH allele matched to get frequency
                q_1, q_2 = query_matrix[loc_i]
                d_1, d_2 = db_tensor[idx, loc_i]

                # Identify the shared value(s)
                shared_vals = []
                if abs(q_1 - d_1) < 0.01 or abs(q_1 - d_2) < 0.01:
                    shared_vals.append(q_1)
                if abs(q_2 - d_1) < 0.01 or abs(q_2 - d_2) < 0.01:
                    shared_vals.append(q_2)

                if not shared_vals:
                    continue

                # Get freq of the shared allele
                freq_map = allele_freqs[loc_i]
                # Conservative: use the highest frequency among shared alleles
                f = min([freq_map.get(v, 0.05) for v in shared_vals])

                # Paternity Index: 1 / (2 * f)
                pi = 1.0 / (2.0 * f)
                log_clr += np.log10(pi)

            elif mut_vec[loc_i]:
                # Mutation penalty: log10(0.002) approx -2.7
                log_clr += -2.7

            else:
                # Exclusion penalty (severe)
                log_clr += -5.0

        # Convert back to standard number
        clr_val = 10**log_clr

        # Posterior Probability (assuming Prior = 0.5)
        # Prob = CLR / (CLR + 1)
        posterior = clr_val / (clr_val + 1) if clr_val > 0 else 0

        results.append(
            {
                "person_id": str(pid),
                "clr": float(clr_val),
                "posterior": float(posterior),
                "consistent_loci": int(shared_vec.sum()),
                "mutated_loci": int(mut_vec.sum()),
                "inconclusive_loci": int(inc_vec.sum()),
            }
        )

    # Sort by CLR descending and take top 10
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
