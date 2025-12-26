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
    import numpy as np
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

    users = database_df["PersonID"]

    def split_df(_df):
        df = _df.copy()
        df_2 = _df.copy()
        for col in df.columns[1:]:
            df[col] += ","
            df_2[col] = df[col].str.split(",", expand=True)[0].replace("-", np.nan).astype(float)
            df[col] = df[col].str.split(",", expand=True)[1]
            df.loc[df[col] == "", col] = df_2.loc[df[col] == "", col]
            df[col] = df[col].replace("-", np.nan).astype(float)
        df.drop("PersonID", axis=1, inplace=True)
        df_2.drop("PersonID", axis=1, inplace=True)
        return df, df_2
    
    q, q_2 = split_df(pd.DataFrame([query_profile]))

    df, df_2 = split_df(database_df)
    m1 = (df - q.iloc[0]).abs()
    m2 = (df - q_2.iloc[0]).abs()
    m3 = (df_2 - q.iloc[0]).abs()
    m4 = (df_2 - q_2.iloc[0]).abs()

    score = ((m1 == 0) | (m2 == 0) | (m3 == 0) | (m4 == 0)).sum(axis=1) + (
        ((m1 == 1) | (m2 == 1) | (m3 == 1) | (m4 == 1)).sum(axis=1) * 0.002
    )

    return [{
        "person_id": users[index],
        "clr": score[index],
    } for index in np.argsort(score)[::-1][:10]]


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
