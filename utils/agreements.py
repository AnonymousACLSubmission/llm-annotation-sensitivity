import numpy as np
import pandas as pd
import itertools
from utils.general import mode_value
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.inter_rater import cohens_kappa
from statsmodels.stats.inter_rater import to_table


# Compute Fleiss' Kappa per condition using the raw ratings from df_long
def within_agreement_fleiss_kappa(df, condition, rating_column='HS'):
    # Filter raw ratings for the given condition.
    sub = df[df['condition'] == condition]
    # Get unique rating categories (0 and 1)
    categories = sorted(df[rating_column].dropna().unique())
    # Pivot the data so rows are tweet_ids and columns are rating categories
    pivot = sub.pivot_table(index='tweet_id', 
                            columns=rating_column, 
                            aggfunc='size', 
                            fill_value=0)
    # Ensure that every expected category appears in the pivot table
    for cat in categories:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[categories]  # re-order columns in ascending order
    M = pivot.to_numpy()
    if M.shape[0] == 0:
        return np.nan
    # Use the prepackaged fleiss_kappa from statsmodels
    return fleiss_kappa(M)

# Compute across-condition agreement using Cohen's Kappa on modal ratings.
def between_agreement_cohens_kappa(agg_df, cond1, cond2, measure):
    # "measure" should be either 'HS' or 'OL'.
    col_modal = f'modal_{measure}'
    # Extract modal ratings for each condition from the aggregated DataFrame.
    df1 = agg_df[agg_df['condition'] == cond1][['tweet_id', col_modal]].rename(
        columns={col_modal: f'{col_modal}_{cond1}'}
    )
    df2 = agg_df[agg_df['condition'] == cond2][['tweet_id', col_modal]].rename(
        columns={col_modal: f'{col_modal}_{cond2}'}
    )
    # Merge on tweet_id 
    # df with 2 cols: modal rating cond1, modal rating cond2 (tweet_id dropped)
    merged = pd.merge(df1, df2, on='tweet_id').drop('tweet_id', axis = 1)
    # format for statsmodels.kappa
    merged_formated = to_table(merged.copy())[0] 
    if not np.any(merged_formated):
        return np.nan
    # Compute Cohen's kappa between the two sets of modal ratings.
    kappa_results = cohens_kappa(merged_formated)
    kappa = kappa_results['kappa']
    return kappa