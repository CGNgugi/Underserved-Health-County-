"""
UCS Scoring Module - Improved Data Cleaning & Transformation
=============================================================

This module provides enhanced functions for computing the Underserved County Score (UCS)
with improved error handling, metadata return, and flexibility.

Author: Ngugi, Cynthia | Strathmore University
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from typing import Optional


def score_subdomain(
    df: pd.DataFrame,
    indicators: list[str],
    polarity: str,
    name: str = '',
    imputation_strategy: str = 'mean',
    return_metadata: bool = False
) -> np.ndarray | tuple[np.ndarray, dict] | None:
    """
    PCA-weighted sub-domain scorer with robust error handling.
    Returns array where HIGH = more underserved.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with county-level data
    indicators : list[str]
        List of indicator column names to include in subdomain score
    polarity : str
        'positive' = higher indicator values = more underserved
        'negative' = lower indicator values = more underserved
    name : str, optional
        Subdomain name for logging/metadata
    imputation_strategy : str, default 'mean'
        Strategy for handling missing values: 'mean', 'median', or 'zero'
    return_metadata : bool, default False
        If True, returns tuple of (scores, metadata_dict)
    
    Returns
    -------
    np.ndarray or tuple
        Scores array, optionally with metadata dict containing:
        - 'indicators_used': list of valid indicators
        - 'variance_explained': float (only if n_indicators > 1)
        - 'loadings': np.ndarray (PCA loadings, only if n_indicators > 1)
        - 'n_indicators': int
        - 'polarity': str
    
    Method:
    1. Select valid indicators from data
    2. Impute missing values (configurable strategy)
    3. Remove constant columns (zero variance)
    4. StandardScaler (zero mean, unit variance)
    5. PCA(1 component) — captures maximum correlated variance
    6. Weighted score = dot(X_scaled, PC1_loadings) / sum(|loadings|)
    7. Polarity correction: 'negative' indicators inverted
    
    Rationale for PCA weighting over equal weights:
    - KDHS indicators within sub-domains are correlated (e.g. DPT1-DPT2-DPT3)
    - PCA down-weights redundant indicators, up-weights unique variance
    - More defensible than arbitrary equal weights in LMIC evidence synthesis
    
    Improvements over original:
    - Handles constant columns (zero variance) gracefully
    - Validates data after imputation to prevent NaN propagation
    - Returns PCA loadings for interpretability
    - Configurable imputation strategy
    - Type hints for IDE support and documentation
    """
    # 1. Select valid indicators from data
    valid = [c for c in indicators if c in df.columns]
    if not valid:
        print(f"  ⚠  No valid indicators: {name}")
        return None
    
    missing_n = len(indicators) - len(valid)
    if missing_n:
        print(f"  ℹ  {name}: {missing_n} absent → using {len(valid)}")
    
    # 2. Extract and impute missing values
    sub = df[valid].copy()
    
    if imputation_strategy == 'mean':
        fill_value = sub.mean()
    elif imputation_strategy == 'median':
        fill_value = sub.median()
    elif imputation_strategy == 'zero':
        fill_value = 0
    else:
        raise ValueError(f"Unknown imputation_strategy: {imputation_strategy}")
    
    sub = sub.fillna(fill_value)
    
    # 3. Check for constant columns (zero variance) and remove them
    variances = sub.var()
    non_constant_cols = variances[variances > 1e-10].index.tolist()
    
    if len(non_constant_cols) < len(valid):
        removed = set(valid) - set(non_constant_cols)
        print(f"  ⚠  {name}: {len(removed)} constant columns removed: {removed}")
        valid = non_constant_cols
        sub = sub[valid]
    
    if len(valid) == 0:
        print(f"  ⚠  No non-constant indicators: {name}")
        return None
    
    # 4. Final validation - check for remaining NaNs
    if sub.isna().any().any():
        print(f"  ⚠  {name}: NaNs remain after imputation, filling with 0")
        sub = sub.fillna(0)
    
    # 5. Standardize
    X_s = StandardScaler().fit_transform(sub)
    
    # 6. PCA or direct scoring
    metadata = {
        'indicators_used': valid,
        'n_indicators': len(valid),
        'polarity': polarity
    }
    
    if len(valid) == 1:
        # Single indicator: use standardized value directly
        score = X_s[:, 0]
        # For single indicator, loadings is just [1.0]
        loadings = np.array([1.0])
        metadata['variance_explained'] = 1.0  # 100% by definition
        metadata['loadings'] = loadings
        metadata['method'] = 'direct_standardized'
    else:
        # Multiple indicators: use PCA
        pca = PCA(n_components=1)
        pca.fit(X_s)
        loadings = pca.components_[0]
        variance_explained = pca.explained_variance_ratio_[0]
        
        # Check for potential issues
        if variance_explained < 0.5:
            print(f"  ℹ  {name}: Low variance explained ({variance_explained:.1%}), indicators may be uncorrelated")
        
        # Weighted score: dot product normalized by sum of absolute loadings
        score = np.dot(X_s, loadings) / np.sum(np.abs(loadings))
        
        metadata['variance_explained'] = variance_explained
        metadata['loadings'] = loadings
        metadata['method'] = 'pca_weighted'
    
    # 7. Apply polarity correction
    if polarity == 'negative':
        score = -score
        # Also negate loadings for interpretability
        metadata['loadings'] = -metadata['loadings']
    
    if return_metadata:
        return score, metadata
    return score


def compute_composite_scores(
    pivot_df: pd.DataFrame,
    INDICATORS: dict,
    DOMAIN_NAMES: list[str],
    domain_weights: Optional[dict] = None,
    weighting_method: str = 'weighted_variance',  # 'weighted_variance', 'variance', 'cv', or 'equal'
    imputation_strategy: str = 'mean',
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute composite domain scores and UCS from subdomain indicators.
    
    Parameters
    ----------
    pivot_df : pd.DataFrame
        Input dataframe with county-level indicator data
    INDICATORS : dict
        Nested dict of {domain: {subdomain: {polarity, description, indicators}}}
    DOMAIN_NAMES : list[str]
        List of domain names to include in UCS
    domain_weights : dict | None, optional
        Optional custom weights for each domain. If provided, overrides weighting_method.
    weighting_method : str, default 'weighted_variance'
        Method for calculating domain weights when domain_weights is None:
        - 'weighted_variance': Weight by coefficient of variation (CV = std/mean)
          Higher CV = more relative variability = more discriminative
        - 'variance': Weight by raw variance (higher variance = more discriminative)
        - 'cv': Alias for weighted_variance
        - 'equal': Equal weights for all domains
    imputation_strategy : str, default 'mean'
        Strategy for handling missing values
    verbose : bool, default True
        If True, prints progress messages
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        - composite_scores: DataFrame with domain scores and UCS
        - subindex_scores_df: DataFrame with all subdomain scores
        - subdomain_meta: Dict mapping subdomain names to descriptions
    
    Notes on weighting methods:
    ---------------------------
    1. Weighted Variance (CV):
       - CV = standard deviation / mean
       - Captures RELATIVE variability (normalized by domain level)
       - Best for domains with different measurement scales
       - Higher CV = more dispersion relative to the mean
       
    2. Raw Variance:
       - Variance = standard deviation^2
       - Captures ABSOLUTE variability
       - May overweight domains with larger absolute values
       
    3. Equal Weights:
       - All domains weighted equally
       - Simple but may not reflect domain importance
    """
    # Initialize output DataFrames
    composite_scores = pd.DataFrame(index=pivot_df.index)
    subindex_scores_df = pd.DataFrame(index=pivot_df.index)
    subdomain_meta = {}  # store descriptions for dashboard
    subdomain_metadata = {}  # store detailed metadata for each subdomain
    
    # Set default equal weights if not provided
    if domain_weights is None:
        domain_weights = {}
    
    # Process each domain
    for domain, subdomains in INDICATORS.items():
        if verbose:
            print(f"\n▶ {domain}")
        
        domain_subs = []
        sub_names = []
        
        for sub_name, cfg in subdomains.items():
            result = score_subdomain(
                pivot_df, 
                cfg['indicators'], 
                cfg['polarity'], 
                sub_name,
                imputation_strategy=imputation_strategy,
                return_metadata=True
            )
            
            if result is not None:
                s, meta = result
                col = f"{domain} | {sub_name}"
                subindex_scores_df[col] = s
                
                domain_subs.append(s)
                sub_names.append(col)
                
                subdomain_meta[col] = cfg['description']
                subdomain_metadata[col] = {
                    'description': cfg['description'],
                    'polarity': cfg['polarity'],
                    'n_indicators': meta['n_indicators'],
                    'indicators_used': meta['indicators_used'],
                    'variance_explained': meta.get('variance_explained'),
                    'method': meta.get('method'),
                    'loadings': meta.get('loadings')
                }
                
                if verbose:
                    var_exp = meta.get('variance_explained', 'N/A')
                    if var_exp != 'N/A':
                        var_exp = f"{var_exp:.1%}"
                    print(f"  ✓ {sub_name} [{cfg['polarity']}] (var explained: {var_exp})")
            else:
                if verbose:
                    print(f"  ✗ {sub_name} [failed]")
        
        # Aggregate subdomains to domain score
        if domain_subs:
            # Use equal weights for subdomains within each domain
            composite_scores[domain] = np.mean(domain_subs, axis=0)
    
    # Get valid domains that exist in composite_scores
    valid_domains = [d for d in DOMAIN_NAMES if d in composite_scores.columns]
    
    if not valid_domains:
        raise ValueError("No valid domains found in composite scores")
    
    # Apply domain weights if provided, otherwise calculate weights based on weighting_method
    if domain_weights:
        # Normalize weights
        total_weight = sum(domain_weights.get(d, 1.0) for d in valid_domains)
        weighted_sum = sum(
            composite_scores[d] * (domain_weights.get(d, 1.0) / total_weight)
            for d in valid_domains
        )
        composite_scores['UCS_raw'] = weighted_sum
        if verbose:
            print(f"\n  ℹ  Using custom domain weights")
    elif weighting_method in ['weighted_variance', 'cv']:
        # Weighted Variance (Coefficient of Variation): CV = std / mean
        # This normalizes variance by the domain's mean level
        # Higher CV = more relative variability = more discriminative between counties
        domain_means = composite_scores[valid_domains].mean()
        domain_stds = composite_scores[valid_domains].std()
        
        # Calculate CV (handle zero means to avoid division by zero)
        cv_values = domain_stds / domain_means.replace(0, np.nan)
        cv_values = cv_values.fillna(0)
        
        # Handle any remaining NaN or inf values
        cv_values = cv_values.replace([np.inf, -np.inf], 0)
        
        # Normalize to get weights
        cv_weights = cv_values / cv_values.sum()
        
        if verbose:
            print(f"\n  ℹ  Using weighted variance (CV) domain weights:")
            for d in valid_domains:
                cv = cv_values[d]
                weight = cv_weights[d]
                print(f"      {d.replace(' Index', '')}: {weight:.3f} (CV={cv:.3f}, μ={domain_means[d]:.3f}, σ={domain_stds[d]:.3f})")
        
        # Weighted UCS = sum of (domain_score * CV_weight)
        composite_scores['UCS_raw'] = (composite_scores[valid_domains] * cv_weights).sum(axis=1)
    elif weighting_method == 'variance':
        # Variance-based weights: weight by domain variance (higher variance = more discriminative)
        domain_variances = composite_scores[valid_domains].var()
        variance_weights = domain_variances / domain_variances.sum()
        
        if verbose:
            print(f"\n  ℹ  Using variance-based domain weights:")
            for d in valid_domains:
                print(f"      {d.replace(' Index', '')}: {variance_weights[d]:.3f} (var={domain_variances[d]:.3f})")
        
        # Weighted UCS = sum of (domain_score * weight)
        composite_scores['UCS_raw'] = (composite_scores[valid_domains] * variance_weights).sum(axis=1)
    else:
        # Equal weights (original method)
        composite_scores['UCS_raw'] = composite_scores[valid_domains].mean(axis=1)
        if verbose:
            print(f"\n  ℹ  Using equal domain weights")
    
    # Min-Max normalize to 0-100
    ucs_values = composite_scores['UCS_raw'].values.reshape(-1, 1)
    composite_scores['UCS'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(ucs_values).flatten()
    
    # Add summary statistics
    composite_scores['UCS_rank'] = composite_scores['UCS'].rank(ascending=False).astype(int)
    composite_scores['UCS_percentile'] = composite_scores['UCS'].rank(pct=True) * 100
    
    # Summary output
    if verbose:
        print(f"\n{'='*55}")
        print(f"✓ UCS computed for {len(composite_scores)} counties")
        print(f"  Domains included: {valid_domains}")
        # Show which weighting was used
        if domain_weights:
            print(f"  Weighting: Custom weights")
        elif weighting_method in ['weighted_variance', 'cv']:
            print(f"  Weighting: Weighted Variance (CV)")
        elif weighting_method == 'variance':
            print(f"  Weighting: Variance-based")
        else:
            print(f"  Weighting: Equal")
        print(f"  UCS range: {composite_scores['UCS'].min():.1f} – {composite_scores['UCS'].max():.1f}")
        print(f"  UCS mean: {composite_scores['UCS'].mean():.1f}, std: {composite_scores['UCS'].std():.1f}")
        print(f"\n  Top 5 underserved counties:")
        top5 = composite_scores['UCS'].sort_values(ascending=False).head(5)
        for county, ucs in top5.items():
            rank = composite_scores.loc[county, 'UCS_rank']
            print(f"    {county}: {ucs:.1f} (rank #{int(rank)})")
    
    return composite_scores, subindex_scores_df, subdomain_meta


# Legacy wrapper for backward compatibility
def score_subdomain_legacy(df, indicators, polarity, name=''):
    """
    Legacy version of score_subdomain - wraps new function for backward compatibility.
    """
    return score_subdomain(df, indicators, polarity, name, return_metadata=False)


def compare_weighting_methods(
    pivot_df: pd.DataFrame,
    INDICATORS: dict,
    DOMAIN_NAMES: list[str],
    imputation_strategy: str = 'mean'
) -> pd.DataFrame:
    """
    Compare UCS scores computed with different weighting methods.
    
    This is useful for sensitivity analysis and understanding how 
    different weighting approaches affect the final scores.
    
    Parameters
    ----------
    pivot_df : pd.DataFrame
        Input dataframe with county-level indicator data
    INDICATORS : dict
        Nested dict of {domain: {subdomain: {polarity, description, indicators}}}
    DOMAIN_NAMES : list[str]
        List of domain names to include in UCS
    imputation_strategy : str, default 'mean'
        Strategy for handling missing values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with UCS scores from each weighting method and their correlations
    
    Example
    --------
    >>> results = compare_weighting_methods(pivot_df, INDICATORS, DOMAIN_NAMES)
    >>> print(results.corr())  # Correlation between methods
    """
    methods = ['weighted_variance', 'variance', 'equal']
    results = {}
    
    print("Comparing weighting methods...")
    print("=" * 50)
    
    for method in methods:
        try:
            composite_scores, _, _ = compute_composite_scores(
                pivot_df,
                INDICATORS,
                DOMAIN_NAMES,
                weighting_method=method,
                imputation_strategy=imputation_strategy,
                verbose=False
            )
            results[method] = composite_scores['UCS']
            print(f"  ✓ {method}: UCS range [{composite_scores['UCS'].min():.1f}, {composite_scores['UCS'].max():.1f}]")
        except Exception as e:
            print(f"  ✗ {method}: {str(e)}")
    
    if not results:
        raise ValueError("No weighting methods produced valid results")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Calculate correlations
    print("\n" + "=" * 50)
    print("Correlation between methods:")
    print(comparison_df.corr().round(3))
    
    # Calculate weight comparisons
    print("\n" + "=" * 50)
    print("Domain weight comparisons:")
    
    # Get weights for each method
    _, _, _ = compute_composite_scores(
        pivot_df,
        INDICATORS,
        DOMAIN_NAMES,
        weighting_method='weighted_variance',
        imputation_strategy=imputation_strategy,
        verbose=True
    )
    
    return comparison_df
