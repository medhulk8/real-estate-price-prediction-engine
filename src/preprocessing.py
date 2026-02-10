"""
Data Preprocessing Module for Real Estate Price Prediction

This module contains all data transformation logic used in both:
1. Model training (notebooks/analysis.ipynb)
2. Production inference (src/api.py)

CRITICAL: All transformations must be deterministic and use only training set statistics
to prevent data leakage.

Architecture decisions:
- Chronological split (70/15/15) to prevent temporal leakage
- Train-only aggregates for location features (area/project median prices)
- Unseen category handling with fallback to global statistics
- Conservative outlier removal (domain-driven, not percentile-based)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Any


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse INSTANCE_DATE column to datetime.

    Args:
        df: DataFrame with INSTANCE_DATE column

    Returns:
        DataFrame with parsed INSTANCE_DATE
    """
    df = df.copy()
    df['INSTANCE_DATE'] = pd.to_datetime(df['INSTANCE_DATE'])
    return df


def filter_by_procedure(df: pd.DataFrame, keep_procedures: List[str] = None) -> pd.DataFrame:
    """
    Filter dataset by transaction type (PROCEDURE_EN).

    Based on EDA, we typically keep only 'Sale' transactions for market price prediction.
    This removes mortgages, gifts, etc. that don't represent true market values.

    Args:
        df: Input DataFrame
        keep_procedures: List of procedure types to keep. If None, keep all.

    Returns:
        Filtered DataFrame with row count logged
    """
    df = df.copy()
    original_count = len(df)

    if keep_procedures is not None:
        df = df[df['PROCEDURE_EN'].isin(keep_procedures)]
        filtered_count = len(df)
        removed_pct = (1 - filtered_count / original_count) * 100
        print(f"Filtered PROCEDURE_EN: kept {filtered_count:,} / {original_count:,} rows ({removed_pct:.1f}% removed)")

    return df


def remove_data_errors(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove obvious data errors based on domain knowledge.

    Rules (conservative, only remove clear errors):
    1. ACTUAL_AREA <= 0 (impossible)
    2. PROCEDURE_AREA <= 0 (impossible)
    3. TRANS_VALUE <= 0 (impossible for sale prices)
    4. Extreme outliers based on price-per-area ratios (will be defined based on EDA)

    Args:
        df: Input DataFrame
        verbose: Print removal statistics

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    original_count = len(df)

    # Rule 1 & 2: Remove invalid areas
    df = df[(df['ACTUAL_AREA'] > 0) & (df['PROCEDURE_AREA'] > 0)]

    # Rule 3: Remove invalid transaction values
    df = df[df['TRANS_VALUE'] > 0]

    # Rule 4: Remove extreme price-per-area outliers (very conservative)
    # We'll use 0.1th and 99.9th percentiles as thresholds
    df['_temp_price_per_sqft'] = df['TRANS_VALUE'] / df['ACTUAL_AREA']
    p001 = df['_temp_price_per_sqft'].quantile(0.001)
    p999 = df['_temp_price_per_sqft'].quantile(0.999)
    df = df[(df['_temp_price_per_sqft'] >= p001) & (df['_temp_price_per_sqft'] <= p999)]
    df = df.drop(columns=['_temp_price_per_sqft'])

    final_count = len(df)
    removed = original_count - final_count
    removed_pct = (removed / original_count) * 100

    if verbose:
        print(f"Removed {removed:,} obvious data errors ({removed_pct:.2f}%)")
        print(f"Remaining: {final_count:,} rows")

    return df


def chronological_split(
    df: pd.DataFrame,
    date_column: str = 'INSTANCE_DATE',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically to prevent temporal leakage.

    Oldest 70% -> Train
    Next 15% -> Validation
    Newest 15% -> Test

    Args:
        df: Input DataFrame with parsed dates
        date_column: Name of date column to sort by
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)

    Returns:
        train_df, val_df, test_df
    """
    df = df.copy()
    df = df.sort_values(date_column).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nChronological Split:")
    print(f"  Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%) - {train_df[date_column].min()} to {train_df[date_column].max()}")
    print(f"  Val:   {len(val_df):,} rows ({len(val_df)/n*100:.1f}%) - {val_df[date_column].min()} to {val_df[date_column].max()}")
    print(f"  Test:  {len(test_df):,} rows ({len(test_df)/n*100:.1f}%) - {test_df[date_column].min()} to {test_df[date_column].max()}")

    return train_df, val_df, test_df


def handle_missing_values(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    numeric_impute_strategy: str = 'median'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values using training set statistics only.

    Categorical: Fill with "Unknown"
    Numeric: Fill with median (computed on train only)
    Empty strings: Convert to "Unknown"

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        numeric_impute_strategy: 'median' or 'mean'

    Returns:
        train_df, val_df, test_df, imputation_values (dict to store in artifact)
    """
    # Compute imputation values from TRAIN only
    imputation_values = {}

    # Categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        imputation_values[col] = 'Unknown'

    # Numeric columns (exclude target)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'TRANS_VALUE' in numeric_cols:
        numeric_cols.remove('TRANS_VALUE')

    for col in numeric_cols:
        if numeric_impute_strategy == 'median':
            imputation_values[col] = train_df[col].median()
        else:
            imputation_values[col] = train_df[col].mean()

    # Apply imputation to all splits
    def apply_imputation(df):
        df = df.copy()

        # Categorical: empty strings and nulls -> "Unknown"
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].replace('', 'Unknown')
            df[col] = df[col].astype(str)  # Ensure string type

        # Numeric: fill with train statistics
        for col in numeric_cols:
            df[col] = df[col].fillna(imputation_values[col])

        return df

    train_df = apply_imputation(train_df)
    val_df = apply_imputation(val_df) if val_df is not None else None
    test_df = apply_imputation(test_df) if test_df is not None else None

    return train_df, val_df, test_df, imputation_values


def create_time_features(
    df: pd.DataFrame,
    min_train_date: datetime = None,
    max_train_date: datetime = None,
    date_column: str = 'INSTANCE_DATE'
) -> Tuple[pd.DataFrame, datetime, datetime]:
    """
    Create time-based features from INSTANCE_DATE.

    Features created:
    - year, month, quarter (standard datetime components)
    - days_since_start: days since min_train_date (trend proxy)
    - transaction_age_days: days from transaction to max_train_date (recency)

    CRITICAL: min_train_date and max_train_date MUST be computed once on training
    and reused for validation/test/API to prevent leakage.

    Args:
        df: DataFrame with parsed INSTANCE_DATE
        min_train_date: Minimum date in training set (compute if None)
        max_train_date: Maximum date in training set (compute if None)
        date_column: Name of date column

    Returns:
        df with time features, min_train_date, max_train_date
    """
    df = df.copy()

    # Compute reference dates from this dataset if not provided (training mode)
    if min_train_date is None:
        min_train_date = df[date_column].min()
    if max_train_date is None:
        max_train_date = df[date_column].max()

    # Extract standard datetime features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['quarter'] = df[date_column].dt.quarter

    # Create trend/recency features
    df['days_since_start'] = (df[date_column] - min_train_date).dt.days
    df['transaction_age_days'] = (max_train_date - df[date_column]).dt.days

    return df, min_train_date, max_train_date


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio-based features with safe division.

    Features:
    - area_ratio = ACTUAL_AREA / PROCEDURE_AREA
    - rooms_density = ROOMS_EN_numeric / ACTUAL_AREA

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with ratio features
    """
    df = df.copy()

    # Area ratio (safe division)
    df['area_ratio'] = np.where(
        df['PROCEDURE_AREA'] > 0,
        df['ACTUAL_AREA'] / df['PROCEDURE_AREA'],
        1.0  # Default to 1.0 if can't compute
    )

    # Extract numeric part from ROOMS_EN (e.g., "2 B/R" -> 2)
    # Handle various formats: "2 B/R", "Studio", "3", etc.
    def extract_room_number(room_str):
        if pd.isna(room_str) or room_str == '':
            return 0
        room_str = str(room_str).strip()
        # Try to extract the first number
        import re
        match = re.search(r'(\d+)', room_str)
        if match:
            return int(match.group(1))
        # Handle "Studio" case
        if 'studio' in room_str.lower():
            return 0  # or 1, depending on interpretation
        return 0  # Default

    df['ROOMS_EN_numeric'] = df['ROOMS_EN'].apply(extract_room_number)

    # Rooms density (safe division)
    df['rooms_density'] = np.where(
        df['ACTUAL_AREA'] > 0,
        df['ROOMS_EN_numeric'] / df['ACTUAL_AREA'],
        0.0  # Default to 0.0 if can't compute
    )

    return df


def create_aggregate_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    agg_columns: List[str] = None,
    target_col: str = 'TRANS_VALUE'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Create train-only aggregate features (THE SECRET SAUCE for location premium).

    For each categorical column (e.g., AREA_EN, PROJECT_EN):
    - Compute median price and transaction count in TRAINING set only
    - Join these stats to train/val/test
    - For unseen categories in val/test, use global fallback values
    - Add flags: is_unseen_area, is_unseen_project, etc.

    This captures location/project effects without data leakage.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        agg_columns: Columns to aggregate (default: ['AREA_EN', 'PROJECT_EN', 'MASTER_PROJECT_EN'])
        target_col: Target column to aggregate

    Returns:
        train_df, val_df, test_df, aggregate_maps (dict to store in artifact)
    """
    if agg_columns is None:
        agg_columns = ['AREA_EN', 'PROJECT_EN', 'MASTER_PROJECT_EN']

    aggregate_maps = {}
    global_median = train_df[target_col].median()

    for col in agg_columns:
        # Compute aggregates on TRAIN only
        agg_stats = train_df.groupby(col)[target_col].agg([
            ('median_price', 'median'),
            ('txn_count', 'count')
        ]).reset_index()

        agg_stats.columns = [col, f'{col}_median_price', f'{col}_txn_count']
        aggregate_maps[col] = agg_stats

        # Join to train
        train_df = train_df.merge(agg_stats, on=col, how='left')
        train_df[f'is_unseen_{col.lower()}'] = False  # All categories are seen in train

        # Join to val/test with unseen handling
        for df_name, df in [('val', val_df), ('test', test_df)]:
            if df is not None:
                df = df.merge(agg_stats, on=col, how='left')

                # Mark unseen categories
                df[f'is_unseen_{col.lower()}'] = df[f'{col}_median_price'].isna()

                # Fill unseen with global values
                df[f'{col}_median_price'] = df[f'{col}_median_price'].fillna(global_median)
                df[f'{col}_txn_count'] = df[f'{col}_txn_count'].fillna(0)

                # Update the appropriate DataFrame
                if df_name == 'val':
                    val_df = df
                else:
                    test_df = df

    return train_df, val_df, test_df, aggregate_maps


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = 'TRANS_VALUE',
    log_transform_target: bool = True,
    drop_cols: List[str] = None,
    use_price_per_sqm: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare final feature matrix and target vector.

    KEY IMPROVEMENT: Predicts price_per_sqm instead of absolute price to reduce variance.

    Why price_per_sqm works better:
    - Removes size-driven variance (properties from $200k to $4.6M)
    - Model learns value density (location + quality), not size scaling
    - Final price reconstructed as: predicted_price_per_sqm × actual_area
    - Common approach in real estate modeling

    Args:
        df: Input DataFrame with all features
        target_col: Name of target column (default: 'TRANS_VALUE')
        log_transform_target: Whether to apply log1p to target
        drop_cols: Additional columns to drop
        use_price_per_sqm: If True, target is price_per_sqm (default: True)

    Returns:
        X (features), y (target - either price or price_per_sqm in log space)
    """
    if drop_cols is None:
        drop_cols = ['TRANSACTION_NUMBER', 'INSTANCE_DATE']

    df = df.copy()

    # Filter out rows with invalid area (required for price_per_sqm)
    if use_price_per_sqm:
        initial_len = len(df)
        df = df[df['ACTUAL_AREA'] > 0].copy()
        removed = initial_len - len(df)
        if removed > 0:
            print(f"  Removed {removed} rows with ACTUAL_AREA <= 0 ({removed/initial_len*100:.2f}%)")

    # Extract target
    if use_price_per_sqm:
        # Create price per square meter/foot target
        y = (df[target_col] / df['ACTUAL_AREA']).copy()
        print(f"  Using price_per_sqm target (value density approach)")
    else:
        y = df[target_col].copy()
        print(f"  Using absolute price target (traditional approach)")

    if log_transform_target:
        y = np.log1p(y)

    # Drop unnecessary columns
    drop_cols_final = drop_cols + [target_col]

    # Also drop Period columns from EDA (year_month, year_quarter) if they exist
    # These are pandas Period objects that CatBoost can't handle
    period_cols = ['year_month', 'year_quarter', 'price_per_sqft_temp']
    for col in period_cols:
        if col in df.columns and col not in drop_cols_final:
            drop_cols_final.append(col)

    X = df.drop(columns=[c for c in drop_cols_final if c in df.columns])

    return X, y


def get_categorical_feature_indices(X: pd.DataFrame) -> List[int]:
    """
    Get indices of categorical features for CatBoost.

    CatBoost needs to know which features are categorical so it can
    use its native categorical handling (no need for one-hot encoding).

    Args:
        X: Feature DataFrame

    Returns:
        List of column indices that are categorical
    """
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
    return categorical_indices


# ==================== INFERENCE-TIME PREPROCESSING ====================

def preprocess_for_inference(
    input_data: Dict[str, Any],
    preprocessing_metadata: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocess a single input record for API inference.

    This function applies the SAME transformations as training, using
    stored metadata from the artifact to ensure consistency.

    Args:
        input_data: Dictionary with input features from API request
        preprocessing_metadata: Stored metadata from training (dates, medians, aggregates, etc.)

    Returns:
        DataFrame with single row ready for model prediction
    """
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # CRITICAL: Set INSTANCE_DATE to max_train_date (API doesn't receive date)
    df['INSTANCE_DATE'] = preprocessing_metadata['max_train_date']

    # Handle missing values using stored imputation values
    imputation_values = preprocessing_metadata['imputation_values']
    for col, value in imputation_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
            if isinstance(value, str):  # Categorical
                df[col] = df[col].replace('', value)
                df[col] = df[col].astype(str)

    # Create time features using stored min/max dates
    df, _, _ = create_time_features(
        df,
        min_train_date=preprocessing_metadata['min_train_date'],
        max_train_date=preprocessing_metadata['max_train_date']
    )

    # Create ratio features
    df = create_ratio_features(df)

    # Add aggregate features using stored maps
    aggregate_maps = preprocessing_metadata['aggregate_maps']
    global_median = preprocessing_metadata['global_median_price']

    for col, agg_stats in aggregate_maps.items():
        # Merge with aggregate stats
        df = df.merge(agg_stats, on=col, how='left')

        # Mark unseen and fill with global values
        df[f'is_unseen_{col.lower()}'] = df[f'{col}_median_price'].isna()
        df[f'{col}_median_price'] = df[f'{col}_median_price'].fillna(global_median)
        df[f'{col}_txn_count'] = df[f'{col}_txn_count'].fillna(0)

    # Prepare features (drop identifiers)
    X, _ = prepare_features_and_target(df, log_transform_target=False)

    # Ensure column order matches training
    expected_columns = preprocessing_metadata['feature_order']
    X = X[expected_columns]

    return X


# ==================== COMPLETE PREPROCESSING PIPELINE ====================

def run_full_preprocessing_pipeline(
    df: pd.DataFrame,
    keep_procedures: List[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Run the complete preprocessing pipeline for training.

    This is the master function that orchestrates all preprocessing steps
    and returns everything needed for training and inference.

    Steps:
    1. Parse dates
    2. Filter by procedure type
    3. Remove data errors
    4. Chronological split
    5. Handle missing values
    6. Create time features
    7. Create ratio features
    8. Create aggregate features
    9. Prepare final X, y matrices

    Args:
        df: Raw input DataFrame
        keep_procedures: Transaction types to keep
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        Dictionary containing:
        - X_train, y_train, X_val, y_val, X_test, y_test
        - preprocessing_metadata (to store in artifact)
        - categorical_indices (for CatBoost)
    """
    print("=" * 80)
    print("RUNNING FULL PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Parse dates
    print("\n[1/9] Parsing dates...")
    df = parse_dates(df)

    # Step 2: Filter by procedure
    print("\n[2/9] Filtering by transaction type...")
    if keep_procedures is not None:
        df = filter_by_procedure(df, keep_procedures)

    # Step 3: Remove data errors
    print("\n[3/9] Removing data errors...")
    df = remove_data_errors(df, verbose=True)

    # Step 4: Chronological split
    print("\n[4/9] Splitting data chronologically...")
    train_df, val_df, test_df = chronological_split(df, train_ratio=train_ratio, val_ratio=val_ratio)

    # Step 5: Handle missing values (using train statistics)
    print("\n[5/9] Handling missing values...")
    train_df, val_df, test_df, imputation_values = handle_missing_values(train_df, val_df, test_df)

    # Step 6: Create time features (compute min/max from train)
    print("\n[6/9] Creating time features...")
    train_df, min_train_date, max_train_date = create_time_features(train_df)
    val_df, _, _ = create_time_features(val_df, min_train_date, max_train_date)
    test_df, _, _ = create_time_features(test_df, min_train_date, max_train_date)

    # Step 7: Create ratio features
    print("\n[7/9] Creating ratio features...")
    train_df = create_ratio_features(train_df)
    val_df = create_ratio_features(val_df)
    test_df = create_ratio_features(test_df)

    # Step 8: Create aggregate features (train-only stats)
    print("\n[8/9] Creating aggregate features...")
    train_df, val_df, test_df, aggregate_maps = create_aggregate_features(
        train_df, val_df, test_df
    )

    # Step 9: Prepare final feature matrices
    print("\n[9/9] Preparing final feature matrices...")

    # Save actual areas BEFORE prepare_features_and_target (for price reconstruction)
    train_areas = train_df['ACTUAL_AREA'].values
    val_areas = val_df['ACTUAL_AREA'].values
    test_areas = test_df['ACTUAL_AREA'].values

    X_train, y_train = prepare_features_and_target(train_df, log_transform_target=True, use_price_per_sqm=True)
    X_val, y_val = prepare_features_and_target(val_df, log_transform_target=True, use_price_per_sqm=True)
    X_test, y_test = prepare_features_and_target(test_df, log_transform_target=True, use_price_per_sqm=True)

    # Get categorical indices for CatBoost
    categorical_indices = get_categorical_feature_indices(X_train)

    # Store preprocessing metadata for artifact
    preprocessing_metadata = {
        'min_train_date': min_train_date,
        'max_train_date': max_train_date,
        'global_median_price': train_df['TRANS_VALUE'].median(),
        'imputation_values': imputation_values,
        'aggregate_maps': aggregate_maps,
        'feature_order': X_train.columns.tolist(),
        'categorical_indices': categorical_indices,
        'keep_procedures': keep_procedures
    }

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"Val:   {X_val.shape[0]:,} rows × {X_val.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
    print(f"Categorical features: {len(categorical_indices)}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'train_areas': train_areas,
        'val_areas': val_areas,
        'test_areas': test_areas,
        'preprocessing_metadata': preprocessing_metadata,
        'categorical_indices': categorical_indices
    }

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(X_train.columns)}")
    print(f"Categorical features: {len(categorical_indices)}")
    print("=" * 80)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessing_metadata': preprocessing_metadata,
        'categorical_indices': categorical_indices
    }


def preprocess_for_inference(
    input_data: pd.DataFrame,
    preprocessing_metadata: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocess a single input row for inference using saved preprocessing metadata.
    
    This applies the SAME transformations as training but WITHOUT the target column.
    
    Args:
        input_data: DataFrame with one row of input features
        preprocessing_metadata: Saved metadata from training
        
    Returns:
        DataFrame ready for model.predict()
    """
    df = input_data.copy()
    
    # Extract metadata
    max_train_date = preprocessing_metadata['max_train_date']
    imputation_values = preprocessing_metadata['imputation_values']
    aggregate_maps = preprocessing_metadata['aggregate_maps']
    feature_order = preprocessing_metadata['feature_order']
    
    # Set INSTANCE_DATE to max training date (current time approximation)
    # max_train_date is already a Timestamp, no need to convert
    df['INSTANCE_DATE'] = max_train_date

    # Create time features (returns tuple, we only need the DataFrame)
    df, _, _ = create_time_features(df, max_train_date=max_train_date)

    # Apply saved imputation values (don't call handle_missing_values which returns a tuple)
    for col, fill_value in imputation_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
            if isinstance(fill_value, str):  # Categorical column
                df[col] = df[col].replace('', fill_value)
                df[col] = df[col].astype(str)

    # Create ratio features
    df = create_ratio_features(df)
    
    # Apply aggregate features (using saved maps, not computing new ones)
    for col, agg_df in aggregate_maps.items():
        # Convert DataFrame to dict mapping: category -> median_price
        agg_dict = dict(zip(agg_df[col], agg_df[f'{col}_median_price']))

        target_col = f"{col}_median_price"
        # Map values, use global median for unseen categories
        global_median = preprocessing_metadata['global_median_price']
        df[target_col] = df[col].map(agg_dict).fillna(global_median)

        # Flag unseen categories
        unseen_col = f"is_unseen_{col.lower()}"
        df[unseen_col] = (~df[col].isin(agg_dict.keys())).astype(int)
    
    # Ensure all required features exist and are in correct order
    for col in feature_order:
        if col not in df.columns:
            # Add missing column with default value (assume numeric)
            df[col] = 0
    
    # Select only the features used during training, in the same order
    X = df[feature_order]
    
    return X
