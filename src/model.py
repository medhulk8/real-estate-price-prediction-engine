"""
Model Training Module for Real Estate Price Prediction

This module handles:
1. CatBoost model training with optimal hyperparameters
2. Model evaluation metrics (R², MAE, MAPE)
3. Confidence interval computation from validation residuals
4. Model artifact serialization

Key Technical Decisions:
- Model: CatBoost (native categorical handling, robust to outliers)
- Loss: MAE (robust to price outliers, more stable than MSE)
- Target: log1p(TRANS_VALUE) (handles right-skewed distribution)
- Early stopping on validation set
- Fixed random seed for reproducibility
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_indices: List[int],
    hyperparameters: Dict[str, Any] = None,
    random_seed: int = 42
) -> CatBoostRegressor:
    """
    Train CatBoost regression model with MAE loss.

    CatBoost advantages:
    - Handles categorical features natively (no one-hot encoding needed)
    - Built-in handling for unseen categories
    - Robust to outliers (especially with MAE loss)
    - Fast inference for production API

    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_val: Validation features
        y_val: Validation target (log-transformed)
        categorical_indices: List of categorical feature column indices
        hyperparameters: Custom hyperparameters (if None, use defaults)
        random_seed: Random seed for reproducibility

    Returns:
        Trained CatBoostRegressor model
    """
    # Default hyperparameters (light tuning)
    if hyperparameters is None:
        hyperparameters = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'random_seed': random_seed,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'use_best_model': True
        }

    print("=" * 80)
    print("TRAINING CATBOOST MODEL")
    print("=" * 80)
    print(f"\nHyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print(f"\nCategorical features: {len(categorical_indices)} columns")

    # Create CatBoost Pool objects (efficient data structure)
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categorical_indices
    )

    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=categorical_indices
    )

    # Initialize and train model
    model = CatBoostRegressor(**hyperparameters)

    print("\n" + "=" * 80)
    print("Training in progress...")
    print("=" * 80)

    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=hyperparameters.get('verbose', 100)
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Best validation MAE (log-space): {model.best_score_['validation']['MAE']:.4f}")

    return model


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Dataset",
    safe_mape_threshold: float = 1.0
) -> Dict[str, float]:
    """
    Compute regression metrics in ORIGINAL price space.

    Metrics:
    - R² Score: Proportion of variance explained (higher is better, max 1.0)
    - MAE: Mean Absolute Error in currency units (lower is better)
    - MAPE: Mean Absolute Percentage Error (lower is better, interpret carefully)

    MAPE Safety: Skip rows where y_true <= threshold to avoid division issues.

    Args:
        y_true: True values (original price space, NOT log)
        y_pred: Predicted values (original price space, NOT log)
        dataset_name: Name for printing (e.g., "Validation", "Test")
        safe_mape_threshold: Minimum y_true value for MAPE computation

    Returns:
        Dictionary with metrics
    """
    # R² Score
    r2 = r2_score(y_true, y_pred)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (safe computation)
    valid_mask = y_true > safe_mape_threshold
    if valid_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[valid_mask], y_pred[valid_mask]) * 100
        mape_excluded = len(y_true) - valid_mask.sum()
    else:
        mape = np.nan
        mape_excluded = len(y_true)

    metrics = {
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'mape_excluded_count': mape_excluded
    }

    # Print results
    print(f"\n{'=' * 80}")
    print(f"{dataset_name.upper()} METRICS (Original Price Space)")
    print(f"{'=' * 80}")
    print(f"R² Score:  {r2:.4f}")
    print(f"MAE:       {mae:,.2f}")
    if not np.isnan(mape):
        print(f"MAPE:      {mape:.2f}% (excluded {mape_excluded} rows with y_true <= {safe_mape_threshold})")
    else:
        print(f"MAPE:      Not computable (all values <= {safe_mape_threshold})")

    return metrics


def evaluate_model(
    model: CatBoostRegressor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on train, validation, and test sets.

    Predictions are made in log-space, then inverse-transformed to original space
    for metric computation.

    Args:
        model: Trained CatBoost model
        X_train, y_train: Training data (y in log-space)
        X_val, y_val: Validation data (y in log-space)
        X_test, y_test: Test data (y in log-space)

    Returns:
        Dictionary with metrics for each dataset
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)

    results = {}

    # Train set
    y_train_pred_log = model.predict(X_train)
    y_train_true = np.expm1(y_train)  # Inverse log1p transform
    y_train_pred = np.expm1(y_train_pred_log)
    results['train'] = compute_metrics(y_train_true, y_train_pred, "Train")

    # Validation set
    y_val_pred_log = model.predict(X_val)
    y_val_true = np.expm1(y_val)
    y_val_pred = np.expm1(y_val_pred_log)
    results['val'] = compute_metrics(y_val_true, y_val_pred, "Validation")

    # Test set
    y_test_pred_log = model.predict(X_test)
    y_test_true = np.expm1(y_test)
    y_test_pred = np.expm1(y_test_pred_log)
    results['test'] = compute_metrics(y_test_true, y_test_pred, "Test")

    return results


def compute_confidence_intervals(
    model: CatBoostRegressor,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_quantiles: int = 3,
    quantile_levels: Tuple[float, float] = (0.05, 0.95)
) -> Dict[str, Any]:
    """
    Compute empirical confidence intervals from validation residuals.

    Strategy:
    1. Predict on validation set (held-out data)
    2. Bucket predictions into quantile bins (e.g., low/mid/high price)
    3. For each bin, compute residual quantiles (5th and 95th percentile)
    4. At inference: use predicted price to select bin, apply bin's CI

    This gives prediction-dependent confidence intervals:
    - Low-price predictions: tighter CI
    - High-price predictions: wider CI (more uncertainty)

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target (log-space)
        n_quantiles: Number of price buckets (default 3: low/mid/high)
        quantile_levels: Lower and upper quantile for CI (default 5th and 95th)

    Returns:
        Dictionary with CI statistics by bucket
    """
    print("\n" + "=" * 80)
    print("COMPUTING CONFIDENCE INTERVALS")
    print("=" * 80)

    # Predict on validation (original price space)
    y_val_pred_log = model.predict(X_val)
    y_val_true = np.expm1(y_val)
    y_val_pred = np.expm1(y_val_pred_log)

    # Compute absolute residuals
    residuals = np.abs(y_val_true - y_val_pred)

    # Create prediction quantile bins
    quantile_edges = [0] + [i / n_quantiles for i in range(1, n_quantiles)] + [1]
    pred_quantiles = pd.qcut(y_val_pred, q=n_quantiles, labels=False, duplicates='drop')

    # Compute residual quantiles for each prediction bucket
    ci_stats = {
        'bucket_edges': [],
        'buckets': []
    }

    print(f"\nUsing {n_quantiles} prediction buckets for CI estimation:")
    print(f"Quantile levels: {quantile_levels[0]*100:.0f}th and {quantile_levels[1]*100:.0f}th percentile\n")

    for bucket_id in range(n_quantiles):
        bucket_mask = (pred_quantiles == bucket_id)
        bucket_residuals = residuals[bucket_mask]
        bucket_preds = y_val_pred[bucket_mask]

        if len(bucket_residuals) > 0:
            # Compute residual quantiles for this bucket
            lower_error = np.quantile(bucket_residuals, quantile_levels[0])
            upper_error = np.quantile(bucket_residuals, quantile_levels[1])

            bucket_min = bucket_preds.min()
            bucket_max = bucket_preds.max()

            ci_stats['bucket_edges'].append((bucket_min, bucket_max))
            ci_stats['buckets'].append({
                'bucket_id': bucket_id,
                'pred_range': (bucket_min, bucket_max),
                'n_samples': len(bucket_residuals),
                'lower_error': lower_error,
                'upper_error': upper_error,
                'median_error': np.median(bucket_residuals)
            })

            print(f"Bucket {bucket_id} (n={len(bucket_residuals):,}):")
            print(f"  Prediction range: {bucket_min:,.0f} - {bucket_max:,.0f}")
            print(f"  Error (5th percentile): ±{lower_error:,.0f}")
            print(f"  Error (95th percentile): ±{upper_error:,.0f}")
            print(f"  Median error: {np.median(bucket_residuals):,.0f}\n")

    print("=" * 80)
    print("CI computation complete. These will be used for API predictions.")
    print("=" * 80)

    return ci_stats


def get_feature_importance(
    model: CatBoostRegressor,
    X_train: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from trained CatBoost model.

    CatBoost provides two types of importance:
    - PredictionValuesChange: How much predictions change when feature is excluded
    - LossFunctionChange: How much loss increases when feature is excluded

    We use PredictionValuesChange as default (more intuitive).

    Args:
        model: Trained CatBoost model
        X_train: Training features (for column names)
        top_n: Number of top features to return

    Returns:
        DataFrame with features sorted by importance
    """
    importance_values = model.get_feature_importance()
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)

    return importance_df.head(top_n)


def save_model_artifact(
    model: CatBoostRegressor,
    preprocessing_metadata: Dict[str, Any],
    ci_stats: Dict[str, Any],
    feature_importance: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    save_path: str,
    area_unit: str = "sqm"
) -> None:
    """
    Save complete model artifact for production deployment.

    The artifact contains EVERYTHING needed for inference:
    - Trained CatBoost model
    - Preprocessing metadata (dates, medians, aggregates, feature order)
    - Confidence interval statistics
    - Feature importance (optional, for API key_factors)
    - Performance metrics (for monitoring)
    - Area unit (sqm or sqft)

    Args:
        model: Trained CatBoost model
        preprocessing_metadata: From preprocessing pipeline
        ci_stats: Confidence interval statistics
        feature_importance: Feature importance DataFrame
        metrics: Model performance metrics
        save_path: Path to save artifact (e.g., 'models/trained_model.pkl')
        area_unit: Unit for area measurements ('sqm' or 'sqft')
    """
    # Save CatBoost model separately to avoid serialization issues
    import os
    model_dir = os.path.dirname(save_path)
    model_filename = os.path.splitext(os.path.basename(save_path))[0]
    catboost_path = os.path.join(model_dir, f"{model_filename}.cbm")

    # Save CatBoost model in its native format
    model.save_model(catboost_path)

    # Create artifact WITHOUT the model (we'll store the path instead)
    artifact = {
        'model_path': catboost_path,  # Path to CatBoost model file
        'preprocessing_metadata': preprocessing_metadata,
        'ci_stats': ci_stats,
        'feature_importance': feature_importance,
        'metrics': metrics,
        'area_unit': area_unit,
        'model_version': '1.0',
        'trained_at': pd.Timestamp.now()
    }

    # Use pickle protocol 4 for Python 3.8-3.13 compatibility
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(artifact, f, protocol=4)

    print("\n" + "=" * 80)
    print("MODEL ARTIFACT SAVED")
    print("=" * 80)
    print(f"Metadata file: {save_path}")
    metadata_size_mb = os.path.getsize(save_path) / (1024**2)
    print(f"  Size: {metadata_size_mb:.2f} MB")
    print(f"CatBoost model: {catboost_path}")
    catboost_size_mb = os.path.getsize(catboost_path) / (1024**2)
    print(f"  Size: {catboost_size_mb:.2f} MB")
    print(f"\nArtifact contents:")
    print(f"  - Trained CatBoost model (saved separately for compatibility)")
    print(f"  - Preprocessing metadata (dates, aggregates, feature order)")
    print(f"  - Confidence interval statistics ({len(ci_stats['buckets'])} buckets)")
    print(f"  - Feature importance (top {len(feature_importance)} features)")
    print(f"  - Performance metrics (train/val/test)")
    print(f"  - Area unit: {area_unit}")
    print("=" * 80)


def load_model_artifact(artifact_path: str) -> Dict[str, Any]:
    """
    Load saved model artifact.

    Args:
        artifact_path: Path to saved artifact

    Returns:
        Dictionary with all artifact components
    """
    import pickle
    from catboost import CatBoostRegressor

    # Load the artifact metadata with pure pickle
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)

    # Load the CatBoost model separately
    catboost_path = artifact['model_path']
    model = CatBoostRegressor()
    model.load_model(catboost_path)
    artifact['model'] = model  # Add model to artifact

    print(f"Model artifact loaded from: {artifact_path}")
    print(f"CatBoost model loaded from: {catboost_path}")
    print(f"Model version: {artifact.get('model_version', 'unknown')}")
    print(f"Trained at: {artifact.get('trained_at', 'unknown')}")
    return artifact


def predict_with_confidence(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    ci_stats: Dict[str, Any],
    extrapolation_multiplier: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence intervals.

    Process:
    1. Predict in log-space, transform to original space
    2. For each prediction, find appropriate CI bucket based on predicted value
    3. Apply bucket's error quantiles to get confidence interval
    4. If prediction is outside training range (extrapolation), widen interval

    Args:
        model: Trained CatBoost model
        X: Features for prediction
        ci_stats: Confidence interval statistics from validation
        extrapolation_multiplier: How much to widen CI for extrapolation (default 1.5x)

    Returns:
        predictions, lower_bounds, upper_bounds, confidence_flags
    """
    # Predict in log-space and transform
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)

    # Initialize arrays
    lower_bounds = np.zeros(len(y_pred))
    upper_bounds = np.zeros(len(y_pred))
    confidence_flags = np.array(['medium'] * len(y_pred), dtype=object)

    # Get overall prediction range from CI stats
    all_edges = ci_stats['bucket_edges']
    min_pred = min([edge[0] for edge in all_edges])
    max_pred = max([edge[1] for edge in all_edges])

    # For each prediction, find appropriate bucket and apply CI
    for i, pred in enumerate(y_pred):
        # Check for extrapolation
        is_extrapolation = (pred < min_pred) or (pred > max_pred)

        # Find appropriate bucket
        bucket_idx = 0
        for j, (bucket_min, bucket_max) in enumerate(all_edges):
            if bucket_min <= pred <= bucket_max:
                bucket_idx = j
                break

        # Get error quantiles from bucket
        bucket = ci_stats['buckets'][bucket_idx]
        lower_error = bucket['lower_error']
        upper_error = bucket['upper_error']

        # Apply extrapolation multiplier if needed
        if is_extrapolation:
            lower_error *= extrapolation_multiplier
            upper_error *= extrapolation_multiplier
            confidence_flags[i] = 'low'
        else:
            confidence_flags[i] = 'high'

        # Compute confidence bounds
        lower_bounds[i] = max(0, pred - lower_error)  # Can't be negative
        upper_bounds[i] = pred + upper_error

    return y_pred, lower_bounds, upper_bounds, confidence_flags


# ==================== SEGMENTED EVALUATION UTILITIES ====================

def evaluate_by_price_buckets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_buckets: int = 5
) -> pd.DataFrame:
    """
    Evaluate model performance by price quantile buckets.

    This shows if model performs differently on low vs high-priced properties.

    Args:
        y_true: True prices
        y_pred: Predicted prices
        n_buckets: Number of quantile buckets

    Returns:
        DataFrame with metrics by bucket
    """
    # Create price buckets based on true values
    bucket_labels = [f"Q{i+1}" for i in range(n_buckets)]
    buckets = pd.qcut(y_true, q=n_buckets, labels=bucket_labels, duplicates='drop')

    results = []
    for bucket in bucket_labels:
        if bucket not in buckets.values:
            continue

        mask = (buckets == bucket)
        bucket_y_true = y_true[mask]
        bucket_y_pred = y_pred[mask]

        if len(bucket_y_true) > 0:
            bucket_metrics = {
                'bucket': bucket,
                'n_samples': len(bucket_y_true),
                'price_min': bucket_y_true.min(),
                'price_max': bucket_y_true.max(),
                'price_median': np.median(bucket_y_true),
                'r2': r2_score(bucket_y_true, bucket_y_pred),
                'mae': mean_absolute_error(bucket_y_true, bucket_y_pred),
                'mape': mean_absolute_percentage_error(bucket_y_true, bucket_y_pred) * 100
            }
            results.append(bucket_metrics)

    return pd.DataFrame(results)


def evaluate_by_category(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    category: pd.Series,
    category_name: str,
    min_samples: int = 30
) -> pd.DataFrame:
    """
    Evaluate model performance by categorical feature.

    Shows if model performs differently across property types, areas, etc.

    Args:
        y_true: True prices
        y_pred: Predicted prices
        category: Categorical feature values (e.g., AREA_EN, PROP_TYPE_EN)
        category_name: Name of category for display
        min_samples: Minimum samples to include category in results

    Returns:
        DataFrame with metrics by category
    """
    results = []
    for cat_value in category.unique():
        mask = (category == cat_value)
        cat_y_true = y_true[mask]
        cat_y_pred = y_pred[mask]

        if len(cat_y_true) >= min_samples:
            cat_metrics = {
                'category': cat_value,
                'n_samples': len(cat_y_true),
                'price_median': np.median(cat_y_true),
                'r2': r2_score(cat_y_true, cat_y_pred),
                'mae': mean_absolute_error(cat_y_true, cat_y_pred),
                'mape': mean_absolute_percentage_error(cat_y_true, cat_y_pred) * 100
            }
            results.append(cat_metrics)

    df = pd.DataFrame(results).sort_values('n_samples', ascending=False)
    return df
