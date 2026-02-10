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
    random_seed: int = 42,
    quantile_alpha: float = 0.5
) -> CatBoostRegressor:
    """
    Train CatBoost regression model with Quantile loss.

    Quantile regression advantages:
    - Predicts median (50th percentile) which is robust to outliers
    - Better performance in real space after log transformation
    - Can train multiple models for different quantiles (confidence intervals)

    CatBoost advantages:
    - Handles categorical features natively (no one-hot encoding needed)
    - Built-in handling for unseen categories
    - Robust to outliers
    - Fast inference for production API

    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_val: Validation features
        y_val: Validation target (log-transformed)
        categorical_indices: List of categorical feature column indices
        hyperparameters: Custom hyperparameters (if None, use defaults)
        random_seed: Random seed for reproducibility
        quantile_alpha: Quantile to predict (0.5 = median, 0.05/0.95 for CI)

    Returns:
        Trained CatBoostRegressor model
    """
    # Default hyperparameters optimized for quantile regression
    if hyperparameters is None:
        hyperparameters = {
            'iterations': 1500,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 5,
            'loss_function': f'Quantile:alpha={quantile_alpha}',
            'eval_metric': f'Quantile:alpha={quantile_alpha}',
            'random_seed': random_seed,
            'verbose': 100,
            'early_stopping_rounds': 75,
            'use_best_model': True
        }

    quantile_label = {0.05: "5th percentile (lower bound)",
                      0.5: "median (50th percentile)",
                      0.95: "95th percentile (upper bound)"}.get(quantile_alpha, f"{quantile_alpha*100:.0f}th percentile")

    print("=" * 80)
    print(f"TRAINING CATBOOST MODEL - {quantile_label}")
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
    metric_name = f'Quantile:alpha={quantile_alpha}'
    print(f"Best validation loss (log-space): {model.best_score_['validation'][metric_name]:.4f}")

    return model


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_indices: List[int],
    random_seed: int = 42
) -> Dict[str, CatBoostRegressor]:
    """
    Train three quantile regression models for median prediction + confidence intervals.

    This is the KEY improvement over residual-based confidence intervals:
    - Quantile models predict percentiles directly in log space
    - When converted to real space, they maintain proper ordering
    - Much better real-space performance than MAE + residuals

    Models trained:
    - alpha=0.05: Lower bound (5th percentile)
    - alpha=0.50: Median prediction (50th percentile) - main model
    - alpha=0.95: Upper bound (95th percentile)

    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_val: Validation features
        y_val: Validation target (log-transformed)
        categorical_indices: List of categorical feature column indices
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with three trained models: {'lower': model_0.05, 'median': model_0.50, 'upper': model_0.95}
    """
    print("\n" + "=" * 80)
    print("TRAINING QUANTILE REGRESSION MODELS")
    print("=" * 80)
    print("\nThis trains 3 separate models for robust confidence intervals:")
    print("  1. Lower bound (5th percentile)")
    print("  2. Median prediction (50th percentile)")
    print("  3. Upper bound (95th percentile)")
    print("\nEach model optimizes for its specific quantile.")
    print("=" * 80)

    models = {}

    # Train lower bound model (5th percentile)
    print("\n[1/3] Training LOWER BOUND model...")
    models['lower'] = train_catboost_model(
        X_train, y_train, X_val, y_val,
        categorical_indices=categorical_indices,
        random_seed=random_seed,
        quantile_alpha=0.05
    )

    # Train median model (50th percentile) - this is the main prediction
    print("\n[2/3] Training MEDIAN model...")
    models['median'] = train_catboost_model(
        X_train, y_train, X_val, y_val,
        categorical_indices=categorical_indices,
        random_seed=random_seed,
        quantile_alpha=0.5
    )

    # Train upper bound model (95th percentile)
    print("\n[3/3] Training UPPER BOUND model...")
    models['upper'] = train_catboost_model(
        X_train, y_train, X_val, y_val,
        categorical_indices=categorical_indices,
        random_seed=random_seed,
        quantile_alpha=0.95
    )

    print("\n" + "=" * 80)
    print("ALL QUANTILE MODELS TRAINED SUCCESSFULLY")
    print("=" * 80)

    return models


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
    models: Dict[str, CatBoostRegressor],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_areas: np.ndarray = None,
    val_areas: np.ndarray = None,
    test_areas: np.ndarray = None,
    use_price_per_sqm: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate quantile regression models on train, validation, and test sets.

    Uses the median (50th percentile) model for evaluation metrics.

    KEY FEATURE: Supports price_per_sqm target with price reconstruction.
    - If use_price_per_sqm=True: Model predicts price_per_sqm, we reconstruct total price
    - If use_price_per_sqm=False: Model predicts total price directly (old approach)

    Args:
        models: Dictionary with 'lower', 'median', 'upper' quantile models
        X_train, y_train: Training data (y in log-space, either price or price_per_sqm)
        X_val, y_val: Validation data (y in log-space)
        X_test, y_test: Test data (y in log-space)
        train_areas, val_areas, test_areas: ACTUAL_AREA for price reconstruction
        use_price_per_sqm: If True, reconstruct price from price_per_sqm

    Returns:
        Dictionary with metrics for each dataset
    """
    print("\n" + "=" * 80)
    if use_price_per_sqm:
        print("EVALUATING MODEL (price_per_sqm target with reconstruction)")
    else:
        print("EVALUATING MODEL (absolute price target)")
    print("=" * 80)

    # Use the median model for evaluation
    median_model = models['median']
    results = {}

    # Train set
    y_train_pred_log = median_model.predict(X_train)
    if use_price_per_sqm and train_areas is not None:
        # Reconstruct price = price_per_sqm × area
        y_train_true_ppsm = np.expm1(y_train)
        y_train_pred_ppsm = np.expm1(y_train_pred_log)
        y_train_true = y_train_true_ppsm * train_areas
        y_train_pred = y_train_pred_ppsm * train_areas
    else:
        # Direct price prediction
        y_train_true = np.expm1(y_train)
        y_train_pred = np.expm1(y_train_pred_log)
    results['train'] = compute_metrics(y_train_true, y_train_pred, "Train")

    # Validation set
    y_val_pred_log = median_model.predict(X_val)
    if use_price_per_sqm and val_areas is not None:
        y_val_true_ppsm = np.expm1(y_val)
        y_val_pred_ppsm = np.expm1(y_val_pred_log)
        y_val_true = y_val_true_ppsm * val_areas
        y_val_pred = y_val_pred_ppsm * val_areas
    else:
        y_val_true = np.expm1(y_val)
        y_val_pred = np.expm1(y_val_pred_log)
    results['val'] = compute_metrics(y_val_true, y_val_pred, "Validation")

    # Test set
    y_test_pred_log = median_model.predict(X_test)
    if use_price_per_sqm and test_areas is not None:
        y_test_true_ppsm = np.expm1(y_test)
        y_test_pred_ppsm = np.expm1(y_test_pred_log)
        y_test_true = y_test_true_ppsm * test_areas
        y_test_pred = y_test_pred_ppsm * test_areas
    else:
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
    models: Dict[str, CatBoostRegressor],
    preprocessing_metadata: Dict[str, Any],
    feature_importance: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    save_path: str,
    area_unit: str = "sqm"
) -> None:
    """
    Save complete model artifact with all three quantile models.

    The artifact contains EVERYTHING needed for inference:
    - Three quantile regression models (lower, median, upper bounds)
    - Preprocessing metadata (dates, medians, aggregates, feature order)
    - Feature importance (for API key_factors)
    - Performance metrics (for monitoring)
    - Area unit (sqm or sqft)

    Args:
        models: Dictionary with 'lower', 'median', 'upper' quantile models
        preprocessing_metadata: From preprocessing pipeline
        feature_importance: Feature importance DataFrame
        metrics: Model performance metrics
        save_path: Path to save artifact (e.g., 'models/trained_model.pkl')
        area_unit: Unit for area measurements ('sqm' or 'sqft')
    """
    import os
    model_dir = os.path.dirname(os.path.abspath(save_path))
    model_filename = os.path.splitext(os.path.basename(save_path))[0]

    # Save all three quantile models separately
    model_paths = {}
    for quantile_name in ['lower', 'median', 'upper']:
        model_path = os.path.join(model_dir, f"{model_filename}_{quantile_name}.cbm")
        models[quantile_name].save_model(model_path)
        model_paths[quantile_name] = model_path

    # Create artifact WITHOUT the models (we'll store the paths instead)
    artifact = {
        'model_paths': model_paths,  # Paths to three CatBoost model files
        'preprocessing_metadata': preprocessing_metadata,
        'feature_importance': feature_importance,
        'metrics': metrics,
        'area_unit': area_unit,
        'model_version': '2.0',  # v2.0 with quantile regression
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

    total_model_size = 0
    for quantile_name, model_path in model_paths.items():
        model_size_mb = os.path.getsize(model_path) / (1024**2)
        total_model_size += model_size_mb
        print(f"Model ({quantile_name}): {os.path.basename(model_path)}")
        print(f"  Size: {model_size_mb:.2f} MB")

    print(f"\nArtifact contents:")
    print(f"  - Three quantile regression models (5th, 50th, 95th percentiles)")
    print(f"  - Total model size: {total_model_size:.2f} MB")
    print(f"  - Preprocessing metadata (dates, aggregates, feature order)")
    print(f"  - Feature importance (top {len(feature_importance)} features)")
    print(f"  - Performance metrics (train/val/test)")
    print(f"  - Area unit: {area_unit}")
    print("=" * 80)


def load_model_artifact(artifact_path: str) -> Dict[str, Any]:
    """
    Load saved model artifact with all quantile models.

    Handles both v1.0 (single model) and v2.0 (three quantile models) artifacts
    for backward compatibility.

    Args:
        artifact_path: Path to saved artifact

    Returns:
        Dictionary with all artifact components including loaded models
    """
    import pickle
    import os
    from catboost import CatBoostRegressor

    # Load the artifact metadata with pure pickle
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)

    artifact_dir = os.path.dirname(os.path.abspath(artifact_path))
    version = artifact.get('model_version', '1.0')

    print(f"Model artifact loaded from: {artifact_path}")
    print(f"Model version: {version}")

    # Handle v2.0 (quantile regression models)
    if version == '2.0' and 'model_paths' in artifact:
        models = {}
        for quantile_name, model_path in artifact['model_paths'].items():
            # Make path absolute if relative
            if not os.path.isabs(model_path):
                model_path = os.path.join(artifact_dir, os.path.basename(model_path))

            model = CatBoostRegressor()
            model.load_model(model_path)
            models[quantile_name] = model
            print(f"  - Loaded {quantile_name} quantile model from: {os.path.basename(model_path)}")

        artifact['models'] = models

    # Handle v1.0 (single model) - backward compatibility
    elif 'model_path' in artifact:
        catboost_path = artifact['model_path']
        if not os.path.isabs(catboost_path):
            catboost_path = os.path.join(artifact_dir, os.path.basename(catboost_path))

        model = CatBoostRegressor()
        model.load_model(catboost_path)
        artifact['model'] = model  # Keep old 'model' key for compatibility
        print(f"  - Loaded single model from: {os.path.basename(catboost_path)}")
        print("  WARNING: This is a v1.0 model. Retrain with quantile regression for better performance.")

    print(f"Trained at: {artifact.get('trained_at', 'unknown')}")
    return artifact


def predict_with_confidence(
    models: Dict[str, CatBoostRegressor],
    X: pd.DataFrame,
    actual_areas: np.ndarray = None,
    use_price_per_sqm: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence intervals using quantile regression models.

    This uses three separate quantile models:
    - models['lower']: Predicts 5th percentile
    - models['median']: Predicts 50th percentile (median) - main prediction
    - models['upper']: Predicts 95th percentile

    KEY FEATURE: Supports price_per_sqm target with price reconstruction.
    - If use_price_per_sqm=True and actual_areas provided: Reconstruct price = price_per_sqm × area
    - Otherwise: Direct price prediction

    Args:
        models: Dictionary with 'lower', 'median', 'upper' quantile models
        X: Features for prediction
        actual_areas: ACTUAL_AREA values for price reconstruction (if using price_per_sqm)
        use_price_per_sqm: If True, reconstruct price from price_per_sqm

    Returns:
        predictions, lower_bounds, upper_bounds, confidence_flags (all in price space)
    """
    # Predict with all three models in log-space
    y_pred_log_lower = models['lower'].predict(X)
    y_pred_log_median = models['median'].predict(X)
    y_pred_log_upper = models['upper'].predict(X)

    # Transform to real space (price_per_sqm or price)
    lower_bounds_ppsm = np.expm1(y_pred_log_lower)
    predictions_ppsm = np.expm1(y_pred_log_median)
    upper_bounds_ppsm = np.expm1(y_pred_log_upper)

    # Reconstruct prices if using price_per_sqm target
    if use_price_per_sqm and actual_areas is not None:
        lower_bounds = lower_bounds_ppsm * actual_areas
        predictions = predictions_ppsm * actual_areas
        upper_bounds = upper_bounds_ppsm * actual_areas
    else:
        lower_bounds = lower_bounds_ppsm
        predictions = predictions_ppsm
        upper_bounds = upper_bounds_ppsm

    # Compute confidence interval width as percentage of prediction
    ci_width_pct = ((upper_bounds - lower_bounds) / predictions) * 100

    # Assign confidence flags based on CI width
    # Narrow CI = high confidence, Wide CI = low confidence
    confidence_flags = np.array(['medium'] * len(predictions), dtype=object)
    confidence_flags[ci_width_pct < 50] = 'high'  # CI < 50% of prediction
    confidence_flags[ci_width_pct > 100] = 'low'  # CI > 100% of prediction

    # Ensure lower < prediction < upper (quantile crossing can happen rarely)
    lower_bounds = np.minimum(lower_bounds, predictions)
    upper_bounds = np.maximum(upper_bounds, predictions)

    # Ensure non-negative prices
    lower_bounds = np.maximum(0, lower_bounds)

    return predictions, lower_bounds, upper_bounds, confidence_flags


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
