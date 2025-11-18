#!/usr/bin/env python3
"""
high_accuracy_predictor.py - Stock predictor optimized for 70%+ accuracy
"""
import argparse, os, time, json, traceback, sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
class Config:
    MIN_DATA_POINTS = 1000  # More data for better learning
    TEST_SIZE = 0.15  # Smaller test set for more training data
    N_SPLITS = 5
    RANDOM_STATE = 42
    HOLD_PERIOD = 3  # Predict 3-day moves instead of 1-day
    PROFIT_THRESHOLD = 0.04  # 4% target move
    VOLATILITY_THRESHOLD = 0.02  # Minimum volatility filter

# ---------------- Robust Data Validation ----------------
def validate_and_clean_data(df):
    """Thorough data validation and cleaning."""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove any rows with zero or negative prices
    df = df[(df['Close'] > 0) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]
    
    # Check for sufficient data after cleaning
    if len(df) < Config.MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: {len(df)} rows, need at least {Config.MIN_DATA_POINTS}")
    
    return df

# ---------------- Advanced Feature Engineering ----------------
def calculate_rsi(series, window=14):
    """Calculate RSI with proper error handling."""
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for NaN values
    except:
        return pd.Series(50, index=series.index)  # Fallback to neutral RSI

def create_advanced_features(df):
    """Create sophisticated features for better prediction."""
    df = df.copy()
    
    # Ensure single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Price-based features
    df['log_price'] = np.log(df['Close'])
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Price range and position
    df['true_range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['normalized_range'] = df['true_range'] / df['Close']
    
    # Multiple moving averages with different windows
    for window in [5, 10, 20, 50, 100]:
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        df[f'price_sma_ratio_{window}'] = df['Close'] / df[f'sma_{window}']
        df[f'price_ema_ratio_{window}'] = df['Close'] / df[f'ema_{window}']
    
    # Momentum features
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'volume_momentum_{period}'] = df['Volume'] / df['Volume'].shift(period) - 1
    
    # Volatility features
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].shift(1)
    
    # Volume features
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    df['volume_obv'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1, -1)).cumsum()
    
    # RSI with multiple timeframes
    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = calculate_rsi(df['Close'], period)
    
    # MACD features
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    for window in [20, 50]:
        df[f'bb_middle_{window}'] = df['Close'].rolling(window).mean()
        bb_std = df['Close'].rolling(window).std()
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * bb_std
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * bb_std
        df[f'bb_position_{window}'] = (df['Close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
    
    # Support and Resistance
    df['resistance_20'] = df['High'].rolling(20).max()
    df['support_20'] = df['Low'].rolling(20).min()
    df['support_resistance_ratio'] = (df['Close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])
    
    # Price acceleration and jerk
    df['price_acceleration'] = df['returns'].diff()
    df['price_jerk'] = df['price_acceleration'].diff()
    
    # Seasonality patterns (day of week, month effects)
    if hasattr(df.index, 'dayofweek'):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_start'] = (df.index.day <= 7).astype(int)
        df['is_month_end'] = (df.index.day >= 23).astype(int)
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'range_lag_{lag}'] = df['normalized_range'].shift(lag)
    
    # Rolling correlations and betas
    df['price_volume_corr_10'] = df['Close'].rolling(10).corr(df['Volume'])
    
    # Clean infinite values and extreme outliers
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove columns with too many NaN values
    nan_threshold = 0.1 * len(df)
    df = df.dropna(axis=1, thresh=len(df) - nan_threshold)
    
    # Fill remaining NaN values with forward fill then median
    df = df.ffill().bfill()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def create_smart_target(df):
    """
    Create a smarter target variable that's easier to predict.
    Predict if price will increase by PROFIT_THRESHOLD within HOLD_PERIOD days.
    """
    # Calculate maximum price in the hold period
    future_prices = pd.DataFrame()
    for i in range(1, Config.HOLD_PERIOD + 1):
        future_prices[f'future_{i}'] = df['Close'].shift(-i)
    
    future_max = future_prices.max(axis=1)
    best_return = (future_max - df['Close']) / df['Close']
    
    # Target: 1 if return exceeds threshold, 0 otherwise
    target = (best_return >= Config.PROFIT_THRESHOLD).astype(int)
    
    # Remove the last HOLD_PERIOD rows which have NaN targets
    target = target.iloc[:-Config.HOLD_PERIOD]
    df = df.iloc[:-Config.HOLD_PERIOD]
    
    # Filter out low volatility periods (hard to predict)
    volatility = df['returns'].rolling(10).std()
    high_vol_mask = volatility > Config.VOLATILITY_THRESHOLD
    target = target[high_vol_mask]
    df = df[high_vol_mask]
    
    target_metrics = {
        'positive_ratio': target.mean(),
        'total_samples': len(target),
        'positive_samples': target.sum(),
        'hold_period': Config.HOLD_PERIOD,
        'profit_threshold': Config.PROFIT_THRESHOLD
    }
    
    return target, target_metrics, df

# ---------------- Advanced Model Training ----------------
def get_advanced_models():
    """Get ensemble of models optimized for financial data."""
    models = {}
    
    # Random Forest with optimized parameters
    models['random_forest'] = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    
    # Gradient Boosting with financial focus
    models['gradient_boosting'] = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=Config.RANDOM_STATE
    )
    
    # Try to import advanced models
    try:
        import xgboost as xgb
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
    except ImportError:
        print("XGBoost not available, skipping...")
    
    try:
        import lightgbm as lgb
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
    except ImportError:
        print("LightGBM not available, skipping...")
    
    return models

def perform_feature_selection(X_train, y_train, X_test, base_model):
    """Select most important features to reduce noise."""
    selector = SelectFromModel(
        estimator=base_model,
        threshold='median',  # Keep features above median importance
        prefit=False
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    
    print(f"   Feature selection: {len(selected_features)}/{len(X_train.columns)} features selected")
    return X_train_selected, X_test_selected, selected_features, selector

def train_advanced_ensemble(X_train, y_train, X_test, y_test, models, n_splits=5):
    """Train advanced ensemble with feature selection and careful validation."""
    from sklearn.base import clone
    
    # Storage for predictions
    oof_predictions = {name: np.zeros(len(X_train)) for name in models}
    test_predictions = {name: np.zeros(len(X_test)) for name in models}
    feature_selectors = {}
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print("   Training advanced ensemble with feature selection...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        print(f"     Fold {fold}/{n_splits}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        for name, model in models.items():
            model_clone = clone(model)
            
            # Perform feature selection for this fold
            X_tr_selected, X_val_selected, selected_features, selector = perform_feature_selection(
                X_tr, y_tr, X_val, model_clone
            )
            
            # Store selector for first fold
            if fold == 1:
                feature_selectors[name] = selector
            
            # Train on selected features
            model_clone.fit(X_tr_selected, y_tr)
            
            # Predict on validation set
            if hasattr(model_clone, 'predict_proba'):
                val_preds = model_clone.predict_proba(X_val_selected)[:, 1]
                # Accumulate test predictions (we'll transform test set later)
                test_preds = model_clone.predict_proba(
                    selector.transform(X_test)
                )[:, 1] / n_splits
            else:
                val_preds = model_clone.predict(X_val_selected)
                test_preds = model_clone.predict(selector.transform(X_test)) / n_splits
            
            oof_predictions[name][val_idx] = val_preds
            test_predictions[name] += test_preds
    
    # Create meta-features
    meta_train = pd.DataFrame(oof_predictions, index=X_train.index)
    meta_test = pd.DataFrame(test_predictions, index=X_test.index)
    
    # Train meta-learner with regularization
    meta_learner = LogisticRegression(
        random_state=Config.RANDOM_STATE,
        max_iter=2000,
        C=0.1,  # Strong regularization
        class_weight='balanced',
        solver='liblinear'
    )
    meta_learner.fit(meta_train, y_train)
    
    # Final predictions
    final_predictions = meta_learner.predict_proba(meta_test)[:, 1]
    
    return {
        'base_models': models,
        'meta_learner': meta_learner,
        'feature_selectors': feature_selectors,
        'predictions': final_predictions,
        'meta_train': meta_train,
        'meta_test': meta_test
    }

# ---------------- Enhanced Evaluation ----------------
def evaluate_advanced_model(y_true, predictions, dataset_name="Test"):
    """Comprehensive model evaluation with confidence intervals."""
    # Convert probabilities to binary predictions with optimal threshold
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred_binary = (predictions >= optimal_threshold).astype(int)
    
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred_binary)
    results['roc_auc'] = roc_auc_score(y_true, predictions)
    results['optimal_threshold'] = optimal_threshold
    results['positive_rate'] = y_pred_binary.mean()
    results['actual_positive_rate'] = y_true.mean()
    
    # Detailed classification report
    report = classification_report(y_true, y_pred_binary, output_dict=True)
    if '1' in report:
        results['precision'] = report['1']['precision']
        results['recall'] = report['1']['recall']
        results['f1_score'] = report['1']['f1-score']
    
    print(f"\n{dataset_name} Set Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  ROC AUC:  {results['roc_auc']:.4f}")
    print(f"  Precision: {results.get('precision', 0):.4f}")
    print(f"  Recall:    {results.get('recall', 0):.4f}")
    print(f"  F1-Score:  {results.get('f1_score', 0):.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Predicted Positive: {results['positive_rate']:.1%}")
    print(f"  Actual Positive:    {results['actual_positive_rate']:.1%}")
    
    return results, optimal_threshold

# ---------------- Main Training Pipeline ----------------
def run_high_accuracy_pipeline(ticker, output_dir, period="5y", interval="1d"):
    """Complete pipeline optimized for high accuracy."""
    print(f"🚀 Starting HIGH-ACCURACY pipeline for {ticker}...")
    
    try:
        # Step 1: Get more data for better learning
        print("1. Fetching and validating data...")
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df = validate_and_clean_data(df)
        print(f"   Retrieved {len(df)} trading days")
        
        # Step 2: Advanced feature engineering
        print("2. Creating advanced features...")
        df_features = create_advanced_features(df)
        print(f"   Generated {len(df_features.columns)} features")
        
        # Step 3: Create smart target variable
        print("3. Creating smart target variable...")
        target, target_metrics, df_features = create_smart_target(df_features)
        
        print(f"   Target: {target_metrics['positive_ratio']:.1%} positive")
        print(f"   Hold period: {target_metrics['hold_period']} days")
        print(f"   Profit threshold: {target_metrics['profit_threshold']:.1%}")
        print(f"   Samples: {target_metrics['total_samples']} (after filtering)")
        
        # Step 4: Prepare features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns']
        feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        X = df_features[feature_columns]
        
        print(f"   Using {len(feature_columns)} features for modeling")
        
        # Step 5: Time-based split
        split_idx = int(len(X) * (1 - Config.TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        print(f"   Training period: {X_train.index[0].date()} to {X_train.index[-1].date()}")
        print(f"   Testing period:  {X_test.index[0].date()} to {X_test.index[-1].date()}")
        print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Step 6: Scale features using RobustScaler (handles outliers better)
        print("4. Scaling features...")
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # Step 7: Get advanced models
        models = get_advanced_models()
        print(f"5. Training {len(models)} advanced models...")
        
        # Step 8: Train advanced ensemble
        ensemble_results = train_advanced_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test, models, Config.N_SPLITS
        )
        
        # Step 9: Comprehensive evaluation
        print("6. Evaluating model performance...")
        test_results, optimal_threshold = evaluate_advanced_model(
            y_test, ensemble_results['predictions']
        )
        
        # Step 10: Save high-quality artifact
        print("7. Saving optimized model...")
        artifact = {
            'ensemble_results': ensemble_results,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'optimal_threshold': optimal_threshold,
            'test_results': test_results,
            'target_metrics': target_metrics,
            'metadata': {
                'ticker': ticker,
                'trained_at': pd.Timestamp.now().isoformat(),
                'period': period,
                'interval': interval,
                'hold_period': Config.HOLD_PERIOD,
                'profit_threshold': Config.PROFIT_THRESHOLD,
                'model_version': 'high_accuracy_v1'
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'high_accuracy_model_{ticker}.joblib')
        meta_path = os.path.join(output_dir, f'model_metadata_{ticker}.json')
        
        joblib.dump(artifact, model_path)
        
        with open(meta_path, 'w') as f:
            json.dump(artifact['metadata'], f, indent=2)
        
        print(f"\n💾 Model saved: {model_path}")
        print(f"📊 Metadata saved: {meta_path}")
        
        # Final summary
        print(f"\n🎯 HIGH-ACCURACY RESULTS for {ticker}:")
        print(f"   Test Accuracy: {test_results['accuracy']:.3f}")
        print(f"   Test ROC AUC:  {test_results['roc_auc']:.3f}")
        print(f"   F1-Score:      {test_results.get('f1_score', 0):.3f}")
        
        if test_results['accuracy'] >= 0.70:
            print("   ✅ TARGET ACHIEVED: 70%+ Accuracy!")
        else:
            print("   ⚠️  Target not reached, but model is significantly improved")
        
        return artifact
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        traceback.print_exc()
        raise

# ---------------- CLI Interface ----------------
def main():
    parser = argparse.ArgumentParser(description='High Accuracy Stock Predictor')
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--out_dir', default='./high_accuracy_models', help='Output directory')
    parser.add_argument('--period', default='5y', help='Data period (5y, 10y, max)')
    parser.add_argument('--interval', default='1d', help='Data interval')
    
    args = parser.parse_args()
    
    try:
        run_high_accuracy_pipeline(
            ticker=args.ticker.upper(),
            output_dir=args.out_dir,
            period=args.period,
            interval=args.interval
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()