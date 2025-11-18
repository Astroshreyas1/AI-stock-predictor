#!/usr/bin/env python3
"""
final_predictor.py - Stock predictor with guaranteed trades and no NaN issues
"""
import argparse, os, time, json, traceback, sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")

def get_clean_data(ticker, period="5y"):
    """Get and thoroughly clean stock data."""
    print(f"📊 Fetching {ticker} data...")
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    print(f"   Raw data: {len(df)} days")
    return df

def create_simple_features(df):
    """Create features with thorough NaN handling."""
    df = df.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'price_sma_{window}'] = df['Close'] / df[f'sma_{window}']
    
    # Volume
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # Volatility
    df['volatility_10'] = df['returns'].rolling(10).std()
    
    # Momentum
    for window in [3, 5]:
        df[f'momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
    
    # Clean ALL NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    print(f"   Clean features: {len(df)} days, {len(df.columns)} columns")
    return df

def create_clean_target(df):
    """Create target with no NaN values."""
    # Simple target: predict if next day will be green (close > open)
    target = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    # Remove the last row which will have NaN for target
    target = target.iloc[:-1]
    df = df.iloc[:-1]
    
    # Final check
    if target.isna().any():
        print("❌ WARNING: NaN in target, removing...")
        valid_idx = target.dropna().index
        target = target.loc[valid_idx]
        df = df.loc[valid_idx]
    
    pos_rate = target.mean()
    print(f"🎯 Target: {pos_rate:.1%} positive (next day green)")
    print(f"   {target.sum():.0f} up vs {len(target)-target.sum():.0f} down")
    
    return target, df

def train_guaranteed_trades_model(X_train, y_train, X_test):
    """Train model that guarantees trades."""
    print("🤖 Training model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=1
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Get probabilities
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Use low threshold to guarantee trades
    threshold = 0.4
    predictions = (test_proba >= threshold).astype(int)
    
    trade_count = predictions.sum()
    print(f"   Threshold: {threshold}")
    print(f"   Trades generated: {trade_count} ({predictions.mean():.1%})")
    
    return model, scaler, test_proba, predictions, threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--out_dir', default='./models', help='Output directory')
    parser.add_argument('--period', default='5y', help='Data period')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting FINAL prediction pipeline...")
        
        # Step 1: Get data
        df_raw = get_clean_data(args.ticker, args.period)
        
        # Step 2: Create features
        df_features = create_simple_features(df_raw)
        
        # Step 3: Create target (returns aligned df_features)
        target, df_features = create_clean_target(df_features)
        
        # Step 4: Prepare features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        X = df_features[feature_cols]
        
        print(f"   Final dataset: {len(X)} samples, {len(feature_cols)} features")
        
        # Step 5: Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        # Final NaN check
        print(f"   NaN check - X_train: {X_train.isna().sum().sum()}")
        print(f"   NaN check - y_train: {y_train.isna().sum()}")
        print(f"   NaN check - X_test: {X_test.isna().sum().sum()}")
        print(f"   NaN check - y_test: {y_test.isna().sum()}")
        
        if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
            raise ValueError("NaN values detected in training data!")
        
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"   Test positive rate: {y_test.mean():.1%}")
        
        # Step 6: Train model
        model, scaler, test_proba, predictions, threshold = train_guaranteed_trades_model(
            X_train, y_train, X_test
        )
        
        # Step 7: Evaluate
        accuracy = accuracy_score(y_test, predictions)
        
        try:
            roc_auc = roc_auc_score(y_test, test_proba)
        except:
            roc_auc = 0.5
        
        print(f"📊 FINAL RESULTS:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   ROC AUC:  {roc_auc:.3f}")
        print(f"   Predictions: {predictions.mean():.1%} BUY")
        print(f"   Actual:      {y_test.mean():.1%} green")
        print(f"   TRADES: {predictions.sum()}")
        
        # Step 8: Backtest
        signals = pd.Series(predictions, index=X_test.index)
        returns = df_features.loc[X_test.index, 'Close'].pct_change().shift(-1).fillna(0)
        strategy_returns = returns * signals
        
        if len(strategy_returns) > 0:
            cum_strategy = (1 + strategy_returns).cumprod()
            cum_buy_hold = (1 + returns).cumprod()
            
            strategy_final = cum_strategy.iloc[-1] - 1
            buy_hold_final = cum_buy_hold.iloc[-1] - 1
            win_rate = (strategy_returns > 0).mean()
            
            if strategy_returns.std() > 0:
                sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
            else:
                sharpe = 0.0
            
            print(f"💰 BACKTEST:")
            print(f"   Strategy: {strategy_final:+.1%}")
            print(f"   Buy & Hold: {buy_hold_final:+.1%}")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Sharpe: {sharpe:.2f}")
        else:
            strategy_final, buy_hold_final, win_rate, sharpe = 0, 0, 0, 0
        
        # Step 9: Save model
        artifact = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'threshold': threshold,
            'metadata': {
                'ticker': args.ticker,
                'trained_at': pd.Timestamp.now().isoformat(),
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'trades': int(predictions.sum()),
                'strategy_return': float(strategy_final),
                'buy_hold_return': float(buy_hold_final),
                'win_rate': float(win_rate)
            }
        }
        
        os.makedirs(args.out_dir, exist_ok=True)
        model_path = os.path.join(args.out_dir, f'final_model_{args.ticker}.joblib')
        joblib.dump(artifact, model_path)
        
        print(f"💾 Model saved: {model_path}")
        
        if predictions.sum() > 0:
            print("✅ SUCCESS! Trading model ready!")
        else:
            print("⚠️  No trades generated - check data quality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()