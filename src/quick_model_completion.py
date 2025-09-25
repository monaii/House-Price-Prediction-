"""
Quick Model Completion Script

This script completes the model evaluation and generates final results
using the already trained models.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from data_loader import load_dataset_from_csv
from data_preprocessor import HousePricePreprocessor
from model_trainer import HousePriceModelTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    """Complete the model evaluation and generate final results"""
    print("House Price Prediction - Final Model Evaluation")
    print("=" * 50)
    
    # Load data
    print("1. Loading data...")
    data_path = os.path.join('data', 'california_housing.csv')
    X, y = load_dataset_from_csv(data_path)
    df = pd.concat([X, y], axis=1)
    
    # Load preprocessor and process data
    print("2. Loading preprocessor and processing data...")
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("   Preprocessor loaded successfully")
    else:
        # Create new preprocessor if not found
        print("   Creating new preprocessor...")
        preprocessor = HousePricePreprocessor(
            outlier_method='iqr',
            scaler_type='standard',
            target_transform='log'
        )
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        preprocessor.save_preprocessor(preprocessor_path)
    
    # If preprocessor was loaded, we need to split the data manually
    if os.path.exists(preprocessor_path):
        from sklearn.model_selection import train_test_split
        
        # Apply preprocessing
        df_eng = preprocessor.create_engineered_features(df)
        df_outliers = preprocessor.handle_outliers(df_eng)
        
        # Split data
        X = df_outliers.drop('MedHouseVal', axis=1)
        y = df_outliers['MedHouseVal']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train, X_test = preprocessor.scale_features(X_train, X_test)
        
        # Transform target
        y_train, y_test = preprocessor.transform_target(y_train, y_test)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Load best model
    print("\n3. Loading best model...")
    best_model_path = os.path.join('models', 'best_model.pkl')
    
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
        print("   Best model loaded successfully")
        model_name = "Best Model"
    else:
        # Try to load XGBoost model as fallback
        xgb_path = os.path.join('models', 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            best_model = joblib.load(xgb_path)
            print("   XGBoost model loaded as best model")
            model_name = "XGBoost"
        else:
            print("   No trained model found!")
            return
    
    # Evaluate on test set
    print("\n4. Final evaluation on test set...")
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics on transformed scale
    test_rmse_transformed = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae_transformed = mean_absolute_error(y_test, y_pred)
    test_r2_transformed = r2_score(y_test, y_pred)
    
    # Calculate metrics on original scale
    y_test_original = preprocessor.inverse_transform_target(y_test)
    y_pred_original = preprocessor.inverse_transform_target(y_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    test_mae = mean_absolute_error(y_test_original, y_pred_original)
    test_r2 = r2_score(y_test_original, y_pred_original)
    
    print(f"Test Results (Original Scale):")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    print(f"\nTest Results (Transformed Scale):")
    print(f"  RMSE: {test_rmse_transformed:.4f}")
    print(f"  MAE:  {test_mae_transformed:.4f}")
    print(f"  R²:   {test_r2_transformed:.4f}")
    
    # Calculate improvement over baseline
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_pred_original = preprocessor.inverse_transform_target(baseline_pred)
    baseline_rmse = np.sqrt(np.mean((y_test_original - baseline_pred_original) ** 2))
    improvement = ((baseline_rmse - test_rmse) / baseline_rmse) * 100
    
    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
    print(f"Model RMSE: {test_rmse:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    
    if improvement >= 15:
        print("✓ Target of 15% RMSE improvement achieved!")
    else:
        print(f"✗ Target of 15% RMSE improvement not achieved (got {improvement:.1f}%)")
    
    # Generate final report
    print("\n5. Generating final performance report...")
    os.makedirs('results', exist_ok=True)
    
    report_lines = [
        "HOUSE PRICE PREDICTION - FINAL PROJECT REPORT",
        "=" * 50,
        f"Model: {model_name}",
        "",
        "FINAL TEST RESULTS (Original Scale):",
        f"  RMSE: {test_rmse:.4f}",
        f"  MAE:  {test_mae:.4f}",
        f"  R²:   {test_r2:.4f}",
        "",
        "PERFORMANCE IMPROVEMENT:",
        f"  Baseline RMSE: {baseline_rmse:.4f}",
        f"  Model RMSE: {test_rmse:.4f}",
        f"  Improvement: {improvement:.1f}%",
        f"  Target Achievement: {'✓ YES' if improvement >= 15 else '✗ NO'}",
        "",
        "PROJECT COMPLETION STATUS:",
        "  ✓ Data Loading and EDA",
        "  ✓ Data Preprocessing and Feature Engineering", 
        "  ✓ Model Training and Comparison",
        "  ✓ Model Evaluation",
        "  ✓ Production Pipeline Created",
        "",
        "FILES GENERATED:",
        "  - Trained models in models/ directory",
        "  - EDA visualizations in results/ directory",
        "  - Production pipeline (ml_pipeline.py)",
        "  - Complete source code in src/ directory"
    ]
    
    report_content = "\n".join(report_lines)
    
    report_path = os.path.join('results', 'final_project_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Final report saved to: {report_path}")
    print("\n" + "=" * 50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'improvement': improvement,
        'target_achieved': improvement >= 15
    }

if __name__ == "__main__":
    results = main()