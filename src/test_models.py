"""
Test script for model training and evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from data_loader import load_dataset_from_csv
from data_preprocessor import HousePricePreprocessor
from model_trainer import HousePriceModelTrainer

def main():
    """Main function to test the complete ML pipeline"""
    print("House Price Prediction - Model Training and Evaluation")
    print("=" * 60)
    
    # Load data
    print("1. Loading data...")
    data_path = os.path.join('data', 'california_housing.csv')
    X, y = load_dataset_from_csv(data_path)
    
    # Combine features and target for preprocessing
    df = pd.concat([X, y], axis=1)
    print(f"   Dataset shape: {df.shape}")
    
    # Initialize preprocessor
    print("\n2. Preprocessing data...")
    preprocessor = HousePricePreprocessor(
        outlier_method='iqr',
        scaler_type='standard',
        target_transform='log'
    )
    
    # Fit and transform data
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {len(preprocessor.feature_names)}")
    
    # Initialize model trainer
    print("\n3. Training models...")
    trainer = HousePriceModelTrainer(random_state=42)
    
    # Train all models
    trainer.train_all_models(X_train, y_train, cv_folds=5)
    
    # Display model comparison
    print("\n4. Model Comparison Results:")
    print("-" * 40)
    comparison_df = trainer.get_model_comparison()
    print(comparison_df.round(4))
    
    # Hyperparameter tuning for best models
    print("\n5. Hyperparameter tuning...")
    
    # Get top 3 models for tuning
    top_models = comparison_df.head(3)['Model'].tolist()
    
    for model_name in top_models:
        if model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            print(f"\nTuning {model_name}...")
            trainer.hyperparameter_tuning(
                X_train, y_train, 
                model_name=model_name, 
                search_type='random',
                cv_folds=3  # Reduced for faster execution
            )
    
    # Evaluate on test set
    print("\n6. Final evaluation on test set...")
    test_results = trainer.evaluate_on_test_set(X_test, y_test, preprocessor)
    
    # Calculate improvement over baseline
    if test_results:
        # Simple baseline: mean of training targets
        baseline_pred = np.full_like(y_test, y_train.mean())
        if preprocessor.target_transform == 'log':
            baseline_pred_original = preprocessor.inverse_transform_target(baseline_pred)
            y_test_original = preprocessor.inverse_transform_target(y_test)
        else:
            baseline_pred_original = baseline_pred
            y_test_original = y_test
        
        baseline_rmse = np.sqrt(np.mean((y_test_original - baseline_pred_original) ** 2))
        improvement = ((baseline_rmse - test_results['test_rmse']) / baseline_rmse) * 100
        
        print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
        print(f"Model RMSE: {test_results['test_rmse']:.4f}")
        print(f"Improvement: {improvement:.1f}%")
        
        # Check if we achieved the 15% improvement target
        if improvement >= 15:
            print("✓ Target of 15% RMSE improvement achieved!")
        else:
            print(f"✗ Target of 15% RMSE improvement not achieved (got {improvement:.1f}%)")
    
    # Save models
    print("\n7. Saving models...")
    os.makedirs('models', exist_ok=True)
    trainer.save_models('models')
    
    # Generate and save report
    print("\n8. Generating performance report...")
    os.makedirs('results', exist_ok=True)
    report_path = os.path.join('results', 'model_performance_report.txt')
    report = trainer.generate_model_report(test_results, report_path)
    
    print("\nModel training and evaluation completed!")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Best CV RMSE: {trainer.best_score:.4f}")
    
    if test_results:
        print(f"Test RMSE: {test_results['test_rmse']:.4f}")
        print(f"Test R²: {test_results['test_r2']:.4f}")

if __name__ == "__main__":
    main()