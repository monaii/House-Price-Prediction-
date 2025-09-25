"""
Test script for data preprocessing pipeline
"""

import pandas as pd
import numpy as np
from data_preprocessor import HousePricePreprocessor
from data_loader import load_dataset_from_csv
import os

def main():
    """Test the preprocessing pipeline"""
    print("Testing Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Load data
    print("Loading California Housing dataset...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'california_housing.csv')
    X, y = load_dataset_from_csv(data_path)
    
    # Combine features and target into single dataframe for preprocessing
    df = pd.concat([X, y], axis=1)
    print(f"Original dataset shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = HousePricePreprocessor(
        outlier_method='iqr',
        scaler_type='standard',
        target_transform='log'
    )
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    # Display results
    print("\nPreprocessing Results:")
    print("-" * 30)
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    print(f"\nFeature names ({len(preprocessor.feature_names)}):")
    for i, feature in enumerate(preprocessor.feature_names):
        print(f"  {i+1:2d}. {feature}")
    
    print(f"\nTarget transformation applied: {preprocessor.target_log_transform}")
    print(f"Scaler type: {preprocessor.scaler_type}")
    
    # Show outlier handling summary
    if preprocessor.outlier_bounds:
        print(f"\nOutlier bounds applied:")
        for feature, bounds in preprocessor.outlier_bounds.items():
            print(f"  {feature}: {bounds['outliers_capped']} outliers capped")
    
    # Save preprocessor
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    preprocessor.save_preprocessor(preprocessor_path)
    
    # Test inverse transformation
    print(f"\nTesting inverse transformation...")
    sample_pred = y_train[:5].values
    original_scale = preprocessor.inverse_transform_target(sample_pred)
    print(f"Sample transformed targets: {sample_pred}")
    print(f"Back to original scale: {original_scale}")
    
    print("\nPreprocessing pipeline test completed successfully!")

if __name__ == "__main__":
    main()