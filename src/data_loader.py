"""
Data Loader Module for House Price Prediction Project

This module handles loading and initial preparation of the California Housing dataset.
The California Housing dataset is a classic regression dataset with real-world housing data.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import os

def load_california_housing_data():
    """
    Load the California Housing dataset from scikit-learn.
    
    Returns:
        tuple: (X, y, feature_names, target_name) where:
            - X: Feature matrix (pandas DataFrame)
            - y: Target vector (pandas Series) 
            - feature_names: List of feature names
            - target_name: Name of target variable
    """
    print("Loading California Housing dataset...")
    
    # Load the dataset
    housing = fetch_california_housing()
    
    # Convert to pandas DataFrame for easier manipulation
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {y.name} (median house value in hundreds of thousands of dollars)")
    
    return X, y, housing.feature_names, 'MedHouseVal'

def save_dataset(X, y, data_dir='data'):
    """
    Save the dataset to CSV files for future use.
    
    Args:
        X: Feature matrix
        y: Target vector
        data_dir: Directory to save the data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Combine features and target
    full_data = X.copy()
    full_data['MedHouseVal'] = y
    
    # Save to CSV
    filepath = os.path.join(data_dir, 'california_housing.csv')
    full_data.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    
    return filepath

def load_dataset_from_csv(filepath):
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    data = pd.read_csv(filepath)
    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    
    return X, y

def get_dataset_info(X, y):
    """
    Display comprehensive information about the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    print(f"\nDataset Shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]:,}")
    print(f"Number of features: {X.shape[1]}")
    
    print(f"\nFeature Names and Descriptions:")
    feature_descriptions = {
        'MedInc': 'Median income in block group',
        'HouseAge': 'Median house age in block group', 
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average number of household members',
        'Latitude': 'Block group latitude',
        'Longitude': 'Block group longitude'
    }
    
    for feature in X.columns:
        desc = feature_descriptions.get(feature, 'No description available')
        print(f"  • {feature}: {desc}")
    
    print(f"\nTarget Variable:")
    print(f"  • {y.name}: Median house value (in hundreds of thousands of dollars)")
    print(f"  • Range: ${y.min()*100:.0f}k - ${y.max()*100:.0f}k")
    print(f"  • Mean: ${y.mean()*100:.0f}k")
    print(f"  • Median: ${y.median()*100:.0f}k")
    
    print(f"\nMissing Values:")
    missing_counts = X.isnull().sum()
    if missing_counts.sum() == 0:
        print("  ✓ No missing values found")
    else:
        for feature, count in missing_counts.items():
            if count > 0:
                print(f"  • {feature}: {count} missing values")

if __name__ == "__main__":
    # Load the dataset
    X, y, feature_names, target_name = load_california_housing_data()
    
    # Display dataset information
    get_dataset_info(X, y)
    
    # Save dataset for future use
    save_dataset(X, y)