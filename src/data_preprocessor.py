"""
Data Preprocessing Module for House Price Prediction

This module implements comprehensive data preprocessing based on EDA insights:
- Feature engineering (new features creation)
- Outlier handling using IQR method
- Feature scaling and normalization
- Target variable transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import joblib

class HousePricePreprocessor:
    """
    Comprehensive preprocessor for California Housing dataset
    """
    
    def __init__(self, outlier_method='iqr', scaler_type='standard', target_transform='log'):
        """
        Initialize the preprocessor
        
        Args:
            outlier_method (str): Method for outlier handling ('iqr', 'none')
            scaler_type (str): Type of scaler ('standard', 'robust')
            target_transform (str): Target transformation ('log', 'none')
        """
        self.outlier_method = outlier_method
        self.scaler_type = scaler_type
        self.target_transform = target_transform
        self.scaler = None
        self.outlier_bounds = {}
        self.feature_names = None
        self.target_log_transform = False
        
    def create_engineered_features(self, df):
        """
        Create new features based on EDA insights
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df_eng = df.copy()
        
        # 1. Rooms per household
        df_eng['RoomsPerHousehold'] = df_eng['AveRooms'] / df_eng['AveOccup']
        
        # 2. Bedroom ratio
        df_eng['BedroomRatio'] = df_eng['AveBedrms'] / df_eng['AveRooms']
        
        # 3. Population density (people per room)
        df_eng['PopulationDensity'] = df_eng['Population'] / df_eng['AveRooms']
        
        # 4. Income per room (economic indicator)
        df_eng['IncomePerRoom'] = df_eng['MedInc'] / df_eng['AveRooms']
        
        # 5. Geographic features
        # Distance from center of California (approximate)
        ca_center_lat, ca_center_lon = 36.7783, -119.4179
        df_eng['DistanceFromCenter'] = np.sqrt(
            (df_eng['Latitude'] - ca_center_lat)**2 + 
            (df_eng['Longitude'] - ca_center_lon)**2
        )
        
        # 6. Coastal proximity (based on longitude - more negative = closer to coast)
        df_eng['CoastalProximity'] = -df_eng['Longitude']  # Higher values = closer to coast
        
        # 7. Polynomial features for strongest predictor (MedInc)
        df_eng['MedInc_squared'] = df_eng['MedInc'] ** 2
        df_eng['MedInc_cubed'] = df_eng['MedInc'] ** 3
        
        # 8. Age categories
        df_eng['HouseAge_category'] = pd.cut(df_eng['HouseAge'], 
                                           bins=[0, 10, 25, 40, np.inf], 
                                           labels=['New', 'Modern', 'Mature', 'Old'])
        
        # Convert categorical to dummy variables
        age_dummies = pd.get_dummies(df_eng['HouseAge_category'], prefix='Age')
        df_eng = pd.concat([df_eng, age_dummies], axis=1)
        df_eng.drop('HouseAge_category', axis=1, inplace=True)
        
        print(f"Feature engineering complete. Added {len(df_eng.columns) - len(df.columns)} new features.")
        print(f"Total features: {len(df_eng.columns)}")
        
        return df_eng
    
    def handle_outliers(self, df, target_col='MedHouseVal'):
        """
        Handle outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        if self.outlier_method == 'none':
            return df
            
        df_clean = df.copy()
        outlier_summary = {}
        
        # Features to apply outlier capping (based on EDA insights)
        features_to_cap = ['AveBedrms', 'Population', 'AveOccup', 'RoomsPerHousehold', 
                          'PopulationDensity', 'MedInc_squared', 'MedInc_cubed']
        
        for feature in features_to_cap:
            if feature in df_clean.columns:
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers before capping
                outliers_before = len(df_clean[(df_clean[feature] < lower_bound) | 
                                             (df_clean[feature] > upper_bound)])
                
                # Cap outliers
                df_clean[feature] = df_clean[feature].clip(lower=lower_bound, upper=upper_bound)
                
                # Store bounds for future use
                self.outlier_bounds[feature] = {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'outliers_capped': outliers_before
                }
                
                outlier_summary[feature] = outliers_before
        
        print(f"Outlier handling complete:")
        for feature, count in outlier_summary.items():
            print(f"  {feature}: {count} outliers capped")
            
        return df_clean
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using specified scaler
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: Scaled training and test features
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def transform_target(self, y_train, y_test=None):
        """
        Transform target variable
        
        Args:
            y_train (pd.Series): Training target
            y_test (pd.Series, optional): Test target
            
        Returns:
            tuple: Transformed training and test targets
        """
        if self.target_transform == 'log':
            self.target_log_transform = True
            y_train_transformed = np.log1p(y_train)  # log(1 + x) to handle zeros
            
            if y_test is not None:
                y_test_transformed = np.log1p(y_test)
                return y_train_transformed, y_test_transformed
            
            return y_train_transformed
        
        return y_train, y_test
    
    def inverse_transform_target(self, y_pred):
        """
        Inverse transform predictions back to original scale
        
        Args:
            y_pred (array-like): Predictions to inverse transform
            
        Returns:
            array-like: Inverse transformed predictions
        """
        if self.target_log_transform:
            return np.expm1(y_pred)  # exp(x) - 1
        return y_pred
    
    def fit_transform(self, df, target_col='MedHouseVal', test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Test set size
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test (all preprocessed)
        """
        print("Starting comprehensive data preprocessing...")
        print(f"Original dataset shape: {df.shape}")
        
        # 1. Feature Engineering
        print("\n1. Feature Engineering...")
        df_engineered = self.create_engineered_features(df)
        
        # 2. Handle outliers
        print("\n2. Outlier Handling...")
        df_clean = self.handle_outliers(df_engineered, target_col)
        
        # 3. Separate features and target
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # 4. Train-test split
        print(f"\n3. Train-Test Split (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # 5. Feature scaling
        print(f"\n4. Feature Scaling ({self.scaler_type})...")
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # 6. Target transformation
        print(f"\n5. Target Transformation ({self.target_transform})...")
        y_train_transformed, y_test_transformed = self.transform_target(y_train, y_test)
        
        print("\nPreprocessing complete!")
        print(f"Final feature count: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train_transformed, y_test_transformed
    
    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor"""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def transform_features_only(self, df):
        """
        Transform features without splitting into train/test
        Used for prediction pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe with features and target
            
        Returns:
            np.ndarray: Transformed features
        """
        # Create engineered features
        df_eng = self.create_engineered_features(df)
        
        # Separate features and target
        target_col = 'MedHouseVal'
        X = df_eng.drop(target_col, axis=1)
        
        # Handle outliers (using existing bounds if available)
        if self.outlier_bounds:
            X = self._apply_outlier_bounds(X)
        
        # Scale features (using existing scaler if available)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return X_scaled
    
    def _apply_outlier_bounds(self, X):
        """Apply existing outlier bounds to new data"""
        X_capped = X.copy()
        for feature, bounds in self.outlier_bounds.items():
            if feature in X_capped.columns:
                lower_bound, upper_bound = bounds
                X_capped[feature] = X_capped[feature].clip(lower=lower_bound, upper=upper_bound)
        return X_capped
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load a saved preprocessor"""
        return joblib.load(filepath)