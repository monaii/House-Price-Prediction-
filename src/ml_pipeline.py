"""
Production-Ready ML Pipeline for House Price Prediction

This module provides a complete pipeline for making house price predictions
using the trained models and preprocessors.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictionPipeline:
    """
    Complete ML pipeline for house price prediction
    """
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize the prediction pipeline
        
        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor file
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
        # Default paths
        if model_path is None:
            model_path = os.path.join('models', 'best_model.pkl')
        if preprocessor_path is None:
            preprocessor_path = os.path.join('models', 'preprocessor.pkl')
            
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
    def load_pipeline(self):
        """Load the trained model and preprocessor"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                self.feature_names = self.preprocessor.feature_names
                print(f"Preprocessor loaded from {self.preprocessor_path}")
            else:
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            self.is_loaded = True
            print("Pipeline loaded successfully!")
            
        except Exception as e:
            print(f"Error loading pipeline: {str(e)}")
            self.is_loaded = False
            
    def validate_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and prepare input data
        
        Args:
            data: Input data as dictionary or DataFrame
            
        Returns:
            pd.DataFrame: Validated input data
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
        
        # Required original features for California Housing dataset
        required_features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        # Check for required features
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Ensure correct data types
        for feature in required_features:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        # Check for NaN values
        if data.isnull().any().any():
            raise ValueError("Input data contains NaN values")
        
        return data[required_features]
    
    def predict_single(self, house_data: Dict) -> Dict:
        """
        Make prediction for a single house
        
        Args:
            house_data: Dictionary with house features
            
        Returns:
            Dict: Prediction results
        """
        # Validate input
        df = self.validate_input(house_data)
        
        # Add dummy target column for preprocessing
        df['MedHouseVal'] = 0  # Will be ignored during prediction
        
        # Apply preprocessing (without splitting)
        X_processed = self.preprocessor.transform_features_only(df)
        
        # Make prediction
        prediction_transformed = self.model.predict(X_processed)
        
        # Inverse transform to get original scale
        prediction_original = self.preprocessor.inverse_transform_target(prediction_transformed)
        
        return {
            'predicted_price': float(prediction_original[0]),
            'predicted_price_log': float(prediction_transformed[0]),
            'input_features': house_data,
            'confidence': 'high' if abs(prediction_transformed[0]) < 2 else 'medium'
        }
    
    def predict_batch(self, houses_data: Union[List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Make predictions for multiple houses
        
        Args:
            houses_data: List of dictionaries or DataFrame with house features
            
        Returns:
            pd.DataFrame: Prediction results
        """
        if isinstance(houses_data, list):
            df = pd.DataFrame(houses_data)
        else:
            df = houses_data.copy()
        
        # Validate input
        df_validated = self.validate_input(df)
        
        # Add dummy target column
        df_validated['MedHouseVal'] = 0
        
        # Apply preprocessing
        X_processed = self.preprocessor.transform_features_only(df_validated)
        
        # Make predictions
        predictions_transformed = self.model.predict(X_processed)
        predictions_original = self.preprocessor.inverse_transform_target(predictions_transformed)
        
        # Create results DataFrame
        results = df_validated.drop('MedHouseVal', axis=1).copy()
        results['predicted_price'] = predictions_original
        results['predicted_price_log'] = predictions_transformed
        results['confidence'] = ['high' if abs(p) < 2 else 'medium' for p in predictions_transformed]
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model (if available)
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        # Check if model has feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            feature_names = self.feature_names
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("Model does not support feature importance")
            return None
    
    def explain_prediction(self, house_data: Dict) -> Dict:
        """
        Provide explanation for a prediction
        
        Args:
            house_data: Dictionary with house features
            
        Returns:
            Dict: Prediction with explanation
        """
        # Get basic prediction
        prediction_result = self.predict_single(house_data)
        
        # Add feature importance if available
        feature_importance = self.get_feature_importance()
        
        # Create explanation
        explanation = {
            'prediction': prediction_result,
            'model_type': type(self.model).__name__,
            'key_factors': None
        }
        
        if feature_importance is not None:
            top_features = feature_importance.head(5)
            explanation['key_factors'] = top_features.to_dict('records')
        
        return explanation

def create_sample_input() -> Dict:
    """Create a sample input for testing"""
    return {
        'MedInc': 5.0,      # Median income in block group
        'HouseAge': 10.0,   # Median house age in block group
        'AveRooms': 6.0,    # Average number of rooms per household
        'AveBedrms': 1.2,   # Average number of bedrooms per household
        'Population': 3000, # Block group population
        'AveOccup': 3.0,    # Average number of household members
        'Latitude': 34.0,   # Block group latitude
        'Longitude': -118.0 # Block group longitude
    }

def main():
    """Demo function showing how to use the pipeline"""
    print("House Price Prediction Pipeline Demo")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = HousePricePredictionPipeline()
    
    # Load models
    pipeline.load_pipeline()
    
    if not pipeline.is_loaded:
        print("Failed to load pipeline. Make sure models are trained first.")
        return
    
    # Create sample data
    sample_house = create_sample_input()
    print(f"\nSample house data:")
    for key, value in sample_house.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    try:
        result = pipeline.predict_single(sample_house)
        print(f"\nPrediction Results:")
        print(f"  Predicted Price: ${result['predicted_price']:,.2f}")
        print(f"  Confidence: {result['confidence']}")
        
        # Get explanation
        explanation = pipeline.explain_prediction(sample_house)
        print(f"\nModel: {explanation['model_type']}")
        
        if explanation['key_factors']:
            print("\nTop 5 Important Features:")
            for factor in explanation['key_factors']:
                print(f"  {factor['feature']}: {factor['importance']:.4f}")
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()