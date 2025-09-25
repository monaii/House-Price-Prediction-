"""
Model Training Module for House Price Prediction

This module implements multiple regression models and provides comprehensive
model comparison and evaluation capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import time
from datetime import datetime

class HousePriceModelTrainer:
    """
    Comprehensive model trainer for house price prediction
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
        
    def initialize_models(self):
        """Initialize all regression models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(random_state=self.random_state),
            
            'Lasso Regression': Lasso(random_state=self.random_state),
            
            'ElasticNet': ElasticNet(random_state=self.random_state),
            
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Support Vector Regression': SVR(kernel='rbf'),
            
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        print(f"Initialized {len(self.models)} regression models")
        
    def evaluate_model(self, model, X_train, y_train, cv_folds=5):
        """
        Evaluate a single model using cross-validation
        
        Args:
            model: Scikit-learn model
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Evaluation metrics
        """
        # Cross-validation scores
        cv_rmse_scores = np.sqrt(-cross_val_score(
            model, X_train, y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error'
        ))
        
        cv_mae_scores = -cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring='neg_mean_absolute_error'
        )
        
        cv_r2_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring='r2'
        )
        
        return {
            'cv_rmse_mean': cv_rmse_scores.mean(),
            'cv_rmse_std': cv_rmse_scores.std(),
            'cv_mae_mean': cv_mae_scores.mean(),
            'cv_mae_std': cv_mae_scores.std(),
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std(),
            'cv_scores': {
                'rmse': cv_rmse_scores,
                'mae': cv_mae_scores,
                'r2': cv_r2_scores
            }
        }
    
    def train_all_models(self, X_train, y_train, cv_folds=5):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
        """
        print("Training and evaluating all models...")
        print("=" * 60)
        
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            try:
                # Evaluate model
                results = self.evaluate_model(model, X_train, y_train, cv_folds)
                
                # Fit model on full training data
                model.fit(X_train, y_train)
                
                # Calculate training time
                training_time = time.time() - start_time
                results['training_time'] = training_time
                
                # Store results
                self.model_results[name] = results
                
                # Check if this is the best model so far
                if results['cv_rmse_mean'] < self.best_score:
                    self.best_score = results['cv_rmse_mean']
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"  RMSE: {results['cv_rmse_mean']:.4f} (+/- {results['cv_rmse_std']:.4f})")
                print(f"  MAE:  {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
                print(f"  R²:   {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
                print(f"  Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        print(f"\nBest model: {self.best_model_name} (RMSE: {self.best_score:.4f})")
        
    def get_model_comparison(self):
        """
        Get a comparison dataframe of all model results
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.model_results:
            print("No model results available. Train models first.")
            return None
        
        comparison_data = []
        for name, results in self.model_results.items():
            comparison_data.append({
                'Model': name,
                'CV_RMSE_Mean': results['cv_rmse_mean'],
                'CV_RMSE_Std': results['cv_rmse_std'],
                'CV_MAE_Mean': results['cv_mae_mean'],
                'CV_MAE_Std': results['cv_mae_std'],
                'CV_R2_Mean': results['cv_r2_mean'],
                'CV_R2_Std': results['cv_r2_std'],
                'Training_Time': results['training_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('CV_RMSE_Mean')
        return df
    
    def evaluate_on_test_set(self, X_test, y_test, preprocessor=None):
        """
        Evaluate the best model on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            preprocessor: Preprocessor for inverse transformation
            
        Returns:
            dict: Test set evaluation results
        """
        if self.best_model is None:
            print("No trained model available. Train models first.")
            return None
        
        print(f"\nEvaluating {self.best_model_name} on test set...")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # If preprocessor is provided, inverse transform predictions and targets
        if preprocessor is not None:
            y_test_original = preprocessor.inverse_transform_target(y_test)
            y_pred_original = preprocessor.inverse_transform_target(y_pred)
        else:
            y_test_original = y_test
            y_pred_original = y_pred
        
        # Calculate metrics on original scale
        test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        test_mae = mean_absolute_error(y_test_original, y_pred_original)
        test_r2 = r2_score(y_test_original, y_pred_original)
        
        # Calculate metrics on transformed scale
        test_rmse_transformed = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae_transformed = mean_absolute_error(y_test, y_pred)
        test_r2_transformed = r2_score(y_test, y_pred)
        
        test_results = {
            'model_name': self.best_model_name,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse_transformed': test_rmse_transformed,
            'test_mae_transformed': test_mae_transformed,
            'test_r2_transformed': test_r2_transformed,
            'predictions': y_pred_original,
            'predictions_transformed': y_pred
        }
        
        print(f"Test Results (Original Scale):")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")
        
        return test_results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name=None, search_type='grid', cv_folds=5):
        """
        Perform hyperparameter tuning for specified model or best model
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of model to tune (if None, uses best model)
            search_type: 'grid' or 'random'
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best model after tuning
        """
        if model_name is None:
            if self.best_model_name is None:
                print("No best model available. Train models first.")
                return None
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Support Vector Regression': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        base_model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv_folds,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv_folds,
                scoring='neg_mean_squared_error', n_jobs=-1,
                n_iter=20, random_state=self.random_state
            )
        
        # Perform search
        start_time = time.time()
        search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        # Get best model
        best_tuned_model = search.best_estimator_
        best_score = np.sqrt(-search.best_score_)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV RMSE: {best_score:.4f}")
        print(f"Tuning time: {tuning_time:.2f}s")
        
        # Update best model if this is better
        if best_score < self.best_score:
            self.best_model = best_tuned_model
            self.best_model_name = f"{model_name} (Tuned)"
            self.best_score = best_score
            print(f"New best model: {self.best_model_name}")
        
        return best_tuned_model
    
    def save_models(self, models_dir='models'):
        """Save all trained models"""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save all models
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            filepath = os.path.join(models_dir, filename)
            joblib.dump(model, filepath)
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = os.path.join(models_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
        
        # Save model results
        results_path = os.path.join(models_dir, 'model_results.pkl')
        joblib.dump(self.model_results, results_path)
        
        print(f"Models saved to {models_dir}/")
    
    def generate_model_report(self, test_results=None, save_path=None):
        """
        Generate comprehensive model performance report
        
        Args:
            test_results: Test set evaluation results
            save_path: Path to save the report
            
        Returns:
            str: Report content
        """
        report = []
        report.append("HOUSE PRICE PREDICTION - MODEL PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model comparison
        if self.model_results:
            report.append("MODEL COMPARISON (Cross-Validation Results)")
            report.append("-" * 50)
            
            comparison_df = self.get_model_comparison()
            report.append(comparison_df.to_string(index=False))
            report.append("")
        
        # Best model details
        if self.best_model_name:
            report.append(f"BEST MODEL: {self.best_model_name}")
            report.append("-" * 30)
            best_results = self.model_results.get(self.best_model_name, {})
            
            cv_rmse = best_results.get('cv_rmse_mean', 'N/A')
            cv_mae = best_results.get('cv_mae_mean', 'N/A')
            cv_r2 = best_results.get('cv_r2_mean', 'N/A')
            
            if cv_rmse != 'N/A':
                report.append(f"Cross-Validation RMSE: {cv_rmse:.4f}")
            else:
                report.append(f"Cross-Validation RMSE: {cv_rmse}")
                
            if cv_mae != 'N/A':
                report.append(f"Cross-Validation MAE:  {cv_mae:.4f}")
            else:
                report.append(f"Cross-Validation MAE:  {cv_mae}")
                
            if cv_r2 != 'N/A':
                report.append(f"Cross-Validation R²:   {cv_r2:.4f}")
            else:
                report.append(f"Cross-Validation R²:   {cv_r2}")
            report.append("")
        
        # Test results
        if test_results:
            report.append("TEST SET PERFORMANCE")
            report.append("-" * 25)
            report.append(f"Model: {test_results['model_name']}")
            report.append(f"Test RMSE: {test_results['test_rmse']:.4f}")
            report.append(f"Test MAE:  {test_results['test_mae']:.4f}")
            report.append(f"Test R²:   {test_results['test_r2']:.4f}")
            report.append("")
        
        report_content = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            print(f"Report saved to {save_path}")
        
        return report_content