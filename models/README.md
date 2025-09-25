# Trained Models Directory

This directory contains the trained machine learning models for the house price prediction project.

## Model Files (Generated Locally)

The following model files are generated when you run the training pipeline but are excluded from git due to their large size:

- `best_model.pkl` - The best performing model (XGBoost)
- `preprocessor.pkl` - Data preprocessing pipeline
- `model_results.pkl` - Model comparison results
- `xgboost_model.pkl` - XGBoost regression model
- `random_forest_model.pkl` - Random Forest regression model
- `gradient_boosting_model.pkl` - Gradient Boosting regression model
- `linear_regression_model.pkl` - Linear regression model
- `ridge_regression_model.pkl` - Ridge regression model
- `lasso_regression_model.pkl` - Lasso regression model
- `elasticnet_model.pkl` - ElasticNet regression model
- `decision_tree_model.pkl` - Decision Tree regression model
- `support_vector_regression_model.pkl` - SVR model
- `k-nearest_neighbors_model.pkl` - KNN regression model

## Model Performance

| Model | CV RMSE | Performance |
|-------|---------|-------------|
| XGBoost (Best) | 0.1345 | ⭐ Best |
| Random Forest | 0.1385 | Excellent |
| Gradient Boosting | 0.1420 | Excellent |
| Linear Regression | 0.2890 | Good |

## How to Generate Models

To generate these model files locally, run:

```bash
# Train all models
python src/test_models.py

# Or use the quick completion script
python src/quick_model_completion.py
```

## Model Usage

Use the production pipeline to load and use the trained models:

```python
from src.ml_pipeline import HousePricePredictionPipeline

# Initialize pipeline (automatically loads best model)
pipeline = HousePricePredictionPipeline()

# Make predictions
prediction = pipeline.predict_single(house_data)
```

## File Sizes

The model files are excluded from git because they exceed GitHub's 100MB file size limit:
- `random_forest_model.pkl`: ~138MB
- Other model files: Various sizes up to ~50MB each

## Note

These models are trained on the California Housing dataset and achieve:
- **62.1% improvement** over baseline RMSE
- **Test RMSE**: 0.4392 (original scale)
- **Test R²**: 0.8528