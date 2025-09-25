# House Price Prediction Project

A comprehensive machine learning project for predicting California house prices using the California Housing dataset.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline to predict house prices with the goal of achieving at least 15% improvement over baseline RMSE. The project successfully achieved **62.1% improvement** over baseline.

## 📊 Final Results

- **RMSE**: 0.4392 (Original Scale)
- **MAE**: 0.2793
- **R²**: 0.8528
- **Improvement over Baseline**: 62.1% ✓
- **Target Achievement**: YES (exceeded 15% target)

## 🏗️ Project Structure

```
house/
├── data/
│   └── california_housing.csv          # Dataset
├── src/
│   ├── data_loader.py                  # Data loading utilities
│   ├── data_preprocessor.py            # Data preprocessing pipeline
│   ├── eda_analyzer.py                 # Exploratory data analysis
│   ├── model_trainer.py                # Model training and evaluation
│   ├── ml_pipeline.py                  # Production prediction pipeline
│   └── test_*.py                       # Test scripts
├── models/
│   ├── preprocessor.pkl                # Trained preprocessor
│   ├── best_model.pkl                  # Best performing model
│   └── *.pkl                          # Individual model files
├── results/
│   ├── eda_visualizations.png          # EDA plots
│   └── final_project_report.txt        # Final performance report
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Running the Complete Pipeline

1. **Load and explore data:**
```bash
python src/test_eda.py
```

2. **Preprocess data:**
```bash
python src/test_preprocessing.py
```

3. **Train and evaluate models:**
```bash
python src/test_models.py
```

4. **Use production pipeline:**
```python
from src.ml_pipeline import HousePricePredictionPipeline

# Initialize pipeline
pipeline = HousePricePredictionPipeline()

# Make prediction
sample_data = {
    'MedInc': 5.0, 'HouseAge': 10.0, 'AveRooms': 6.0,
    'AveBedrms': 1.2, 'Population': 3000.0, 'AveOccup': 3.0,
    'Latitude': 34.0, 'Longitude': -118.0
}
prediction = pipeline.predict_single(sample_data)
print(f"Predicted price: ${prediction:.2f}")
```

## 🔧 Features

### Data Preprocessing
- **Feature Engineering**: 12 new features including polynomial terms, ratios, and density metrics
- **Outlier Handling**: IQR-based outlier detection and capping
- **Scaling**: StandardScaler for feature normalization
- **Target Transformation**: Log transformation for better model performance

### Models Implemented
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost (Best Performer)
- Support Vector Regression
- K-Nearest Neighbors

### Model Selection
- Cross-validation for model comparison
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Automated best model selection based on RMSE

## 📈 Model Performance

| Model | CV RMSE | Test RMSE | R² Score |
|-------|---------|-----------|----------|
| XGBoost (Best) | 0.1345 | 0.1298 | 0.8662 |
| Random Forest | 0.1385 | - | - |
| Gradient Boosting | 0.1420 | - | - |
| Linear Regression | 0.2890 | - | - |

## 🛠️ Technical Implementation

### Key Components

1. **HousePricePreprocessor**: Handles all data preprocessing steps
2. **HousePriceModelTrainer**: Manages model training and evaluation
3. **HousePricePredictionPipeline**: Production-ready prediction interface
4. **EDAAnalyzer**: Comprehensive exploratory data analysis

### Production Features
- Model persistence with joblib
- Input validation and error handling
- Batch prediction capabilities
- Feature importance analysis
- Prediction explanations

## 📊 Data Insights

- **Dataset**: 20,640 California housing records
- **Features**: 8 original + 12 engineered = 20 total features
- **Target**: Median house value (log-transformed)
- **Split**: 80% training, 20% testing

### Key Findings from EDA
- Strong correlation between median income and house values
- Geographic clustering of high-value properties
- Age and room ratios significantly impact pricing
- Population density affects house values

## 🎯 Project Achievements

✅ **Data Loading & EDA**: Complete analysis with visualizations  
✅ **Data Preprocessing**: Advanced feature engineering and cleaning  
✅ **Model Training**: 10 different algorithms implemented  
✅ **Model Evaluation**: Comprehensive performance analysis  
✅ **Production Pipeline**: Ready-to-use prediction interface  
✅ **Target Achievement**: 62.1% RMSE improvement (exceeded 15% goal)  

## 📝 Usage Examples

### Making Predictions

```python
from src.ml_pipeline import HousePricePredictionPipeline

# Initialize pipeline
pipeline = HousePricePredictionPipeline()

# Single prediction
house_data = {
    'MedInc': 8.0, 'HouseAge': 5.0, 'AveRooms': 7.0,
    'AveBedrms': 1.0, 'Population': 2500.0, 'AveOccup': 2.8,
    'Latitude': 37.8, 'Longitude': -122.4
}
price = pipeline.predict_single(house_data)
print(f"Predicted price: ${price:.2f}")

# Batch predictions
import pandas as pd
df = pd.DataFrame([house_data])  # Can contain multiple rows
predictions = pipeline.predict_batch(df)
```

### Feature Importance

```python
importance = pipeline.get_feature_importance()
print("Top 5 most important features:")
for feature, score in importance[:5]:
    print(f"{feature}: {score:.4f}")
```

## 🔍 Model Interpretability

The project includes feature importance analysis and prediction explanations to understand model decisions:

- **Feature Importance**: Identifies which features most influence predictions
- **Prediction Explanations**: Provides insights into individual predictions
- **Model Comparison**: Detailed performance metrics across all models

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- xgboost >= 1.5.0

## 🤝 Contributing

This project demonstrates a complete ML workflow from data exploration to production deployment. Key learning outcomes include:

- End-to-end ML pipeline development
- Advanced feature engineering techniques
- Model comparison and selection strategies
- Production-ready code organization
- Performance optimization and evaluation

## 📄 License

This project is for educational and demonstration purposes.

---

**Project Status**: ✅ COMPLETED  
**Performance Target**: ✅ EXCEEDED (62.1% vs 15% target)  
**Production Ready**: ✅ YES