# House Price Prediction Project

## 🎯 Project Overview
This project implements a comprehensive machine learning regression pipeline to predict house prices. The goal is to build, compare, and optimize multiple regression models while achieving a significant reduction in RMSE (Root Mean Square Error).

## 📚 Learning Objectives
- Build end-to-end ML regression pipeline using scikit-learn and pandas
- Compare multiple regression algorithms
- Implement proper data preprocessing and feature engineering
- Apply cross-validation and hyperparameter tuning
- Achieve measurable performance improvements (target: 15% RMSE reduction)

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and tools
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization
- **plotly**: Interactive visualizations
- **xgboost**: Gradient boosting framework

## 📁 Project Structure
```
house-price-prediction/
├── data/                   # Dataset files
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── pipeline.py
├── models/                # Saved model files
├── results/               # Performance metrics and plots
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Project
```bash
python src/main.py
```

## 📊 Expected Outcomes
- Comprehensive EDA with insights about house price factors
- Multiple trained regression models (Linear, Random Forest, Gradient Boosting, XGBoost)
- Model comparison report with performance metrics
- Optimized final model with 15%+ RMSE improvement
- Production-ready prediction pipeline

## 📈 Performance Metrics
- **RMSE** (Root Mean Square Error) - Primary metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Cross-validation scores**

## 🎓 Key Learning Points
1. **Data Preprocessing**: Handling missing values, feature scaling, encoding
2. **Feature Engineering**: Creating meaningful features from raw data
3. **Model Selection**: Comparing different algorithms systematically
4. **Hyperparameter Tuning**: Optimizing model performance
5. **Pipeline Creation**: Building reusable, production-ready code