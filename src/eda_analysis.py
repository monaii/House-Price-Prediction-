"""
Exploratory Data Analysis Script for House Price Prediction Project

This script performs comprehensive EDA on the California Housing dataset
and generates insights for feature engineering and model selection.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load dataset and perform initial exploration."""
    from data_loader import load_dataset_from_csv
    import os
    
    print("🔍 Loading California Housing Dataset...")
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'california_housing.csv')
    
    X, y = load_dataset_from_csv(data_path)
    
    # Combine for easier analysis
    df = X.copy()
    df['MedHouseVal'] = y
    
    print(f"✅ Dataset loaded: {df.shape}")
    return df

def basic_statistics(df):
    """Generate basic statistical summary."""
    print("\n" + "="*60)
    print("📊 BASIC STATISTICAL SUMMARY")
    print("="*60)
    
    print("\n📋 Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        print(missing[missing > 0])

def analyze_target_variable(df):
    """Analyze the target variable distribution."""
    print("\n" + "="*60)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    target = df['MedHouseVal']
    
    # Basic statistics
    print(f"\n📊 House Value Statistics:")
    print(f"   Mean: ${target.mean()*100:.0f}k")
    print(f"   Median: ${target.median()*100:.0f}k")
    print(f"   Std Dev: ${target.std()*100:.0f}k")
    print(f"   Min: ${target.min()*100:.0f}k")
    print(f"   Max: ${target.max()*100:.0f}k")
    print(f"   Skewness: {target.skew():.3f}")
    print(f"   Kurtosis: {target.kurtosis():.3f}")
    
    # Distribution analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of House Values')
    plt.xlabel('Median House Value ($100k)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.boxplot(target)
    plt.title('Box Plot of House Values')
    plt.ylabel('Median House Value ($100k)')
    
    plt.subplot(1, 3, 3)
    log_values = np.log(target)
    plt.hist(log_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Log-Transformed House Values')
    plt.xlabel('Log(Median House Value)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    # Save with absolute path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'results', 'target_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Target analysis plot saved to {save_path}")
    
    return target

def correlation_analysis(df):
    """Perform correlation analysis."""
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # Save with absolute path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'results', 'correlation_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Correlation matrix plot saved to {save_path}")
    
    # Top correlations with target
    target_corr = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False)
    print("\n🎯 Features most correlated with House Value:")
    print("-" * 45)
    for feature, corr in target_corr.items():
        if feature != 'MedHouseVal':
            print(f"   {feature:<12}: {corr:.3f}")
    
    return corr_matrix

def feature_distributions(df):
    """Analyze feature distributions."""
    print("\n" + "="*60)
    print("📊 FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        
        # Histogram with KDE
        axes[row, col].hist(df[feature], bins=50, alpha=0.7, density=True, color='lightblue')
        
        # Add KDE curve
        kde = gaussian_kde(df[feature])
        x_range = np.linspace(df[feature].min(), df[feature].max(), 100)
        axes[row, col].plot(x_range, kde(x_range), 'r-', linewidth=2)
        
        axes[row, col].set_title(f'{feature}')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Density')
        
        # Add statistics
        mean_val = df[feature].mean()
        axes[row, col].axvline(mean_val, color='red', linestyle='--', alpha=0.7)
        
        # Print statistics
        print(f"\n{feature}:")
        print(f"   Mean: {mean_val:.2f}")
        print(f"   Std: {df[feature].std():.2f}")
        print(f"   Skewness: {df[feature].skew():.3f}")
    
    plt.tight_layout()
    # Save with absolute path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'results', 'feature_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Feature distributions plot saved to {save_path}")

def geographic_analysis(df):
    """Analyze geographic patterns."""
    print("\n" + "="*60)
    print("🗺️ GEOGRAPHIC ANALYSIS")
    print("="*60)
    
    plt.figure(figsize=(15, 10))
    
    # Create scatter plot with house values as color
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                         c=df['MedHouseVal'], 
                         cmap='viridis', 
                         alpha=0.6, 
                         s=20)
    
    plt.colorbar(scatter, label='Median House Value ($100k)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of House Values in California', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save with absolute path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'results', 'geographic_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Geographic analysis plot saved to {save_path}")
    
    # Geographic insights
    coastal_high = df[(df['Longitude'] > -122) & (df['MedHouseVal'] > 3)]
    inland_low = df[(df['Longitude'] < -120) & (df['MedHouseVal'] < 2)]
    
    print(f"\n🏖️ Coastal Analysis:")
    print(f"   High-value coastal properties: {len(coastal_high)}")
    print(f"   Average coastal value: ${coastal_high['MedHouseVal'].mean()*100:.0f}k")
    
    print(f"\n🏔️ Inland Analysis:")
    print(f"   Lower-value inland properties: {len(inland_low)}")
    print(f"   Average inland value: ${inland_low['MedHouseVal'].mean()*100:.0f}k")

def outlier_analysis(df):
    """Detect and analyze outliers."""
    print("\n" + "="*60)
    print("🔍 OUTLIER ANALYSIS")
    print("="*60)
    
    def detect_outliers_iqr(data, feature):
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    outlier_summary = {}
    numeric_features = df.select_dtypes(include=[np.number]).columns
    
    print("\n📊 Outlier Summary:")
    print("-" * 40)
    
    for feature in numeric_features:
        outliers, lower, upper = detect_outliers_iqr(df, feature)
        outlier_percentage = (len(outliers) / len(df)) * 100
        outlier_summary[feature] = {
            'count': len(outliers),
            'percentage': outlier_percentage,
            'lower_bound': lower,
            'upper_bound': upper
        }
        
        print(f"{feature:<12}: {len(outliers):>4} outliers ({outlier_percentage:>5.1f}%)")
    
    # Visualize outliers
    n_features = len(numeric_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        axes_flat[i].boxplot(df[feature])
        axes_flat[i].set_title(f'{feature}')
        axes_flat[i].set_ylabel('Value')
        
        # Add outlier count
        outlier_count = outlier_summary[feature]['count']
        axes_flat[i].text(0.5, 0.95, f'Outliers: {outlier_count}', 
                           transform=axes_flat[i].transAxes, 
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    # Save with absolute path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'results', 'outlier_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Outlier analysis plot saved to {save_path}")
    
    return outlier_summary

def feature_relationships(df):
    """Analyze relationships between features and target."""
    print("\n" + "="*60)
    print("📈 FEATURE-TARGET RELATIONSHIPS")
    print("="*60)
    
    features_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Relationships with House Values', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(features_to_plot):
        row = i // 3
        col = i % 3
        
        # Scatter plot with regression line
        axes[row, col].scatter(df[feature], df['MedHouseVal'], alpha=0.5, s=10)
        
        # Add regression line
        z = np.polyfit(df[feature], df['MedHouseVal'], 1)
        p = np.poly1d(z)
        axes[row, col].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
        
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Median House Value')
        axes[row, col].set_title(f'{feature} vs House Value')
        
        # Add correlation coefficient
        corr = df[feature].corr(df['MedHouseVal'])
        axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                           transform=axes[row, col].transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    # Save with absolute path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'results', 'feature_relationships.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Feature relationships plot saved to {save_path}")

def generate_insights(df, corr_matrix, outlier_summary):
    """Generate key insights and recommendations."""
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    # Target variable insights
    print("\n📊 TARGET VARIABLE (House Values):")
    skewness = df['MedHouseVal'].skew()
    print(f"   • Distribution is {'right' if skewness > 0 else 'left'}-skewed (skewness: {skewness:.3f})")
    print(f"   • Range: ${df['MedHouseVal'].min()*100:.0f}k - ${df['MedHouseVal'].max()*100:.0f}k")
    print(f"   • Mean > Median indicates positive skew" if df['MedHouseVal'].mean() > df['MedHouseVal'].median() else "   • Mean < Median indicates negative skew")
    print(f"   • Consider log transformation for modeling")
    
    # Feature correlations
    print("\n🔗 STRONGEST PREDICTORS:")
    target_corr_sorted = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False)
    for i, (feature, corr) in enumerate(target_corr_sorted.items()):
        if feature != 'MedHouseVal' and i <= 3:
            print(f"   • {feature}: {corr:.3f} correlation")
    
    # Geographic insights
    print("\n🗺️ GEOGRAPHIC PATTERNS:")
    coastal_high = df[(df['Longitude'] > -122) & (df['MedHouseVal'] > 3)]
    print(f"   • Coastal areas show higher property values")
    print(f"   • {len(coastal_high)} high-value coastal properties identified")
    print(f"   • Location features (Lat/Long) are important predictors")
    
    # Outlier insights
    print("\n🔍 OUTLIER ANALYSIS:")
    high_outlier_features = [f for f, info in outlier_summary.items() if info['percentage'] > 5]
    for feature in high_outlier_features[:3]:
        pct = outlier_summary[feature]['percentage']
        print(f"   • {feature}: {pct:.1f}% outliers - consider capping or transformation")
    
    # Feature engineering recommendations
    print("\n🛠️ FEATURE ENGINEERING RECOMMENDATIONS:")
    print("   • Create 'RoomsPerHousehold' = AveRooms / AveOccup")
    print("   • Create 'BedroomRatio' = AveBedrms / AveRooms")
    print("   • Create 'PopulationDensity' feature")
    print("   • Consider polynomial features for MedInc (strongest predictor)")
    print("   • Geographic clustering based on Lat/Long")
    
    # Preprocessing recommendations
    print("\n⚙️ PREPROCESSING RECOMMENDATIONS:")
    print("   • Apply StandardScaler or RobustScaler for feature scaling")
    print("   • Consider log transformation for target variable")
    print("   • Handle outliers using IQR-based capping")
    print("   • No missing values - dataset is clean!")
    
    print("\n" + "="*80)
    print("✅ EDA COMPLETE - Ready for preprocessing and modeling!")

def main():
    """Main function to run complete EDA."""
    # Create results directory
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("🚀 Starting Comprehensive Exploratory Data Analysis...")
    
    # Load data
    df = load_and_explore_data()
    
    # Perform analysis
    basic_statistics(df)
    target = analyze_target_variable(df)
    corr_matrix = correlation_analysis(df)
    feature_distributions(df)
    geographic_analysis(df)
    outlier_summary = outlier_analysis(df)
    feature_relationships(df)
    
    # Generate insights
    generate_insights(df, corr_matrix, outlier_summary)
    
    print("\n🎉 EDA Analysis Complete! Check the results/ folder for visualizations.")

if __name__ == "__main__":
    main()