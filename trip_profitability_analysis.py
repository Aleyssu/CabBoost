import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
import warnings
import gc  # For garbage collection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib  # For saving models
import os  # For directory operations
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def load_data(filepath, sample_size=None):
    """Load the taxi trip data from parquet file with optional sampling"""
    print("Loading data...")
    
    # If sample_size is provided, use it to limit data size
    if sample_size:
        # Read with sampling to limit memory usage
        df = pd.read_parquet(filepath, engine='pyarrow')
        # Random sampling
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        # Set a reasonable default sample size to avoid OOM
        df = pd.read_parquet(filepath, engine='pyarrow')
        df = df.sample(n=min(500000, len(df)), random_state=42)
    
    print(f"Data loaded! Shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data for regression analysis"""
    print("Preprocessing data...")
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['tip_amount', 'trip_distance', 'PULocationID', 'fare_amount'])
    
    # Create datetime features
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (x >= 7 and x <= 10) or (x >= 16 and x <= 19) else 0)
    
    # Create season from month
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    df['season'] = df['month'].map(season_map)
    
    # Create a profitability metric (fare + tip)
    df['total_profit'] = df['fare_amount'] + df['tip_amount']
    
    # Add trip efficiency features
    df['profit_per_mile'] = df.apply(lambda x: x['total_profit'] / x['trip_distance'] if x['trip_distance'] > 0 else 0, axis=1)
    
    # Add distance squared for non-linear relationships
    df['trip_distance_squared'] = df['trip_distance'] ** 2
    
    # Create trip length categories 
    bins = [0, 2, 5, 10, 20, float('inf')]
    labels = ['very_short', 'short', 'medium', 'long', 'very_long']
    df['trip_length_category'] = pd.cut(df['trip_distance'], bins=bins, labels=labels)
    
    # Handle outliers in total_profit (remove top and bottom 1%)
    low_bound = df['total_profit'].quantile(0.01)
    high_bound = df['total_profit'].quantile(0.99)
    df = df[(df['total_profit'] >= low_bound) & (df['total_profit'] <= high_bound)]
    
    # Free up memory by removing unused columns
    columns_to_keep = [
        'PULocationID', 'DOLocationID', 'trip_distance', 'passenger_count',
        'fare_amount', 'tip_amount', 'total_profit', 'profit_per_mile',
        'trip_distance_squared', 'trip_length_category', 'hour', 'day',
        'month', 'year', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'season', 'payment_type'
    ]
    
    df = df[columns_to_keep].copy()
    gc.collect()  # Force garbage collection
    
    print("Preprocessing complete!")
    return df

def analyze_top_regions(df, top_n=5):
    """Analyze the top n most frequent pickup regions"""
    print(f"Analyzing top {top_n} regions by number of trips...")
    
    # Get the top pickup locations by frequency
    top_regions = df['PULocationID'].value_counts().nlargest(top_n).index.tolist()
    print(f"Top {top_n} pickup regions: {top_regions}")
    
    return top_regions

def build_regression_model(df, region_id, max_samples=50000):
    """Build and evaluate a regression model for a specific region using cross-validation"""
    print(f"Building regression model for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # If we have too many samples, take a random subset to avoid memory issues
    if len(region_df) > max_samples:
        print(f"Sampling {max_samples} records from {len(region_df)} for region {region_id}")
        region_df = region_df.sample(n=max_samples, random_state=42)
    
    # Features to use in the model
    numerical_features = [
        'trip_distance', 'passenger_count', 'trip_distance_squared',
        'hour', 'day', 'day_of_week', 'is_weekend', 'is_rush_hour'
    ]
    categorical_features = ['month', 'year', 'payment_type', 'season']
    
    # Define the model pipeline with imputation steps for missing values
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Create a simpler model to avoid memory issues
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=50,  # Reduced from 100
            learning_rate=0.1,
            max_depth=3,      # Reduced from 5
            subsample=0.8,    # Use subset of samples
            random_state=42
        ))
    ])
    
    # Prepare features and target
    X = region_df[numerical_features + categorical_features]
    y = region_df['total_profit']
    
    # Drop rows with NaN in target variable
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Initial train-test split to create a holdout test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data split sizes - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    model.fit(X_train, y_train)
    
    # Cross-validation - fix the index alignment issue by resampling from the array directly
    if len(X_train) > 10000:
        # Use simple cross-validation without sampling to avoid index issues
        cv_scores = cross_val_score(model, X_train.head(10000), y_train.head(10000), cv=3, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Final evaluation on the test set
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Additional metrics
    test_mae = mean_absolute_error(y_test, test_pred)
    test_evs = explained_variance_score(y_test, test_pred)
    
    print(f"Final test metrics:")
    print(f"  - MSE: {test_mse:.2f}")
    print(f"  - RMSE: {np.sqrt(test_mse):.2f}")
    print(f"  - MAE: {test_mae:.2f}")
    print(f"  - R²: {test_r2:.3f}")
    print(f"  - Explained Variance: {test_evs:.3f}")
    
    # Simplified plot creation to save memory
    plt.figure(figsize=(10, 6))
    # Convert to numpy array first to avoid any pandas indexing issues
    y_test_values = y_test.values
    plt.scatter(y_test_values[:1000], test_pred[:1000], alpha=0.5)  # Only plot a subset
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.title(f'Actual vs Predicted Profit for Region {region_id}')
    plt.savefig(f'region_{region_id}_predictions.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Try to get feature importances (simplified to avoid memory issues)
    try:
        feature_importances = model.named_steps['regressor'].feature_importances_
        # Simpler feature names solution
        feature_names = numerical_features + [f"cat_{i}" for i in range(len(feature_importances) - len(numerical_features))]
        feature_names = feature_names[:len(feature_importances)]
            
        # Create a DataFrame with feature names and importances
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        
        # Sort and plot top 10 features only
        top_features = feature_importance_df.sort_values('importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top 10 Feature Importances for Region {region_id}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'region_{region_id}_feature_importance.png', dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot feature importances. Error: {e}")
    
    # Clear memory
    gc.collect()
    
    return {
        'model': None,  # Don't store the model to save memory
        'test_pred': None,  # Don't store predictions to save memory
        'y_test': None,  # Don't store test data to save memory
        'metrics': {
            'test_mse': test_mse,
            'test_rmse': np.sqrt(test_mse),
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_evs': test_evs,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    }

def analyze_profitability_by_trip_length(df, region_id):
    """Analyze profitability by trip length for a specific region"""
    print(f"Analyzing profitability by trip length for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Group by trip length category and calculate mean profitability
    profit_by_length = region_df.groupby('trip_length_category')['total_profit'].agg(['mean', 'count']).reset_index()
    profit_by_length = profit_by_length.rename(columns={'mean': 'avg_profit', 'count': 'trip_count'})
    
    # Also calculate profit per mile
    profit_by_mile = region_df.groupby('trip_length_category').apply(
        lambda x: (x['total_profit'].sum() / x['trip_distance'].sum()) if x['trip_distance'].sum() > 0 else 0
    ).reset_index(name='profit_per_mile')
    
    # Merge the results
    profit_analysis = pd.merge(profit_by_length, profit_by_mile, on='trip_length_category')
    
    return profit_analysis

def analyze_seasonal_effects(df, region_id):
    """Analyze seasonal effects on profitability for a specific region"""
    print(f"Analyzing seasonal effects for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Group by season and trip length, then calculate mean profitability
    seasonal_profit = region_df.groupby(['season', 'trip_length_category'])['total_profit'].mean().reset_index()
    
    return seasonal_profit

def analyze_time_effects(df, region_id):
    """Analyze time of day effects on profitability for a specific region"""
    print(f"Analyzing time of day effects for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Create time of day categories
    bins = [0, 6, 12, 18, 24]
    labels = ['night', 'morning', 'afternoon', 'evening']
    region_df['time_of_day'] = pd.cut(region_df['hour'], bins=bins, labels=labels, include_lowest=True)
    
    # Group by time of day and trip length, then calculate mean profitability
    time_profit = region_df.groupby(['time_of_day', 'trip_length_category'])['total_profit'].mean().reset_index()
    
    return time_profit

def analyze_optimal_timing(df, region_id):
    """Analyze optimal timing for trips in a specific region, considering time of day, day of week, and season"""
    print(f"Analyzing optimal timing for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Time of day analysis
    time_profit = region_df.groupby(['hour', 'trip_length_category'])['total_profit'].mean().reset_index()
    
    # Day of week analysis  
    dow_profit = region_df.groupby(['day_of_week', 'trip_length_category'])['total_profit'].mean().reset_index()
    # Map numerical day of week to names for better readability
    dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    dow_profit['day_name'] = dow_profit['day_of_week'].map(dow_map)
    
    # Season analysis
    season_profit = region_df.groupby(['season', 'trip_length_category'])['total_profit'].mean().reset_index()
    
    # Rush hour vs non-rush hour
    rush_profit = region_df.groupby(['is_rush_hour', 'trip_length_category'])['total_profit'].mean().reset_index()
    rush_profit['rush_hour'] = rush_profit['is_rush_hour'].map({0: 'Non-Rush Hour', 1: 'Rush Hour'})
    
    # Find optimal time for each trip length category
    optimal_timing = []
    
    for trip_cat in region_df['trip_length_category'].unique():
        # Skip if the category doesn't exist
        if not any(time_profit['trip_length_category'] == trip_cat):
            continue
            
        # Best hour
        best_hour_data = time_profit[time_profit['trip_length_category'] == trip_cat]
        best_hour = best_hour_data.loc[best_hour_data['total_profit'].idxmax()]
        
        # Best day of week
        best_dow_data = dow_profit[dow_profit['trip_length_category'] == trip_cat]
        best_dow = best_dow_data.loc[best_dow_data['total_profit'].idxmax()]
        
        # Best season
        best_season_data = season_profit[season_profit['trip_length_category'] == trip_cat]
        best_season = best_season_data.loc[best_season_data['total_profit'].idxmax()]
        
        # Rush hour effect
        best_rush_data = rush_profit[rush_profit['trip_length_category'] == trip_cat]
        best_rush = best_rush_data.loc[best_rush_data['total_profit'].idxmax()]
        
        optimal_timing.append({
            'trip_length_category': trip_cat,
            'best_hour': int(best_hour['hour']),
            'hour_profit': float(best_hour['total_profit']),
            'best_day': best_dow['day_name'],
            'day_profit': float(best_dow['total_profit']),
            'best_season': best_season['season'],
            'season_profit': float(best_season['total_profit']),
            'rush_hour_better': best_rush['rush_hour'],
            'rush_profit': float(best_rush['total_profit'])
        })
    
    return pd.DataFrame(optimal_timing)

def plot_hourly_patterns(df, region_id):
    """Plot hourly profit patterns for different trip lengths in a region"""
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Hourly analysis
    hourly_profit = region_df.groupby(['hour', 'trip_length_category'])['total_profit'].mean().reset_index()
    
    # Pivot for plotting
    pivot_data = hourly_profit.pivot(index='hour', columns='trip_length_category', values='total_profit')
    
    # Plot
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=pivot_data)
    plt.title(f'Hourly Profit Patterns by Trip Length (Region {region_id})', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Average Profit ($)', fontsize=14)
    plt.xticks(range(0, 24, 2))  # Show even hours only for cleaner plot
    plt.grid(True, alpha=0.3)
    plt.legend(title='Trip Length')
    plt.tight_layout()
    plt.savefig(f'hourly_profit_region_{region_id}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return hourly_profit

def plot_weekly_patterns(df, region_id):
    """Plot day of week profit patterns for different trip lengths in a region"""
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Day of week analysis
    dow_profit = region_df.groupby(['day_of_week', 'trip_length_category'])['total_profit'].mean().reset_index()
    
    # Pivot for plotting
    pivot_data = dow_profit.pivot(index='day_of_week', columns='trip_length_category', values='total_profit')
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pivot_data)
    plt.title(f'Day of Week Profit Patterns by Trip Length (Region {region_id})', fontsize=16)
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=14)
    plt.ylabel('Average Profit ($)', fontsize=14)
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.grid(True, alpha=0.3)
    plt.legend(title='Trip Length')
    plt.tight_layout()
    plt.savefig(f'weekly_profit_region_{region_id}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return dow_profit

def plot_profitability_analysis(profit_analysis, region_id):
    """Plot the profitability analysis results"""
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot average profit
    sns.barplot(x='trip_length_category', y='avg_profit', data=profit_analysis, ax=ax1, palette='viridis')
    ax1.set_title(f'Average Profit by Trip Length (Region {region_id})', fontsize=16)
    ax1.set_xlabel('Trip Length Category', fontsize=14)
    ax1.set_ylabel('Average Profit ($)', fontsize=14)
    ax1.tick_params(labelsize=12)
    for i, row in enumerate(profit_analysis.itertuples()):
        ax1.text(i, row.avg_profit + 0.5, f'${row.avg_profit:.2f}\n(n={row.trip_count})', 
                 ha='center', va='bottom', fontsize=12)
    
    # Plot profit per mile
    sns.barplot(x='trip_length_category', y='profit_per_mile', data=profit_analysis, ax=ax2, palette='viridis')
    ax2.set_title(f'Profit per Mile by Trip Length (Region {region_id})', fontsize=16)
    ax2.set_xlabel('Trip Length Category', fontsize=14)
    ax2.set_ylabel('Profit per Mile ($)', fontsize=14)
    ax2.tick_params(labelsize=12)
    for i, row in enumerate(profit_analysis.itertuples()):
        ax2.text(i, row.profit_per_mile + 0.1, f'${row.profit_per_mile:.2f}', 
                 ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'profitability_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_effects(seasonal_profit, region_id):
    """Plot seasonal effects on profitability"""
    plt.figure(figsize=(14, 8))
    
    # Create a pivot table for the heatmap
    pivot_data = seasonal_profit.pivot(index='season', columns='trip_length_category', values='total_profit')
    
    # Plot the heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f'Seasonal Effects on Profit by Trip Length (Region {region_id})', fontsize=16)
    plt.xlabel('Trip Length Category', fontsize=14)
    plt.ylabel('Season', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'seasonal_effects_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_effects(time_profit, region_id):
    """Plot time of day effects on profitability"""
    plt.figure(figsize=(14, 8))
    
    # Create a pivot table for the heatmap
    pivot_data = time_profit.pivot(index='time_of_day', columns='trip_length_category', values='total_profit')
    
    # Plot the heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f'Time of Day Effects on Profit by Trip Length (Region {region_id})', fontsize=16)
    plt.xlabel('Trip Length Category', fontsize=14)
    plt.ylabel('Time of Day', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'time_effects_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_recommendation(profit_analysis, region_id, metrics):
    """Print trip length recommendation for a region based on different criteria"""
    # Most profitable trip length (by average profit)
    most_profitable = profit_analysis.loc[profit_analysis['avg_profit'].idxmax()]
    
    # Most profitable trip length (by profit per mile)
    most_efficient = profit_analysis.loc[profit_analysis['profit_per_mile'].idxmax()]
    
    # Most common trip length
    most_common = profit_analysis.loc[profit_analysis['trip_count'].idxmax()]
    
    print("\n" + "="*80)
    print(f"RECOMMENDATIONS FOR REGION {region_id}")
    print("="*80)
    
    print(f"Most profitable trip length: {most_profitable['trip_length_category']}")
    print(f"  - Average profit: ${most_profitable['avg_profit']:.2f}")
    print(f"  - Trip count: {most_profitable['trip_count']}")
    
    print(f"\nMost efficient trip length (profit per mile): {most_efficient['trip_length_category']}")
    print(f"  - Profit per mile: ${most_efficient['profit_per_mile']:.2f}")
    print(f"  - Average profit: ${most_efficient['avg_profit']:.2f}")
    print(f"  - Trip count: {most_efficient['trip_count']}")
    
    print(f"\nMost common trip length: {most_common['trip_length_category']}")
    print(f"  - Trip count: {most_common['trip_count']}")
    print(f"  - Average profit: ${most_common['avg_profit']:.2f}")
    
    print("\nModel performance metrics:")
    print(f"  - Test R²: {metrics['test_r2']:.3f}")
    
    # Handle different metrics formats from basic and optimized models
    if 'cv_r2_mean' in metrics:
        # Basic model format
        print(f"  - Cross-validation R² (mean ± std): {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    elif 'cv_r2' in metrics:
        # Optimized model format
        print(f"  - Cross-validation R²: {metrics['cv_r2']:.3f}")
    
    print(f"  - RMSE: {metrics['test_rmse']:.2f}")
    print(f"  - MAE: {metrics['test_mae']:.2f}")
    
    print("="*80)
    print(f"OVERALL RECOMMENDATION: Focus on {most_profitable['trip_length_category']} trips for maximum profit")
    print("="*80 + "\n")

def print_optimal_timing_recommendation(optimal_timing, region_id):
    """Print optimal timing recommendations for each trip length category"""
    print("\n" + "="*100)
    print(f"OPTIMAL TIMING RECOMMENDATIONS FOR REGION {region_id}")
    print("="*100)
    
    for _, row in optimal_timing.iterrows():
        print(f"\nTrip Length: {row['trip_length_category']}")
        
        # Format hour in 12-hour format with AM/PM
        hour_12 = row['best_hour'] % 12
        if hour_12 == 0:
            hour_12 = 12
        am_pm = 'AM' if row['best_hour'] < 12 else 'PM'
        
        print(f"  - Best Hour: {hour_12} {am_pm} (Hour {row['best_hour']}) → ${row['hour_profit']:.2f}")
        print(f"  - Best Day: {row['best_day']} → ${row['day_profit']:.2f}")
        print(f"  - Best Season: {row['best_season']} → ${row['season_profit']:.2f}")
        print(f"  - Rush Hour Effect: {row['rush_hour_better']} → ${row['rush_profit']:.2f}")
        
        # Create a simplified optimal time string
        optimal_time = f"{hour_12} {am_pm} on {row['best_day']}s during {row['best_season']}"
        if row['rush_hour_better'] == 'Rush Hour':
            optimal_time += " (during rush hour)"
        
        print(f"  → OPTIMAL TIMING: {optimal_time}")
    
    print("="*100)

def analyze_od_pairs(df, region_id, top_n=5):
    """Analyze the top origin-destination pairs for a specific region"""
    print(f"Analyzing top origin-destination pairs for region {region_id}...")
    
    # Filter data for the specified region as origin
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Get the top destination locations
    top_destinations = region_df['DOLocationID'].value_counts().nlargest(top_n)
    print(f"Top {top_n} destinations from region {region_id}: {top_destinations.index.tolist()}")
    
    # Analyze profitability for each top O-D pair
    od_results = []
    
    for dest_id in top_destinations.index:
        # Filter for this specific O-D pair
        od_df = region_df[region_df['DOLocationID'] == dest_id]
        
        # Calculate metrics
        avg_profit = od_df['total_profit'].mean()
        avg_distance = od_df['trip_distance'].mean()
        profit_per_mile = od_df['profit_per_mile'].mean()
        trip_count = len(od_df)
        
        # Add to results
        od_results.append({
            'origin': region_id,
            'destination': dest_id,
            'avg_profit': avg_profit,
            'avg_distance': avg_distance,
            'profit_per_mile': profit_per_mile,
            'trip_count': trip_count
        })
    
    od_df = pd.DataFrame(od_results)
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    sns.barplot(x='destination', y='avg_profit', data=od_df, palette='viridis')
    plt.title(f'Average Profit by Destination (Origin: Region {region_id})', fontsize=16)
    plt.xlabel('Destination Region ID', fontsize=14)
    plt.ylabel('Average Profit ($)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add annotations
    for i, row in enumerate(od_df.itertuples()):
        plt.text(i, row.avg_profit + 0.5, 
                f'${row.avg_profit:.2f}\n({row.avg_distance:.1f} mi)\nn={row.trip_count}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'od_profitability_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return od_df

def analyze_payment_types(df, region_id):
    """Analyze how payment types impact profitability in a specific region"""
    print(f"Analyzing payment type effects for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Create a mapping for payment types for better readability
    payment_map = {
        1: 'Credit Card',
        2: 'Cash',
        3: 'No Charge',
        4: 'Dispute',
        5: 'Unknown',
        6: 'Voided'
    }
    region_df['payment_type_desc'] = region_df['payment_type'].map(payment_map)
    
    # Group by payment type and trip length, calculate metrics
    payment_profit = region_df.groupby(['payment_type_desc', 'trip_length_category']).agg({
        'total_profit': 'mean',
        'tip_amount': 'mean',
        'fare_amount': 'mean',
        'profit_per_mile': 'mean',
        'PULocationID': 'count'  # count for number of trips
    }).reset_index()
    
    payment_profit = payment_profit.rename(columns={'PULocationID': 'trip_count'})
    
    # Filter out uncommon payment types with too few trips (threshold of 20)
    payment_profit = payment_profit[payment_profit['trip_count'] >= 20]
    
    # Create visualizations
    # Plot 1: Overall payment type distribution
    plt.figure(figsize=(12, 6))
    payment_counts = region_df['payment_type_desc'].value_counts()
    payment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                       colors=sns.color_palette('viridis', len(payment_counts)))
    plt.title(f'Payment Type Distribution (Region {region_id})', fontsize=16)
    plt.ylabel('')  # Hide the 'None' ylabel
    plt.tight_layout()
    plt.savefig(f'payment_distribution_region_{region_id}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Average profit by payment type and trip length
    plt.figure(figsize=(14, 8))
    
    # Create pivot and plot heatmap
    pivot_data = payment_profit.pivot(index='payment_type_desc', 
                                     columns='trip_length_category', 
                                     values='total_profit')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f'Average Profit by Payment Type and Trip Length (Region {region_id})', fontsize=16)
    plt.xlabel('Trip Length Category', fontsize=14)
    plt.ylabel('Payment Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'payment_profit_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Average tip by payment type and trip length (only if relevant)
    plt.figure(figsize=(14, 8))
    
    # Create pivot and plot heatmap
    pivot_data = payment_profit.pivot(index='payment_type_desc', 
                                     columns='trip_length_category', 
                                     values='tip_amount')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f'Average Tip by Payment Type and Trip Length (Region {region_id})', fontsize=16)
    plt.xlabel('Trip Length Category', fontsize=14)
    plt.ylabel('Payment Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'payment_tip_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return payment_profit

def build_optimized_model(df, region_id, max_samples=50000):
    """Build and optimize a regression model for a specific region using GridSearchCV"""
    print(f"Building and optimizing regression model for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # If we have too many samples, take a random subset to avoid memory issues
    if len(region_df) > max_samples:
        print(f"Sampling {max_samples} records from {len(region_df)} for region {region_id}")
        region_df = region_df.sample(n=max_samples, random_state=42)
    
    # Features to use in the model
    numerical_features = [
        'trip_distance', 'passenger_count', 'trip_distance_squared',
        'hour', 'day', 'day_of_week', 'is_weekend', 'is_rush_hour'
    ]
    categorical_features = ['month', 'year', 'payment_type', 'season']
    
    # Define the model pipeline with imputation steps for missing values
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        # Remove polynomial features for initial optimization to reduce complexity
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Create a model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    # Prepare features and target
    X = region_df[numerical_features + categorical_features]
    y = region_df['total_profit']
    
    # Drop rows with NaN in target variable
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # For optimization use a smaller subset
    if len(X) > 10000:
        print(f"Using 10,000 samples for hyperparameter optimization")
        # Create a common index for sampling to maintain alignment
        indices = np.random.RandomState(42).choice(len(X), size=10000, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Define parameter grid
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 4],
        'regressor__subsample': [0.8, 1.0],
        'regressor__min_samples_split': [2, 5]
    }
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='r2',
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the grid search
    print("Starting grid search for hyperparameter optimization...")
    grid_search.fit(X_sample, y_sample)
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Split the full dataset for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the best model on the full training data
    print("Fitting best model on full training data...")
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_evs = explained_variance_score(y_test, test_pred)
    
    print(f"Final test metrics with optimized model:")
    print(f"  - MSE: {test_mse:.2f}")
    print(f"  - RMSE: {np.sqrt(test_mse):.2f}")
    print(f"  - MAE: {test_mae:.2f}")
    print(f"  - R²: {test_r2:.3f}")
    print(f"  - Explained Variance: {test_evs:.3f}")
    
    # Save the model
    model_filename = f'models/region_{region_id}_model.joblib'
    # Create the directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Create feature importance plot
    try:
        feature_importances = best_model.named_steps['regressor'].feature_importances_
        
        # Get feature names from preprocessor
        preprocessor = best_model.named_steps['preprocessor']
        feature_names = []
        
        for name, _, cols in preprocessor.transformers_:
            if hasattr(preprocessor.named_transformers_[name], 'get_feature_names_out'):
                feature_names.extend(preprocessor.named_transformers_[name].get_feature_names_out(cols))
            else:
                feature_names.extend([f"{name}_{col}" for col in cols])
        
        # Use simplified feature names if extraction fails
        if len(feature_names) != len(feature_importances):
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
        
        # Create a DataFrame with feature names and importances
        feature_importance_df = pd.DataFrame({
            'feature': feature_names[:len(feature_importances)],
            'importance': feature_importances
        })
        
        # Sort and plot top 15 features only
        top_features = feature_importance_df.sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top 15 Feature Importances (Optimized Model, Region {region_id})', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'optimized_feature_importance_region_{region_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot feature importances. Error: {e}")
    
    # Clear memory
    gc.collect()
    
    return {
        'model': best_model,  # Return the model now
        'best_params': grid_search.best_params_,
        'metrics': {
            'test_mse': test_mse,
            'test_rmse': np.sqrt(test_mse),
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_evs': test_evs,
            'cv_r2': grid_search.best_score_
        }
    }

def perform_clustering_analysis(df, region_id, n_clusters=4):
    """Perform clustering analysis to identify distinct trip patterns in a region"""
    print(f"Performing clustering analysis for region {region_id}...")
    
    # Filter data for the specified region
    region_df = df[df['PULocationID'] == region_id].copy()
    
    # Limit sample size for clustering to avoid memory issues
    if len(region_df) > 20000:
        region_df = region_df.sample(n=20000, random_state=42)
    
    # Select relevant features for clustering
    clustering_features = [
        'trip_distance', 'fare_amount', 'tip_amount', 'total_profit',
        'profit_per_mile', 'hour', 'day_of_week', 'is_weekend',
        'is_rush_hour', 'passenger_count'
    ]
    
    # Drop rows with missing values
    cluster_df = region_df[clustering_features].dropna()
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df)
    
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the dataframe
    cluster_df['cluster'] = cluster_labels
    
    # Use PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create a dataframe with principal components and cluster labels
    pca_df = pd.DataFrame(
        data=principal_components, 
        columns=['PC1', 'PC2']
    )
    pca_df['cluster'] = cluster_labels
    
    # Add back some key metrics for interpretation
    for feature in ['trip_distance', 'total_profit', 'profit_per_mile', 'hour', 'day_of_week']:
        pca_df[feature] = cluster_df[feature].values
    
    # Visualize the clusters
    plt.figure(figsize=(12, 10))
    
    # Plot PCA scatter with clusters
    plt.subplot(2, 1, 1)
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=50, alpha=0.7)
    plt.title(f'Trip Clusters for Region {region_id} (PCA Visualization)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cluster', title_fontsize=12)
    
    # Calculate cluster characteristics
    cluster_summary = cluster_df.groupby('cluster').agg({
        'trip_distance': 'mean',
        'fare_amount': 'mean',
        'tip_amount': 'mean',
        'total_profit': 'mean',
        'profit_per_mile': 'mean',
        'hour': 'mean',
        'day_of_week': 'mean',
        'is_weekend': 'mean',
        'is_rush_hour': 'mean',
        'passenger_count': 'mean'
    }).reset_index()
    
    # Add a count column
    cluster_counts = cluster_df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    cluster_summary = pd.merge(cluster_summary, cluster_counts, on='cluster')
    
    # Add percentage column
    cluster_summary['percentage'] = cluster_summary['count'] / cluster_summary['count'].sum() * 100
    
    # Normalize for radar chart
    radar_features = ['trip_distance', 'total_profit', 'profit_per_mile', 'hour', 'is_weekend']
    radar_df = cluster_summary[radar_features + ['cluster']].copy()
    
    for feature in radar_features:
        max_val = radar_df[feature].max()
        min_val = radar_df[feature].min()
        if max_val > min_val:
            radar_df[feature] = (radar_df[feature] - min_val) / (max_val - min_val)
    
    # Plot radar chart for each cluster
    plt.subplot(2, 1, 2)
    
    # Set data
    categories = radar_features
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    ax = plt.subplot(2, 1, 2, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each cluster
    for i in range(n_clusters):
        values = radar_df[radar_df['cluster'] == i].iloc[0, :-1].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Cluster Characteristics Comparison (Region {region_id})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'cluster_analysis_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a profitability vs distance scatter plot by cluster
    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        x='trip_distance', 
        y='total_profit',
        hue='cluster',
        size='profit_per_mile',  
        sizes=(20, 200),
        alpha=0.7,
        data=cluster_df
    )
    plt.title(f'Trip Profitability vs Distance by Cluster (Region {region_id})', fontsize=16)
    plt.xlabel('Trip Distance (miles)', fontsize=14)
    plt.ylabel('Total Profit ($)', fontsize=14)
    plt.legend(title='Cluster', title_fontsize=12)
    plt.tight_layout()
    plt.savefig(f'cluster_profitability_region_{region_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table
    print(f"\nCLUSTER SUMMARY FOR REGION {region_id}:")
    print(cluster_summary[['cluster', 'count', 'percentage', 'trip_distance', 'total_profit', 
                          'profit_per_mile', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].to_string(index=False))
    
    # Define cluster types based on characteristics
    cluster_types = []
    
    for _, cluster in cluster_summary.iterrows():
        # Check if it's a short trip
        is_short = cluster['trip_distance'] < cluster_summary['trip_distance'].median()
        
        # Check if it's profitable
        is_profitable = cluster['profit_per_mile'] > cluster_summary['profit_per_mile'].median()
        
        # Check if it's a weekend trip
        is_weekend = cluster['is_weekend'] > 0.5
        
        # Check if it's a rush hour trip
        is_rush = cluster['is_rush_hour'] > 0.5
        
        # Define type based on these characteristics
        cluster_type = ""
        if is_short and is_profitable:
            cluster_type = "Short, high-efficiency trips"
        elif is_short and not is_profitable:
            cluster_type = "Short, low-efficiency trips"
        elif not is_short and is_profitable:
            cluster_type = "Long, high-efficiency trips"
        else:
            cluster_type = "Long, low-efficiency trips"
            
        # Add timing context
        if is_weekend:
            cluster_type += " (weekend)"
        elif is_rush:
            cluster_type += " (rush hour)"
        else:
            cluster_type += " (regular hours)"
            
        cluster_types.append({
            'cluster': cluster['cluster'],
            'type': cluster_type,
            'trip_distance': cluster['trip_distance'],
            'profit_per_mile': cluster['profit_per_mile'],
            'percentage': cluster['percentage']
        })
    
    cluster_types_df = pd.DataFrame(cluster_types)
    print("\nCLUSTER INTERPRETATIONS:")
    print(cluster_types_df.to_string(index=False))
    
    return {
        'cluster_summary': cluster_summary,
        'cluster_types': cluster_types_df
    }

def main():
    """Main function to run the analysis"""
    # Load data with sampling to reduce memory usage
    sample_size = 1000000  # Use only 1M records
    df = load_data("tripdata_combined.parquet", sample_size=sample_size)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Clear memory after preprocessing
    df = None
    gc.collect()
    
    # Get top regions
    top_regions = analyze_top_regions(df_processed, top_n=3)  # Analyze only top 3 regions
    
    # Save top regions for app
    os.makedirs('app_data', exist_ok=True)
    joblib.dump(top_regions, 'app_data/top_regions.joblib')
    
    # Analyze each top region
    results = {}
    
    for region_id in top_regions:
        print(f"\nAnalyzing region {region_id}...")
        
        # Build optimized regression model
        print(f"Building optimized model for region {region_id}")
        model_results = build_optimized_model(df_processed, region_id, max_samples=50000)
        
        # Analyze profitability by trip length
        profit_analysis = analyze_profitability_by_trip_length(df_processed, region_id)
        
        # Analyze seasonal effects
        seasonal_profit = analyze_seasonal_effects(df_processed, region_id)
        
        # Analyze time effects
        time_profit = analyze_time_effects(df_processed, region_id)
        
        # Analyze optimal timing patterns
        optimal_timing = analyze_optimal_timing(df_processed, region_id)
        
        # Plot hourly and weekly patterns
        hourly_data = plot_hourly_patterns(df_processed, region_id)
        weekly_data = plot_weekly_patterns(df_processed, region_id)
        
        # Analyze origin-destination pairs
        od_analysis = analyze_od_pairs(df_processed, region_id)
        
        # Analyze payment types impact
        payment_analysis = analyze_payment_types(df_processed, region_id)
        
        # Perform clustering analysis
        cluster_analysis = perform_clustering_analysis(df_processed, region_id)
        
        # Plot results
        plot_profitability_analysis(profit_analysis, region_id)
        plot_seasonal_effects(seasonal_profit, region_id)
        plot_time_effects(time_profit, region_id)
        
        # Print recommendations
        print_recommendation(profit_analysis, region_id, model_results['metrics'])
        
        # Print optimal timing recommendations
        print_optimal_timing_recommendation(optimal_timing, region_id)
        
        # Store results (minimal data)
        results[region_id] = {
            'profit_analysis': profit_analysis,
            'seasonal_profit': seasonal_profit,
            'time_profit': time_profit,
            'optimal_timing': optimal_timing,
            'od_analysis': od_analysis,
            'payment_analysis': payment_analysis,
            'cluster_analysis': cluster_analysis,
            'metrics': model_results['metrics']
        }
        
        # Save individual region data for app
        region_data_path = f'app_data/region_{region_id}_data.joblib'
        joblib.dump({
            'profit_analysis': profit_analysis,
            'optimal_timing': optimal_timing,
            'hourly_data': hourly_data,
            'weekly_data': weekly_data,
            'od_analysis': od_analysis
        }, region_data_path)
        print(f"Saved region {region_id} data to {region_data_path}")
        
        # Clear memory after each region
        gc.collect()
    
    # Save all results
    joblib.dump(results, 'app_data/all_results.joblib')
    print("Saved all analysis results to app_data/all_results.joblib")
    
    # Generate reports
    generate_summary_report(results, top_regions)
    generate_timing_summary_report(results, top_regions)
    generate_od_summary_report(results, top_regions)
    generate_payment_summary_report(results, top_regions)
    generate_cluster_summary_report(results, top_regions)

def generate_summary_report(results, top_regions):
    """Generate a summary report of the analysis results"""
    print("\n" + "="*100)
    print("SUMMARY REPORT: OPTIMAL TRIP LENGTHS BY REGION")
    print("="*100)
    
    summary_data = []
    
    for region_id in top_regions:
        profit_analysis = results[region_id]['profit_analysis']
        metrics = results[region_id]['metrics']
        
        # Most profitable trip length
        most_profitable = profit_analysis.loc[profit_analysis['avg_profit'].idxmax()]
        
        # Most efficient trip length
        most_efficient = profit_analysis.loc[profit_analysis['profit_per_mile'].idxmax()]
        
        # Create a summary entry with metrics compatible with both optimized and basic models
        summary_entry = {
            'region_id': region_id,
            'most_profitable_length': most_profitable['trip_length_category'],
            'avg_profit': most_profitable['avg_profit'],
            'most_efficient_length': most_efficient['trip_length_category'],
            'profit_per_mile': most_efficient['profit_per_mile'],
            'test_r2': metrics['test_r2']
        }
        
        # Add cross-validation metrics in the appropriate format
        if 'cv_r2_mean' in metrics:
            summary_entry['cv_r2_mean'] = metrics['cv_r2_mean']
            summary_entry['cv_r2_std'] = metrics['cv_r2_std']
        elif 'cv_r2' in metrics:
            summary_entry['cv_r2_mean'] = metrics['cv_r2']
            summary_entry['cv_r2_std'] = 0.0  # No std dev available for optimized model
        
        summary_data.append(summary_entry)
    
    # Create a summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Print the summary table
    print(summary_df.to_string(index=False))
    print("="*100)
    
    # Print detailed metrics table
    print("\nDETAILED MODEL PERFORMANCE METRICS BY REGION")
    print("="*100)
    
    metrics_data = []
    for region_id in top_regions:
        metrics = results[region_id]['metrics']
        
        # Create a metrics entry compatible with both optimized and basic models
        metrics_entry = {
            'region_id': region_id,
            'MSE': f"{metrics['test_mse']:.2f}",
            'RMSE': f"{metrics['test_rmse']:.2f}",
            'MAE': f"{metrics['test_mae']:.2f}",
            'R²': f"{metrics['test_r2']:.3f}",
            'Expl. Var.': f"{metrics['test_evs']:.3f}"
        }
        
        # Add cross-validation metrics in the appropriate format
        if 'cv_r2_mean' in metrics:
            metrics_entry['CV R² (mean)'] = f"{metrics['cv_r2_mean']:.3f}"
            metrics_entry['CV R² (std)'] = f"{metrics['cv_r2_std']:.3f}"
        elif 'cv_r2' in metrics:
            metrics_entry['CV R² (mean)'] = f"{metrics['cv_r2']:.3f}"
            metrics_entry['CV R² (std)'] = "N/A"
            
        metrics_data.append(metrics_entry)
    
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))
    print("="*100)
    
    # Plot a comparison of average profit across regions
    plt.figure(figsize=(12, 8))
    sns.barplot(x='region_id', y='avg_profit', data=summary_df, palette='viridis')
    plt.title('Average Profit by Region (Most Profitable Trip Length)', fontsize=16)
    plt.xlabel('Region ID', fontsize=14)
    plt.ylabel('Average Profit ($)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add annotations
    for i, row in enumerate(summary_df.itertuples()):
        plt.text(i, row.avg_profit + 0.5, 
                f'${row.avg_profit:.2f}\n({row.most_profitable_length})\nR²: {row.test_r2:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('region_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot model performance comparison
    plt.figure(figsize=(12, 8))
    x = range(len(top_regions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot test and CV R² values
    test_r2_vals = [results[region_id]['metrics']['test_r2'] for region_id in top_regions]
    cv_r2_means = []
    cv_r2_stds = []
    
    for region_id in top_regions:
        metrics = results[region_id]['metrics']
        if 'cv_r2_mean' in metrics:
            cv_r2_means.append(metrics['cv_r2_mean'])
            cv_r2_stds.append(metrics['cv_r2_std'])
        elif 'cv_r2' in metrics:
            cv_r2_means.append(metrics['cv_r2'])
            cv_r2_stds.append(0.0)  # No std dev for optimized model
    
    ax.bar([i - width/2 for i in x], test_r2_vals, width, label='Test R²', alpha=0.7)
    ax.bar([i + width/2 for i in x], cv_r2_means, width, label='Cross-Validation R² (mean)', alpha=0.7,
            yerr=cv_r2_stds, capsize=5)
    
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_xlabel('Region ID', fontsize=14)
    ax.set_title('Model Performance Comparison Across Regions', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(region_id) for region_id in top_regions])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Summary report generated and saved!")

def generate_timing_summary_report(results, top_regions):
    """Generate a summary report of the optimal timing results"""
    print("\n" + "="*100)
    print("SUMMARY REPORT: OPTIMAL TIMING BY REGION AND TRIP LENGTH")
    print("="*100)
    
    # Combine timing data from all regions
    all_timing_data = []
    
    for region_id in top_regions:
        optimal_timing = results[region_id]['optimal_timing']
        
        # Add region_id to each row
        optimal_timing_with_region = optimal_timing.copy()
        optimal_timing_with_region['region_id'] = region_id
        
        all_timing_data.append(optimal_timing_with_region)
    
    # Combine all data
    if all_timing_data:
        combined_timing = pd.concat(all_timing_data)
        
        # Create a summary table
        summary_table = combined_timing.pivot(index='trip_length_category', columns='region_id', values=['best_hour', 'best_day', 'best_season'])
        
        print("\nOPTIMAL HOUR BY TRIP LENGTH AND REGION:")
        print(summary_table['best_hour'].to_string())
        
        print("\nOPTIMAL DAY BY TRIP LENGTH AND REGION:")
        print(summary_table['best_day'].to_string())
        
        print("\nOPTIMAL SEASON BY TRIP LENGTH AND REGION:")
        print(summary_table['best_season'].to_string())
        
        # Create and save a visualization of best hours
        plt.figure(figsize=(12, 8))
        
        # Hour heatmap
        hour_pivot = combined_timing.pivot(index='trip_length_category', columns='region_id', values='best_hour')
        sns.heatmap(hour_pivot, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
        
        plt.title('Optimal Hour by Trip Length and Region', fontsize=16)
        plt.tight_layout()
        plt.savefig('optimal_hour_summary.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print("\nSummary visualizations saved to disk.")
    else:
        print("No timing data available for summary.")
    
    print("="*100)

def generate_od_summary_report(results, top_regions):
    """Generate a summary report of the origin-destination analysis results"""
    print("\n" + "="*100)
    print("SUMMARY REPORT: MOST PROFITABLE ORIGIN-DESTINATION PAIRS")
    print("="*100)
    
    # Combine OD data from all regions
    all_od_data = []
    
    for region_id in top_regions:
        od_analysis = results[region_id]['od_analysis']
        all_od_data.append(od_analysis)
    
    # Combine all data
    if all_od_data:
        combined_od = pd.concat(all_od_data)
        
        # Sort by profitability
        top_routes = combined_od.sort_values('avg_profit', ascending=False).head(10)
        
        print("TOP 10 MOST PROFITABLE ROUTES:")
        print(top_routes[['origin', 'destination', 'avg_profit', 'avg_distance', 'profit_per_mile', 'trip_count']].to_string(index=False))
        
        # Create visualization - simplified to avoid colorbar issues
        plt.figure(figsize=(14, 8))
        
        # Use a simpler approach to create the scatter plot
        scatter = plt.scatter(
            combined_od['avg_distance'],
            combined_od['avg_profit'],
            s=combined_od['trip_count']/10,  # Size based on trip count
            c=combined_od['profit_per_mile'],  # Color based on profit per mile
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add labels for top 5 routes
        for idx, row in top_routes.head(5).iterrows():
            plt.annotate(f"{row['origin']} → {row['destination']}", 
                        xy=(row['avg_distance'], row['avg_profit']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.title('Trip Profitability by Distance', fontsize=16)
        plt.xlabel('Average Trip Distance (miles)', fontsize=14)
        plt.ylabel('Average Profit ($)', fontsize=14)
        
        # Add colorbar with direct reference to the scatter plot
        cbar = plt.colorbar(scatter)
        cbar.set_label('Profit per Mile ($)')
        
        # Add legend for bubble size
        plt.text(
            0.95, 0.05, 
            "Bubble size = Number of trips", 
            transform=plt.gca().transAxes,
            ha='right', 
            fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig('od_profitability_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nOrigin-destination summary visualization saved to disk.")
    else:
        print("No origin-destination data available for summary.")
    
    print("="*100)

def generate_payment_summary_report(results, top_regions):
    """Generate a summary report of payment type analysis results"""
    print("\n" + "="*100)
    print("SUMMARY REPORT: PAYMENT TYPE IMPACT ON PROFITABILITY")
    print("="*100)
    
    # Summarize payment type findings
    for region_id in top_regions:
        payment_analysis = results[region_id]['payment_analysis']
        
        # Find most profitable payment type
        by_payment = payment_analysis.groupby('payment_type_desc')['total_profit'].mean().reset_index()
        most_profitable_payment = by_payment.loc[by_payment['total_profit'].idxmax()]
        
        # Find payment type with highest tips
        by_payment_tip = payment_analysis.groupby('payment_type_desc')['tip_amount'].mean().reset_index()
        highest_tip_payment = by_payment_tip.loc[by_payment_tip['tip_amount'].idxmax()]
        
        print(f"\nREGION {region_id} PAYMENT INSIGHTS:")
        print(f"  - Most profitable payment type: {most_profitable_payment['payment_type_desc']} (${most_profitable_payment['total_profit']:.2f} avg profit)")
        print(f"  - Highest tip payment type: {highest_tip_payment['payment_type_desc']} (${highest_tip_payment['tip_amount']:.2f} avg tip)")
    
    # Create a combined visualization comparing payment types across regions
    payment_summary = []
    
    for region_id in top_regions:
        payment_analysis = results[region_id]['payment_analysis']
        
        # Group by payment type
        by_payment = payment_analysis.groupby('payment_type_desc').agg({
            'total_profit': 'mean',
            'tip_amount': 'mean'
        }).reset_index()
        
        # Add region id
        by_payment['region_id'] = region_id
        
        # Add to summary data
        payment_summary.append(by_payment)
    
    # Combine data
    if payment_summary:
        combined_payment = pd.concat(payment_summary)
        
        # Plot comparison
        plt.figure(figsize=(14, 8))
        
        # Create a grouped bar chart
        sns.catplot(
            data=combined_payment, kind="bar",
            x="payment_type_desc", y="total_profit", hue="region_id",
            palette="viridis", alpha=.8, height=6, aspect=2
        )
        
        plt.title('Average Profit by Payment Type Across Regions', fontsize=16)
        plt.xlabel('Payment Type', fontsize=14)
        plt.ylabel('Average Profit ($)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('payment_profit_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot tip comparison
        plt.figure(figsize=(14, 8))
        
        # Create a grouped bar chart for tips
        sns.catplot(
            data=combined_payment, kind="bar",
            x="payment_type_desc", y="tip_amount", hue="region_id",
            palette="viridis", alpha=.8, height=6, aspect=2
        )
        
        plt.title('Average Tip by Payment Type Across Regions', fontsize=16)
        plt.xlabel('Payment Type', fontsize=14)
        plt.ylabel('Average Tip ($)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('payment_tip_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nPayment type comparison visualizations saved to disk.")
    else:
        print("No payment type data available for summary.")
    
    print("="*100)

def generate_cluster_summary_report(results, top_regions):
    """Generate a summary report of the clustering analysis results"""
    print("\n" + "="*100)
    print("SUMMARY REPORT: TRIP PATTERN CLUSTERS")
    print("="*100)
    
    all_cluster_types = []
    
    for region_id in top_regions:
        if 'cluster_analysis' in results[region_id]:
            cluster_types = results[region_id]['cluster_analysis']['cluster_types']
            
            # Add region ID
            cluster_types_with_region = cluster_types.copy()
            cluster_types_with_region['region_id'] = region_id
            
            all_cluster_types.append(cluster_types_with_region)
    
    if all_cluster_types:
        # Combine all cluster types
        combined_clusters = pd.concat(all_cluster_types)
        
        # Print summary by region
        for region_id in top_regions:
            region_clusters = combined_clusters[combined_clusters['region_id'] == region_id]
            
            print(f"\nREGION {region_id} CLUSTER INSIGHTS:")
            for _, cluster in region_clusters.iterrows():
                print(f"  - Cluster {int(cluster['cluster'])}: {cluster['type']} "
                      f"({cluster['percentage']:.1f}% of trips, "
                      f"${cluster['profit_per_mile']:.2f}/mile)")
        
        # Create a simpler visualization comparing cluster distributions
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        plot_data = []
        for region_id in top_regions:
            if 'cluster_analysis' in results[region_id]:
                summary = results[region_id]['cluster_analysis']['cluster_summary']
                for _, row in summary.iterrows():
                    plot_data.append({
                        'region_id': region_id,
                        'cluster': f"Cluster {int(row['cluster'])}",
                        'percentage': row['percentage'],
                        'profit_per_mile': row['profit_per_mile']
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create a simple grouped bar chart
        regions = plot_df['region_id'].unique()
        clusters = plot_df['cluster'].unique()
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.8 / len(clusters)
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        
        # Plot each cluster as a group of bars
        for i, cluster in enumerate(clusters):
            cluster_data = plot_df[plot_df['cluster'] == cluster]
            positions = np.arange(len(regions))
            heights = []
            
            # Get heights for each region
            for region in regions:
                region_data = cluster_data[cluster_data['region_id'] == region]
                if len(region_data) > 0:
                    heights.append(region_data['percentage'].values[0])
                else:
                    heights.append(0)
            
            # Plot the bars
            offset = (i - len(clusters)/2 + 0.5) * bar_width
            bars = ax.bar(positions + offset, heights, bar_width, label=cluster, color=colors[i], alpha=0.8)
            
            # Add profit per mile annotations to bars
            for j, region in enumerate(regions):
                region_data = cluster_data[cluster_data['region_id'] == region]
                if len(region_data) > 0 and heights[j] > 5:  # Only annotate if bar is large enough
                    profit = region_data['profit_per_mile'].values[0]
                    ax.text(
                        j + offset, 
                        heights[j]/2, 
                        f"${profit:.2f}/mi", 
                        ha='center', 
                        va='center', 
                        fontsize=9, 
                        color='white', 
                        fontweight='bold'
                    )
        
        # Set labels and title
        ax.set_ylabel('Percentage of Trips (%)', fontsize=14)
        ax.set_xlabel('Region ID', fontsize=14)
        ax.set_title('Cluster Distribution by Region', fontsize=16)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels(regions)
        ax.legend(title='Cluster')
        
        plt.tight_layout()
        plt.savefig('cluster_distribution_by_region.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nCluster distribution visualization saved to disk.")
    else:
        print("No clustering data available for summary.")
    
    print("="*100)

if __name__ == "__main__":
    main() 