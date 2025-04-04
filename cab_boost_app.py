import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz

# Set page configuration
st.set_page_config(
    page_title="CabBoost - Driver Recommendation System",
    page_icon="🚕",
    layout="wide",
)

# Function to check if required data and models exist
def check_data_availability():
    required_paths = [
        'app_data/top_regions.joblib',
        'app_data/all_results.joblib'
    ]
    
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    
    if missing_paths:
        st.error(f"Missing required data files: {', '.join(missing_paths)}")
        st.info("Please run trip_profitability_analysis.py first to generate the necessary data files.")
        return False
        
    # Check if at least one model exists
    models_exist = False
    if os.path.exists('app_data/top_regions.joblib'):
        top_regions = joblib.load('app_data/top_regions.joblib')
        for region_id in top_regions:
            if os.path.exists(f'models/region_{region_id}_model.joblib'):
                models_exist = True
                break
    
    if not models_exist:
        st.error("No models found. Please run trip_profitability_analysis.py first to generate models.")
        return False
        
    return True

# Function to load data and models
@st.cache_resource
def load_data_and_models():
    top_regions = joblib.load('app_data/top_regions.joblib')
    all_results = joblib.load('app_data/all_results.joblib')
    
    models = {}
    region_data = {}
    
    for region_id in top_regions:
        # Load model if exists
        model_path = f'models/region_{region_id}_model.joblib'
        if os.path.exists(model_path):
            models[region_id] = joblib.load(model_path)
        else:
            st.warning(f"Model for region {region_id} not found. Some predictions may not be available.")
            
        # Load region data
        data_path = f'app_data/region_{region_id}_data.joblib'
        if os.path.exists(data_path):
            region_data[region_id] = joblib.load(data_path)
    
    return top_regions, all_results, models, region_data

# Function to get current time features
def get_current_time_features():
    # Get current time in NY timezone (where taxi data is from)
    ny_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_tz)
    
    hour = current_time.hour
    day = current_time.day
    month = current_time.month
    year = current_time.year
    day_of_week = current_time.weekday()  # 0 = Monday, 6 = Sunday
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (hour >= 7 and hour <= 10) or (hour >= 16 and hour <= 19) else 0
    
    # Map month to season
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    season = season_map[month]
    
    return {
        'hour': hour,
        'day': day,
        'month': month,
        'year': year,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'season': season
    }

# Function to predict profitability
def predict_profitability(region_id, model, features):
    if model is None:
        return None
    
    # Create feature dataframe
    df = pd.DataFrame([features])
    
    # Predict
    try:
        prediction = model.predict(df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Function to recommend optimal trip length category based on time
def recommend_optimal_trip_length(region_data, time_features):
    optimal_timing = region_data['optimal_timing']
    
    # Find trips that match current time features
    matches = []
    
    for _, trip in optimal_timing.iterrows():
        score = 0
        
        # Check hour match (with some flexibility)
        if abs(trip['best_hour'] - time_features['hour']) <= 2:
            score += 2
        elif abs(trip['best_hour'] - time_features['hour']) <= 4:
            score += 1
            
        # Check day of week match
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                  'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        if day_map.get(trip['best_day'], -1) == time_features['day_of_week']:
            score += 3
            
        # Check season match
        if trip['best_season'] == time_features['season']:
            score += 2
            
        # Check rush hour match
        rush_hour_value = 'Rush Hour' if time_features['is_rush_hour'] == 1 else 'Non-Rush Hour'
        if trip['rush_hour_better'] == rush_hour_value:
            score += 2
            
        matches.append({
            'trip_length_category': trip['trip_length_category'],
            'score': score,
            'hour_profit': trip['hour_profit'],
            'day_profit': trip['day_profit'],
            'season_profit': trip['season_profit'],
            'rush_profit': trip['rush_profit']
        })
    
    # Sort by score and then by profit
    sorted_matches = sorted(matches, key=lambda x: (x['score'], x['hour_profit']), reverse=True)
    
    return sorted_matches

def draw_profit_by_hour_chart(region_data, region_id):
    hourly_data = region_data['hourly_data']
    
    # Pivot the data
    pivot_data = hourly_data.pivot(index='hour', columns='trip_length_category', values='total_profit')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    sns.lineplot(data=pivot_data, ax=ax)
    
    # Set labels and title
    ax.set_title(f'Hourly Profit by Trip Length (Region {region_id})', fontsize=16)
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Average Profit ($)', fontsize=14)
    
    # Set ticks
    ax.set_xticks(range(0, 24, 2))
    
    # Add a vertical line for current hour
    current_hour = datetime.now().hour
    ax.axvline(x=current_hour, color='r', linestyle='--', alpha=0.7, label=f'Current hour ({current_hour})')
    
    # Add legend
    ax.legend(title='Trip Length')
    
    # Show the plot
    st.pyplot(fig)

def draw_profit_by_day_chart(region_data, region_id):
    weekly_data = region_data['weekly_data']
    
    # Pivot the data
    pivot_data = weekly_data.pivot(index='day_of_week', columns='trip_length_category', values='total_profit')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    sns.lineplot(data=pivot_data, ax=ax)
    
    # Set labels and title
    ax.set_title(f'Day of Week Profit by Trip Length (Region {region_id})', fontsize=16)
    ax.set_xlabel('Day of Week', fontsize=14)
    ax.set_ylabel('Average Profit ($)', fontsize=14)
    
    # Set ticks
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # Add a vertical line for current day
    current_day = datetime.now().weekday()
    ax.axvline(x=current_day, color='r', linestyle='--', alpha=0.7, label=f'Current day')
    
    # Add legend
    ax.legend(title='Trip Length')
    
    # Show the plot
    st.pyplot(fig)

def draw_od_chart(region_data, region_id):
    od_analysis = region_data['od_analysis']
    
    # Sort by average profit
    top_destinations = od_analysis.sort_values('avg_profit', ascending=False).head(5)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    bars = sns.barplot(x='destination', y='avg_profit', data=top_destinations, ax=ax)
    
    # Add text annotations
    for i, row in enumerate(top_destinations.itertuples()):
        ax.text(i, row.avg_profit/2, f"${row.avg_profit:.2f}\n{row.avg_distance:.1f} mi", 
               ha='center', color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_title(f'Top 5 Destinations from Region {region_id}', fontsize=16)
    ax.set_xlabel('Destination Region ID', fontsize=14)
    ax.set_ylabel('Average Profit ($)', fontsize=14)
    
    # Show the plot
    st.pyplot(fig)

def main():
    # Check if data and models are available
    if not check_data_availability():
        return
    
    # Load data and models
    top_regions, all_results, models, region_data = load_data_and_models()
    
    # App title and description
    st.title("🚕 CabBoost - Driver Recommendation System")
    st.markdown("""
    This app provides real-time recommendations for taxi drivers based on profitability analysis.
    Get insights on where to go, what types of trips to accept, and when to drive for maximum profit.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Driver Settings")
    
    # Region selection
    selected_region = st.sidebar.selectbox(
        "Select your current region",
        options=top_regions,
        format_func=lambda x: f"Region {x}"
    )
    
    # Get current time features
    time_features = get_current_time_features()
    
    # Display current conditions
    st.sidebar.subheader("Current Conditions")
    st.sidebar.write(f"Time: {time_features['hour']}:00")
    st.sidebar.write(f"Day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][time_features['day_of_week']]}")
    st.sidebar.write(f"Season: {time_features['season']}")
    st.sidebar.write(f"Rush Hour: {'Yes' if time_features['is_rush_hour'] == 1 else 'No'}")
    
    # Optional: Allow user to override time for simulation
    if st.sidebar.checkbox("Override current time/day", False):
        time_features['hour'] = st.sidebar.slider("Hour of day", 0, 23, time_features['hour'])
        time_features['day_of_week'] = st.sidebar.selectbox(
            "Day of week",
            options=range(7),
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
            index=time_features['day_of_week']
        )
        time_features['is_weekend'] = 1 if time_features['day_of_week'] >= 5 else 0
        time_features['is_rush_hour'] = 1 if (time_features['hour'] >= 7 and time_features['hour'] <= 10) or (time_features['hour'] >= 16 and time_features['hour'] <= 19) else 0
    
    # Get region data
    region_result = region_data.get(selected_region)
    
    if region_result is None:
        st.error(f"No data available for region {selected_region}")
        return
    
    # Get recommendations
    recommended_trips = recommend_optimal_trip_length(region_result, time_features)
    
    # Main content area - use tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Profit Analysis", "Destinations", "Scenario Comparison"])
    
    # Tab 1: Recommendations
    with tab1:
        st.header(f"Optimal Trip Recommendations for Region {selected_region}")
        
        # Add real-time prediction using the model
        if selected_region in models and models[selected_region] is not None:
            model = models[selected_region]
            
            # Create input features for prediction
            prediction_features = {
                'trip_distance': 0,  # We'll vary this
                'passenger_count': 1,
                'trip_distance_squared': 0,  # Will be calculated
                'hour': time_features['hour'],
                'day': time_features['day'],
                'day_of_week': time_features['day_of_week'],
                'is_weekend': time_features['is_weekend'],
                'is_rush_hour': time_features['is_rush_hour'],
                'month': time_features['month'],
                'year': time_features['year'],
                'payment_type': 1,  # Default to credit card
                'season': time_features['season']
            }
            
            # Create predictions for different trip distances
            st.subheader("Real-time Profit Predictions")
            
            # Let user customize prediction parameters
            with st.expander("Customize Prediction Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    payment_type = st.selectbox(
                        "Payment Method", 
                        options=[1, 2],
                        format_func=lambda x: "Credit Card" if x == 1 else "Cash",
                        index=0
                    )
                    passengers = st.slider("Number of Passengers", 1, 6, 1)
                with col2:
                    use_custom_destination = st.checkbox("Custom Destination", False)
                    if use_custom_destination:
                        destination = st.selectbox(
                            "Destination Region", 
                            options=list(set(region_result['od_analysis']['destination'].tolist())),
                            format_func=lambda x: f"Region {x}"
                        )
            
            # Update features with user selections
            prediction_features['payment_type'] = payment_type
            prediction_features['passenger_count'] = passengers
            
            # Predict for different trip distances
            distances = [1, 3, 5, 10, 15]  # miles
            predictions = []
            
            for dist in distances:
                prediction_features['trip_distance'] = dist
                prediction_features['trip_distance_squared'] = dist ** 2
                
                # Make prediction
                predicted_profit = predict_profitability(selected_region, model, prediction_features)
                
                if predicted_profit is not None:
                    predictions.append({
                        'distance': dist,
                        'profit': predicted_profit,
                        'profit_per_mile': predicted_profit / dist
                    })
            
            if predictions:
                # Display prediction results
                prediction_df = pd.DataFrame(predictions)
                
                # Create columns for the metrics
                pred_cols = st.columns(len(predictions))
                
                for i, pred in enumerate(predictions):
                    trip_category = ""
                    if pred['distance'] <= 2:
                        trip_category = "Very Short"
                    elif pred['distance'] <= 5:
                        trip_category = "Short"
                    elif pred['distance'] <= 10:
                        trip_category = "Medium"
                    else:
                        trip_category = "Long"
                        
                    with pred_cols[i]:
                        st.metric(
                            label=f"{trip_category} ({pred['distance']} mi)",
                            value=f"${pred['profit']:.2f}",
                            delta=f"${pred['profit_per_mile']:.2f}/mile",
                            delta_color="normal"
                        )
                
                # Plot the predictions
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(prediction_df['distance'], prediction_df['profit'], marker='o', linewidth=2)
                
                # Add data points annotations
                for i, row in prediction_df.iterrows():
                    ax.annotate(f"${row['profit']:.2f}", 
                               (row['distance'], row['profit']),
                               textcoords="offset points",
                               xytext=(0,10),
                               ha='center')
                
                # Set labels and title
                ax.set_title(f'Predicted Profit by Trip Distance (Region {selected_region})', fontsize=16)
                ax.set_xlabel('Distance (miles)', fontsize=14)
                ax.set_ylabel('Predicted Profit ($)', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Add current conditions as text
                conditions_text = (
                    f"Time: {time_features['hour']}:00, "
                    f"Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][time_features['day_of_week']]}, "
                    f"Payment: {'Credit Card' if payment_type == 1 else 'Cash'}, "
                    f"Passengers: {passengers}"
                )
                ax.text(0.5, 0.02, conditions_text, transform=ax.transAxes, ha='center', fontsize=12, 
                      bbox=dict(facecolor='white', alpha=0.7))
                
                st.pyplot(fig)
            else:
                st.warning("Could not generate predictions. Try different parameters.")
        else:
            st.warning("Model not available for this region. Only using historical analysis.")
        
        st.markdown("---")
        
        # Display top 3 recommendations
        st.subheader("Recommended Trip Types")
        for i, trip in enumerate(recommended_trips[:3]):
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.subheader(f"#{i+1}: {trip['trip_length_category'].replace('_', ' ').title()} Trips")
                
                with col2:
                    # Explanation of why this is recommended
                    st.markdown("**Why this is recommended now:**")
                    reasons = []
                    if time_features['is_rush_hour'] == 1 and trip['trip_length_category'] in ['short', 'very_short']:
                        reasons.append("✅ Rush hour favors shorter trips due to traffic")
                    if time_features['is_weekend'] == 1 and trip['trip_length_category'] in ['medium', 'long']:
                        reasons.append("✅ Weekends favor longer trips due to less traffic")
                    if trip['score'] >= 5:
                        reasons.append(f"✅ Current time ({time_features['hour']}:00) is optimal for this trip type")
                    if trip['trip_length_category'] in ['medium', 'short']:
                        reasons.append("✅ Good balance of trip volume and profit")
                    
                    if not reasons:
                        reasons.append("✅ Based on historical profitability patterns")
                    
                    for reason in reasons:
                        st.write(reason)
                
                with col3:
                    st.metric(
                        label="Expected Profit", 
                        value=f"${trip['hour_profit']:.2f}"
                    )
                
                st.divider()
        
        # Show profitability chart by trip length
        st.subheader("Profitability Analysis by Trip Length")
        
        profit_analysis = region_result['profit_analysis']
        
        # Create columns for the metrics
        cols = st.columns(len(profit_analysis))
        
        for i, (_, row) in enumerate(profit_analysis.iterrows()):
            with cols[i]:
                st.metric(
                    label=row['trip_length_category'].replace('_', ' ').title(),
                    value=f"${row['avg_profit']:.2f}",
                    delta=f"${row['profit_per_mile']:.2f}/mile",
                    delta_color="normal"
                )
    
    # Tab 2: Profit Analysis
    with tab2:
        st.header(f"Profit Analysis for Region {selected_region}")
        
        # Show hourly profit patterns
        st.subheader("Hourly Profit Patterns")
        draw_profit_by_hour_chart(region_result, selected_region)
        
        # Show weekly profit patterns
        st.subheader("Day of Week Profit Patterns")
        draw_profit_by_day_chart(region_result, selected_region)
    
    # Tab 3: Destinations
    with tab3:
        st.header(f"Best Destinations from Region {selected_region}")
        
        # Show top destinations
        draw_od_chart(region_result, selected_region)
        
        # List top destinations with details
        od_analysis = region_result['od_analysis']
        top_destinations = od_analysis.sort_values('avg_profit', ascending=False).head(5)
        
        for i, row in enumerate(top_destinations.itertuples()):
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.subheader(f"Destination {row.destination}")
                
                with col2:
                    st.write(f"Average Distance: {row.avg_distance:.1f} miles")
                    st.write(f"Profit per Mile: ${row.profit_per_mile:.2f}")
                    st.write(f"Number of Trips: {row.trip_count}")
                
                with col3:
                    st.metric(
                        label="Avg Profit",
                        value=f"${row.avg_profit:.2f}"
                    )
                
                st.divider()
    
    # Tab 4: Scenario Comparison
    with tab4:
        st.header("Compare Different Scenarios")
        st.markdown("""
        Use this tool to compare expected profit across different times, days, and other factors.
        The model will predict profitability based on the parameters you select.
        """)
        
        if selected_region in models and models[selected_region] is not None:
            model = models[selected_region]
            
            # Create two columns for different scenarios
            col1, col2 = st.columns(2)
            
            scenarios = []
            
            # Scenario 1
            with col1:
                st.subheader("Scenario 1")
                
                hour1 = st.slider("Hour of day", 0, 23, time_features['hour'], key="hour1")
                day1 = st.selectbox(
                    "Day of week",
                    options=range(7),
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                    index=time_features['day_of_week'],
                    key="day1"
                )
                is_weekend1 = 1 if day1 >= 5 else 0
                is_rush_hour1 = 1 if (hour1 >= 7 and hour1 <= 10) or (hour1 >= 16 and hour1 <= 19) else 0
                
                payment1 = st.selectbox(
                    "Payment Method", 
                    options=[1, 2],
                    format_func=lambda x: "Credit Card" if x == 1 else "Cash",
                    index=0,
                    key="payment1"
                )
                
                distance1 = st.slider("Trip Distance (miles)", 1, 20, 5, key="distance1")
                
                # Create scenario 1 features
                scenario1 = {
                    'trip_distance': distance1,
                    'passenger_count': 1,
                    'trip_distance_squared': distance1 ** 2,
                    'hour': hour1,
                    'day': time_features['day'],
                    'day_of_week': day1,
                    'is_weekend': is_weekend1,
                    'is_rush_hour': is_rush_hour1,
                    'month': time_features['month'],
                    'year': time_features['year'],
                    'payment_type': payment1,
                    'season': time_features['season']
                }
                
                # Make prediction for scenario 1
                predicted_profit1 = predict_profitability(selected_region, model, scenario1)
                
                if predicted_profit1 is not None:
                    st.metric(
                        label="Predicted Profit",
                        value=f"${predicted_profit1:.2f}",
                        delta=f"${predicted_profit1/distance1:.2f}/mile"
                    )
                    
                    scenarios.append({
                        'name': "Scenario 1",
                        'hour': hour1,
                        'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day1],
                        'payment': "Credit Card" if payment1 == 1 else "Cash",
                        'distance': distance1,
                        'profit': predicted_profit1,
                        'profit_per_mile': predicted_profit1/distance1
                    })
                else:
                    st.error("Could not generate prediction for Scenario 1")
            
            # Scenario 2
            with col2:
                st.subheader("Scenario 2")
                
                hour2 = st.slider("Hour of day", 0, 23, (time_features['hour'] + 12) % 24, key="hour2")
                day2 = st.selectbox(
                    "Day of week",
                    options=range(7),
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                    index=(time_features['day_of_week'] + 2) % 7,
                    key="day2"
                )
                is_weekend2 = 1 if day2 >= 5 else 0
                is_rush_hour2 = 1 if (hour2 >= 7 and hour2 <= 10) or (hour2 >= 16 and hour2 <= 19) else 0
                
                payment2 = st.selectbox(
                    "Payment Method", 
                    options=[1, 2],
                    format_func=lambda x: "Credit Card" if x == 1 else "Cash",
                    index=0,
                    key="payment2"
                )
                
                distance2 = st.slider("Trip Distance (miles)", 1, 20, 10, key="distance2")
                
                # Create scenario 2 features
                scenario2 = {
                    'trip_distance': distance2,
                    'passenger_count': 1,
                    'trip_distance_squared': distance2 ** 2,
                    'hour': hour2,
                    'day': time_features['day'],
                    'day_of_week': day2,
                    'is_weekend': is_weekend2,
                    'is_rush_hour': is_rush_hour2,
                    'month': time_features['month'],
                    'year': time_features['year'],
                    'payment_type': payment2,
                    'season': time_features['season']
                }
                
                # Make prediction for scenario 2
                predicted_profit2 = predict_profitability(selected_region, model, scenario2)
                
                if predicted_profit2 is not None:
                    st.metric(
                        label="Predicted Profit",
                        value=f"${predicted_profit2:.2f}",
                        delta=f"${predicted_profit2/distance2:.2f}/mile"
                    )
                    
                    scenarios.append({
                        'name': "Scenario 2",
                        'hour': hour2,
                        'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day2],
                        'payment': "Credit Card" if payment2 == 1 else "Cash",
                        'distance': distance2,
                        'profit': predicted_profit2,
                        'profit_per_mile': predicted_profit2/distance2
                    })
                else:
                    st.error("Could not generate prediction for Scenario 2")
            
            # Compare scenarios
            if len(scenarios) > 1:
                st.subheader("Scenario Comparison")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(scenarios)
                
                # Plot comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Total profit comparison
                colors = ['#ff9999', '#66b3ff']
                ax1.bar(comparison_df['name'], comparison_df['profit'], color=colors)
                ax1.set_title('Total Profit Comparison', fontsize=14)
                ax1.set_ylabel('Predicted Profit ($)', fontsize=12)
                
                # Add values on top of bars
                for i, v in enumerate(comparison_df['profit']):
                    ax1.text(i, v + 0.5, f"${v:.2f}", ha='center', fontsize=12, fontweight='bold')
                
                # Profit per mile comparison
                ax2.bar(comparison_df['name'], comparison_df['profit_per_mile'], color=colors)
                ax2.set_title('Profit per Mile Comparison', fontsize=14)
                ax2.set_ylabel('Profit per Mile ($)', fontsize=12)
                
                # Add values on top of bars
                for i, v in enumerate(comparison_df['profit_per_mile']):
                    ax2.text(i, v + 0.1, f"${v:.2f}", ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display details table
                st.subheader("Detailed Comparison")
                
                # Format the dataframe for display
                display_df = comparison_df[['name', 'day', 'hour', 'payment', 'distance', 'profit', 'profit_per_mile']].copy()
                display_df['hour'] = display_df['hour'].apply(lambda x: f"{x}:00")
                display_df['profit'] = display_df['profit'].apply(lambda x: f"${x:.2f}")
                display_df['profit_per_mile'] = display_df['profit_per_mile'].apply(lambda x: f"${x:.2f}")
                display_df.columns = ['Scenario', 'Day', 'Hour', 'Payment', 'Distance (mi)', 'Profit', 'Profit/Mile']
                
                st.table(display_df)
                
                # Recommendation based on comparison
                if predicted_profit1 > predicted_profit2:
                    st.success(f"✅ **Scenario 1** is more profitable by ${predicted_profit1 - predicted_profit2:.2f}")
                elif predicted_profit2 > predicted_profit1:
                    st.success(f"✅ **Scenario 2** is more profitable by ${predicted_profit2 - predicted_profit1:.2f}")
                else:
                    st.info("Both scenarios predict equal profitability")
                    
                # Insights
                st.subheader("Key Insights")
                insights = []
                
                # Time of day
                if abs(hour1 - hour2) > 2:
                    if predicted_profit1 > predicted_profit2:
                        insights.append(f"• Driving at {hour1}:00 is more profitable than {hour2}:00 for the selected parameters")
                    else:
                        insights.append(f"• Driving at {hour2}:00 is more profitable than {hour1}:00 for the selected parameters")
                
                # Day of week
                if day1 != day2:
                    day1_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day1]
                    day2_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day2]
                    if predicted_profit1/distance1 > predicted_profit2/distance2:
                        insights.append(f"• {day1_name} shows higher per-mile profit than {day2_name}")
                    else:
                        insights.append(f"• {day2_name} shows higher per-mile profit than {day1_name}")
                
                # Payment type
                if payment1 != payment2:
                    payment1_name = "Credit card" if payment1 == 1 else "Cash"
                    payment2_name = "Credit card" if payment2 == 1 else "Cash"
                    if predicted_profit1 > predicted_profit2 and distance1 == distance2:
                        insights.append(f"• {payment1_name} payments result in higher profits than {payment2_name} for similar trips")
                    elif predicted_profit2 > predicted_profit1 and distance1 == distance2:
                        insights.append(f"• {payment2_name} payments result in higher profits than {payment1_name} for similar trips")
                
                # Display insights
                if insights:
                    for insight in insights:
                        st.markdown(insight)
                else:
                    st.write("No significant differences between scenarios to highlight")
        else:
            st.warning("Model not available for this region. Cannot generate predictions.")

    # Add info about the app
    st.sidebar.markdown("---")
    st.sidebar.subheader("About CabBoost")
    st.sidebar.info(
        """
        CabBoost analyzes taxi trip data to provide data-driven recommendations
        to maximize driver profitability based on location, time, and trip type.
        
        This app is powered by machine learning models trained on NY taxi trip data.
        """
    )

if __name__ == "__main__":
    main() 