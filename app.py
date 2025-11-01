from flask import Flask, render_template, request, jsonify
import datetime as dt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np
import keras
from keras.models import load_model
from tensorflow.keras import losses, metrics
import joblib
from datetime import datetime, timedelta
import requests
import os

app = Flask(__name__)

# Default location coordinates (Iloilo City)
DEFAULT_LOCATION = {
    'name': 'Iloilo City',
    'lat': 10.6969,
    'lon': 122.5644,
    'elevation': 8.0,
    'admin1': 'Western Visayas',
    'admin2': 'Iloilo',
    'admin3': 'Iloilo City'
}

# Load the pre-trained model and scaler
try:
    model = load_model('LSTM_Weather_Forcast_Model_new2.h5', 
                      custom_objects={'mse': losses.MeanSquaredError(),
                                     'mae': metrics.MeanAbsoluteError()})
    print("Model loaded successfully with custom objects.")
except Exception as e:
    print(f"Failed to load model with custom objects: {e}")
    try:
        model = load_model('LSTM_Weather_Forcast_Model_new2.h5', compile=False)
        model.compile(optimizer='adam', loss='mse')
        print("Model loaded successfully (fallback, compiled manually).")
    except Exception as e:
        print(f"Failed to load model completely: {e}")
        model = None

try:
    scaler = joblib.load('iloilo_weather_scaler.pkl')
except:
    scaler = None

# Helper function to convert numpy types to native Python types
def convert_to_native_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

# Function to search for locations
def search_locations(query):
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": query,
        "count": 10,
        "language": "en",
        "countryCode": "PH"
    }
    
    try:
        response = requests.get(geo_url, params=geo_params)
        data = response.json()
        
        if "results" in data:
            # Filter for locations in Western Visayas and format the results
            locations = []
            for place in data["results"]:
                if place.get("admin1", "") == "Western Visayas":
                    location_data = {
                        'name': place.get("name", ""),
                        'lat': float(place.get("latitude", 0)),
                        'lon': float(place.get("longitude", 0)),
                        'elevation': float(place.get("elevation", 0)) if place.get("elevation") else "N/A",
                        'admin1': place.get("admin1", ""),
                        'admin2': place.get("admin2", ""),
                        'admin3': place.get("admin3", "")
                    }
                    locations.append(location_data)
            return locations
        else:
            return []
    except Exception as e:
        print(f"Error searching locations: {e}")
        return []

# Function to fetch weather data for specific date range and location
def fetch_weather_data(start_date, end_date, location_data=DEFAULT_LOCATION):
    lat = location_data['lat']
    lon = location_data['lon']
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_min", 
            "temperature_2m_max", 
            "rain_sum", 
            "wind_speed_10m_mean",
            "relative_humidity_2m_mean", 
            "dew_point_2m_mean", 
            "sunshine_duration"
        ],
        "timezone": "Asia/Singapore",
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process daily data
        daily = response.Daily()
        daily_temperature_2m_min = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
        daily_wind_speed_10m_mean = daily.Variables(3).ValuesAsNumpy()
        daily_relative_humidity_2m_mean = daily.Variables(4).ValuesAsNumpy()
        daily_dew_point_2m_mean = daily.Variables(5).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(6).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}

        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["rain_sum"] = daily_rain_sum
        daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
        daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
        daily_data["dew_point_2m_mean"] = daily_dew_point_2m_mean
        daily_data["sunshine_duration"] = daily_sunshine_duration

        daily_weather_dataframe = pd.DataFrame(data=daily_data)
        daily_weather_dataframe['date'] = pd.to_datetime(daily_weather_dataframe['date'])
        daily_weather_dataframe['sunshine_duration'] = daily_weather_dataframe['sunshine_duration'] / 3600
        daily_weather_dataframe = daily_weather_dataframe.set_index('date')
        
        return daily_weather_dataframe
    
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return get_sample_data(start_date, end_date)

def get_sample_data(start_date, end_date):
    # Generate sample data for the requested date range
    date_range = pd.date_range(start=start_date, end=end_date)
    sample_data = {
        "date": date_range,
        "temperature_2m_min": np.random.uniform(23, 25, len(date_range)),
        "temperature_2m_max": np.random.uniform(28, 32, len(date_range)),
        "rain_sum": np.random.uniform(0, 10, len(date_range)),
        "wind_speed_10m_mean": np.random.uniform(5, 15, len(date_range)),
        "relative_humidity_2m_mean": np.random.uniform(80, 95, len(date_range)),
        "dew_point_2m_mean": np.random.uniform(23, 25, len(date_range)),
        "sunshine_duration": np.random.uniform(6, 12, len(date_range))
    }
    
    df = pd.DataFrame(sample_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

# Function to prepare data for prediction
def prepare_prediction_data(data):
    # Interpolate missing data
    data['sunshine_duration'] = data['sunshine_duration'].interpolate(method='time').bfill().ffill()
    
    # Normalize data
    if scaler is not None:
        scaled_data = scaler.transform(data)
    else:
        # Fallback normalization if scaler not available
        try:
            from sklearn.preprocessing import MinMaxScaler
            fallback_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = fallback_scaler.fit_transform(data)
        except ImportError:
            # If sklearn is not available, use simple manual normalization
            scaled_data = data.copy()
            for col in data.columns:
                if data[col].max() - data[col].min() > 0:
                    scaled_data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                else:
                    scaled_data[col] = 0.5
            scaled_data = scaled_data.values
    
    # Use the last 7 days for prediction
    sequence = scaled_data[-7:].reshape(1, 7, 7)  # 7 days, 7 features
    
    return sequence

# Function to generate predictions
def generate_predictions(sequence, days_to_predict=2):
    predictions = []
    
    if model is None:
        # Return sample predictions if model not available
        return get_sample_predictions(days_to_predict)
    
    # Predict specified number of days
    for _ in range(days_to_predict):
        pred = model.predict(sequence, verbose=0)
        predictions.append(pred[0])
        
        # Update sequence for next prediction
        sequence = np.append(sequence[:, 1:, :], pred.reshape(1, 1, 7), axis=1)
    
    # Inverse transform predictions
    predictions = np.array(predictions)
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    
    return predictions

def get_sample_predictions(days_to_predict):
    predictions = []
    for i in range(days_to_predict):
        predictions.append([
            24.0 + i * 0.2,  # min_temp
            29.5 + i * 0.5,  # max_temp
            max(0, 5.0 - i * 2.0),  # rain_sum
            8.0 + i * 1.0,  # wind_speed
            max(0, min(100, 85.0 - i * 2.0)),  # humidity
            23.5 + i * 0.1,  # dew_point
            max(0, 8.0 + i * 0.5)  # sunshine_duration (hours)
        ])
    return np.array(predictions)

# Historical weather function
def get_historical_weather(day, month, years_range=range(2014, 2025), location_data=DEFAULT_LOCATION):
    historical_data = []
    
    lat = location_data['lat']
    lon = location_data['lon']
    
    for year in years_range:
        try:
            cache_session = requests_cache.CachedSession('.cache', expire_after=86400)
            retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)

            target_date = dt.date(year, month, day)
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "daily": [
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "rain_sum",
                    "wind_speed_10m_mean",
                    "relative_humidity_2m_mean",
                    "dew_point_2m_mean",
                    "sunshine_duration"
                ],
                "timezone": "Asia/Singapore",
            }
            
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()

            # Convert numpy values to native Python types
            historical_data.append({
                'year': int(year),
                'temperature_2m_min': float(daily.Variables(0).ValuesAsNumpy()[0]),
                'temperature_2m_max': float(daily.Variables(1).ValuesAsNumpy()[0]),
                'rain_sum': float(daily.Variables(2).ValuesAsNumpy()[0]),
                'wind_speed_10m_mean': float(daily.Variables(3).ValuesAsNumpy()[0]),
                'relative_humidity_2m_mean': float(daily.Variables(4).ValuesAsNumpy()[0]),
                'dew_point_2m_mean': float(daily.Variables(5).ValuesAsNumpy()[0]),
                'sunshine_duration': float(daily.Variables(6).ValuesAsNumpy()[0])
            })
        except Exception as e:
            print(f"Error fetching historical data for {year}: {e}")
            continue
    return pd.DataFrame(historical_data)

@app.route('/search_locations')
def search_locations_route():
    query = request.args.get('query', '')
    if query:
        locations = search_locations(query)
        # Convert all numpy types to native Python types
        locations = convert_to_native_types(locations)
        return jsonify(locations)
    return jsonify([])

@app.route('/')
def index():
    # Default date range (past 7 days)
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=7)
    
    return render_template('index.html', 
                         default_start_date=start_date.strftime('%Y-%m-%d'),
                         default_end_date=end_date.strftime('%Y-%m-%d'),
                         selected_location=DEFAULT_LOCATION)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    try:
        if request.method == 'POST':
            # Get form data
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            days_to_predict = int(request.form.get('days_to_predict', 2))
            
            # Get location data from form
            location_name = request.form.get('location_name', 'Iloilo City')
            location_lat = float(request.form.get('location_lat', DEFAULT_LOCATION['lat']))
            location_lon = float(request.form.get('location_lon', DEFAULT_LOCATION['lon']))
            location_elevation = request.form.get('location_elevation', DEFAULT_LOCATION['elevation'])
            
            location_data = {
                'name': location_name,
                'lat': location_lat,
                'lon': location_lon,
                'elevation': location_elevation,
                'admin1': 'Western Visayas'
            }
        else:
            # Get data from URL parameters
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            days_to_predict = int(request.args.get('days_to_predict', 2))
            
            # Default to past 7 days if no dates provided
            if not start_date or not end_date:
                end_date_default = dt.date.today()
                start_date_default = end_date_default - dt.timedelta(days=7)
                start_date = start_date_default.strftime('%Y-%m-%d')
                end_date = end_date_default.strftime('%Y-%m-%d')
            
            location_data = DEFAULT_LOCATION
        
        # Fetch weather data for the specified date range and location
        weather_data = fetch_weather_data(start_date, end_date, location_data)
        
        # Check if we have enough data for prediction (at least 7 days)
        if len(weather_data) < 7:
            return f"Error: Need at least 7 days of data for prediction. Only got {len(weather_data)} days."
        
        # Prepare data for prediction
        sequence = prepare_prediction_data(weather_data)
        
        # Generate predictions
        predictions = generate_predictions(sequence, days_to_predict)
        
        # Format past weather data for display (convert to native types)
        past_days = []
        for i, (date, row) in enumerate(weather_data.iterrows()):
            past_days.append({
                'date': date.date(),
                'temperature_2m_min': float(round(row['temperature_2m_min'], 3)),
                'temperature_2m_max': float(round(row['temperature_2m_max'], 3)),
                'rain_sum': float(round(row['rain_sum'], 3)),
                'wind_speed_10m_mean': float(round(row['wind_speed_10m_mean'], 3)),
                'relative_humidity_2m_mean': float(round(row['relative_humidity_2m_mean'], 3)),
                'dew_point_2m_mean': float(round(row['dew_point_2m_mean'], 3)),
                'sunshine_duration': float(round(row['sunshine_duration'] * 3600, 3))
            })
        
        # Format predictions for display (convert to native types)
        last_date = weather_data.index[-1]
        future_dates = [last_date + dt.timedelta(days=i+1) for i in range(days_to_predict)]
        forecast_data = []
        for i, pred in enumerate(predictions):
            forecast_data.append({
                'date': future_dates[i].date(),
                'temperature_2m_min': float(round(pred[0], 3)),
                'temperature_2m_max': float(round(pred[1], 3)),
                'rain_sum': float(round(pred[2], 3)),
                'wind_speed_10m_mean': float(round(pred[3], 3)),
                'relative_humidity_2m_mean': float(round(pred[4], 3)),
                'dew_point_2m_mean': float(round(pred[5], 3)),
                'sunshine_duration': float(round(pred[6] * 3600, 3))
            })
        
        # Get historical context for predicted days
        historical_data_list = []
        for forecast_day in forecast_data:
            d, m = forecast_day['date'].day, forecast_day['date'].month
            historical_df = get_historical_weather(d, m, location_data=location_data)
            if not historical_df.empty:
                # Convert DataFrame to list of dictionaries with native types
                historical_records = []
                for _, row in historical_df.iterrows():
                    historical_records.append({
                        'year': int(row['year']),
                        'temperature_2m_min': float(row['temperature_2m_min']),
                        'temperature_2m_max': float(row['temperature_2m_max']),
                        'rain_sum': float(row['rain_sum']),
                        'wind_speed_10m_mean': float(row['wind_speed_10m_mean']),
                        'relative_humidity_2m_mean': float(row['relative_humidity_2m_mean']),
                        'dew_point_2m_mean': float(row['dew_point_2m_mean']),
                        'sunshine_duration': float(row['sunshine_duration'])
                    })
                historical_data_list.append(historical_records)
            else:
                historical_data_list.append([])

        historical_forecast_1 = historical_data_list[0] if len(historical_data_list) > 0 else []
        historical_forecast_2 = historical_data_list[1] if len(historical_data_list) > 1 else []

        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']

        return render_template('index.html', 
                            historical_data=past_days, 
                            forecast_data=forecast_data,
                            historical_forecast_1=historical_forecast_1,
                            historical_forecast_2=historical_forecast_2,
                            month_names=month_names,
                            start_date=start_date,
                            end_date=end_date,
                            days_to_predict=days_to_predict,
                            selected_location=location_data)
                            
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
