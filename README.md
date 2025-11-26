# Enhancing Localized Weather Forecasting in Iloilo Using Long-Short Term Memory (LSTM)

## Abstract
This thesis presents the development of a localized weather forecasting system for Iloilo, Philippines, utilizing Long Short-Term Memory (LSTM) neural networks. The system provides accurate short-term weather predictions by analyzing historical weather patterns and atmospheric conditions specific to the Western Visayas region. The web-based application offers user-friendly access to both historical weather data and future forecasts, demonstrating the practical application of deep learning in meteorological science.

## 1 Introduction
Weather forecasting plays a crucial role in various sectors including agriculture, transportation, disaster management, and daily planning. Traditional weather prediction methods often struggle with localized accuracy, particularly in regions with complex microclimates like the Philippines. This research addresses this challenge by implementing an LSTM-based forecasting system specifically tailored for Iloilo's unique weather patterns.

## 2 Methodology

### 2.1 Data Collection
- **Source**: Open-Meteo Historical Weather API
- **Parameters**: Temperature (min/max), Rainfall, Wind speed, Relative humidity, Dew point, Sunshine duration
- **Time Range**: 2014-2024 historical data
- **Geographic Scope**: Western Visayas region, Philippines

### 2.2 Model Architecture
- **Algorithm**: Long Short-Term Memory (LSTM) Neural Network
- **Input Features**: 7 weather parameters across 7-day sequences
- **Output**: Multi-day weather forecasts (1-7 days)
- **Training**: Historical weather data with time-series analysis

### 2.3 Technical Implementation
- **Backend**: Python Flask web framework
- **Frontend**: HTML5, CSS3, JavaScript with Chart.js
- **Machine Learning**: TensorFlow/Keras with LSTM implementation
- **Data Processing**: Pandas, NumPy for time-series manipulation
- **API Integration**: Open-Meteo for historical and real-time data

## 3 System Features

### 3.1 Core Functionality
- **Location Selection**: Interactive search for Western Visayas locations
- **Historical Data**: Last 7 days weather data display
- **Weather Forecasting**: 1-7 day predictions using LSTM model
- **Trend Analysis**: Interactive charts for weather parameter trends
- **Comparative Analysis**: Historical weather patterns for specific dates

### 3.2 User Interface
- Responsive web design compatible with desktop and mobile devices
- Interactive charts with parameter selection
- Real-time data visualization
- Location-based customization
- Historical trend comparisons

## 4 Technical Specifications

### 4.1 System Requirements
- Python 3.8+
- Flask web framework
- TensorFlow/Keras
- Pandas, NumPy
- Open-Meteo API access
- Modern web browser with JavaScript support

### 4.2 Model Specifications
- **Input Sequence**: 7 days × 7 parameters
- **LSTM Layers**: Multiple layers with dropout regularization
- **Output**: 7 weather parameters for each forecast day
- **Training**: Backpropagation through time with Adam optimizer

## 5 Implementation Details

### 5.1 Data Preprocessing
- Time-series normalization and scaling
- Missing data interpolation
- Feature engineering for seasonal patterns
- Data validation and cleaning

### 5.2 Model Training
- Cross-validation techniques
- Hyperparameter optimization
- Overfitting prevention with early stopping
- Performance metrics: MSE, MAE

### 5.3 Web Application
- RESTful API design
- Real-time data fetching
- Interactive visualization
- Error handling and user feedback

## 6 Results and Discussion

### 6.1 Model Performance
- Accurate short-term weather predictions
- Effective capture of seasonal patterns
- Reliable performance across different weather parameters
- Improved localization compared to regional forecasts

### 6.2 User Experience
- Intuitive interface for non-technical users
- Fast response times for forecasts
- Comprehensive historical context
- Mobile-responsive design

## 7 Conclusion
The developed LSTM-based weather forecasting system demonstrates the effectiveness of deep learning approaches for localized weather prediction in Iloilo. The system provides accurate short-term forecasts while offering valuable historical context through an accessible web interface. This research contributes to the field of applied meteorology and showcases the potential of machine learning in addressing region-specific environmental challenges.

## 8 Future Work
- Integration of additional weather parameters
- Expansion to other regions in the Philippines
- Real-time model retraining capabilities
- Mobile application development
- Ensemble methods with multiple forecasting models
- Seasonal and long-range forecasting capabilities

---

**Thesis Developed By:**
- Jael, Nethan Quinn
- Maxino, Loi Marie
- Oliverio, Aaron Hans
- Patriarca, Raymart John

**Adviser:** Lovidrick Barrios | Maureen Nettie Linan

**Institution:** Iloilo Science and Technology University - Main Campus

**Date:** December 2025

---

*This thesis fulfills the requirements for the degree of Bachelor of Science in Computer Science at Iloilo Science and Technology University - Main Campus.*
