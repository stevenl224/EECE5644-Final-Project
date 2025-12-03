# Flight Performance Prediction Using Weather Data at Boston Logan Airport

Predicting flight delays, cancellations, air-time, and taxi times at Boston Logan International Airport (BOS) using machine learning models trained on historical flight data (2019-2023) and weather observations.

## Project Overview

This project investigates how weather conditions impact various aspects of flight performance at Boston's busiest airport. We apply multiple machine learning approaches—from interpretable linear models to ensemble methods—to understand the complex relationship between meteorological factors and operational outcomes.

### Key Findings

- **Departure Delay Prediction**: Weather alone explains only ~7% of delay variance (R² = 0.069), indicating operational factors dominate
- **Air-Time Prediction**: Strong linear relationship (R² = 0.98) between flight distance, east-west wind component, and flight duration
- **Cancellation Prediction**: Naive Bayes with SMOTE achieves AUC = 0.86 for severe weather cancellations
- **Temporal Patterns**: Clear cascading delay effect throughout the day, independent of weather
- **Class Imbalance**: 82% of flights are on-time (≤15 min delay), creating classification challenges

## Repository Structure
```
MLFINALPROJ/
├── data/
│   ├── cleaned/                      # Processed datasets
│   │   ├── cleaned_bos_flights_1.csv
│   │   ├── filtered_bos_flights.csv
│   │   ├── merged_flight_weather_data.csv
│   │   └── weather_data.xlsx
│   └── weather_api/                  # Raw weather JSON files (2019-2023)
│
├── notebooks/
│   ├── MLproj.ipynb                  # Main analysis notebook
│   └── naiveBayesCancellationClassifier.ipynb
│
├── scripts/
│   ├── filtered_data.py              # Initial BOS flight filtering
│   ├── cleanFlightData1.py           # Flight data preprocessing
│   ├── getWeatherData.py             # Weather API data collection
│   ├── weatherDataToExcel.py         # JSON to Excel conversion
│   ├── joinWeatherData.py            # Merge flight + weather data
│   │
│   ├── linearRegressionAirTime.py    # Air-time baseline model
│   ├── RidgeRegressionAirTime.py     # Air-time with regularization
│   ├── linearRegressionSTDdelay.py   # Delay variability analysis
│   └── naiveBayesCancellationClassifier.py  # Cancellation prediction
│
├── results/                          # Generated figures
│   ├── on_time_vs_delayed.png
│   ├── delay_histogram.png
│   ├── box_by_airline.png
│   └── humidity_vs_delay.png
│
├── dates.txt                         # Date ranges for weather API
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites
```
Python 3.8+
pip or conda package manager
WeatherAPI.com API key (for data collection)
```

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/MLFINALPROJ.git
cd MLFINALPROJ
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

Required libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imblearn
- openpyxl (for Excel files)
- python-dotenv (for API keys)

**3. Set up environment variables** (for weather data collection only)

Create a `.env` file in the root directory:
```
WEATHER_API_KEY=your_api_key_here
```

## Data Sources

### Flight Data
- **Source**: [Kaggle Airline Delay Dataset](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay)
- **Coverage**: 2019-2023 domestic flights
- **Size**: 55,394 flights departing from BOS after filtering
- **Key Features**: Departure time, delay duration, airline, distance, cancellation status

### Weather Data
- **Source**: [WeatherAPI.com](https://www.weatherapi.com/)
- **Coverage**: Hourly observations at BOS (2019-2023)
- **Features**: Temperature, humidity, wind speed/direction, visibility, precipitation, snow, cloud cover

## Data Pipeline

### 1. Data Collection
```bash
# Fetch weather data from API (requires API key)
python scripts/getWeatherData.py

# Convert JSON weather files to Excel
python scripts/weatherDataToExcel.py
```

### 2. Data Preprocessing
```bash
# Filter for Boston flights only
python scripts/filtered_data.py

# Clean and standardize flight data
python scripts/cleanFlightData1.py

# Merge flight and weather datasets
python scripts/joinWeatherData.py
```

**Output**: `data/cleaned/merged_flight_weather_data.csv` (ready for modeling)

### 3. Run Models

**Departure Delay Prediction:**
```python
# See MLproj.ipynb for full pipeline including:
# - Linear, Ridge, Lasso, Elastic Net
# - Random Forest, Gradient Boosting
# - Logistic Regression, SVM, KNN classification
```

**Air-Time Prediction:**
```bash
python scripts/RidgeRegressionAirTime.py
```

**Cancellation Prediction:**
```bash
python scripts/naiveBayesCancellationClassifier.py
```

**Delay Variability Analysis:**
```bash
python scripts/linearRegressionSTDdelay.py
```

## Models and Performance

### Regression (Delay Duration Prediction)

| Model | R² | RMSE (min) | MAE (min) |
|-------|-----|-----------|----------|
| Linear Regression | 0.035 | 45.26 | 22.29 |
| Ridge | 0.035 | 45.26 | 22.29 |
| Lasso | 0.035 | 45.26 | 22.29 |
| Elastic Net | 0.035 | 45.26 | 22.29 |
| Random Forest | 0.050 | 44.91 | 21.69 |
| **Gradient Boosting** | **0.069** | **44.46** | **21.81** |

### Classification (On-Time vs Delayed)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.823 | 0.823 | 1.000 | 0.903 | 0.639 |
| **SVM** | **0.823** | **0.823** | **1.000** | **0.903** | **0.666** |
| KNN (K=5) | 0.793 | 0.824 | 0.952 | 0.883 | 0.605 |

**Note**: High accuracy is misleading due to 82% class imbalance toward on-time flights.

### Air-Time Prediction (Ridge Regression)
- **R² = 0.98** (excellent fit)
- **Distance coefficient**: 0.125 min/mile (~8 miles per minute of flight time)
- **Wind coefficient**: -0.157 min/mph tailwind (~7.5 mph tailwind = 1 min saved)

### Cancellation Prediction (Naive Bayes + SMOTE)
- **AUC = 0.86** (good discrimination)
- **Accuracy = 77.3%**
- **Conditions**: Moderate/heavy snow, freezing rain

## Key Insights

### Weather Impact is Limited
- Weather variables explain <10% of delay variance
- Temperature, visibility, and precipitation are top predictors
- Most delays are caused by **unobserved operational factors**:
  - Mechanical issues
  - Crew scheduling
  - Air traffic control restrictions
  - Cascading network delays

### Temporal Patterns Matter More
- **Average delay increases 3x from 5 AM (~2 min) to 8 PM (~25 min)**
- Delays cascade throughout the day regardless of weather
- Early morning flights are most reliable

### Where Weather Does Matter
1. **Flight cancellations** during severe weather (AUC = 0.86)
2. **Air-time duration** via wind effects (R² = 0.98)
3. **Delay variability** increases with poor weather conditions

### Airline Differences
- Significant variation in median delays across carriers
- Suggests operational practices > shared weather conditions

## Methodology Highlights

### Feature Engineering
- **East-west wind component**: Assumes westward domestic travel from Boston
- **20-minute time bins**: Aggregates delays for temporal analysis
- **Weather condition stratification**: Separate models per weather type

### Handling Class Imbalance
- **SMOTE** (Synthetic Minority Over-sampling) for cancellation prediction
- **Class weighting** in classification models
- **Stratified K-fold cross-validation**

### Model Selection
- **K-fold CV (k=5)** for all models
- **Grid search** for hyperparameter tuning (Ridge alpha, KNN neighbors)
- **AUC optimization** for cancellation prediction (prioritizes true positives)

## Visualizations

The project generates comprehensive visualizations:
- Class distribution and delay histograms
- Feature importance rankings
- ROC curves and confusion matrices
- Residual plots for regression diagnostics
- Time-of-day delay patterns
- Airline-specific delay distributions
- 3D surface plots for air-time prediction

## Future Work

1. **Incorporate operational data**:
   - Aircraft tail numbers (track cascading delays)
   - Gate assignments
   - Crew schedules
   - Real-time air traffic data

2. **Network analysis**:
   - Model delay propagation through route networks
   - Multi-airport coordination effects

3. **Real-time prediction**:
   - Deploy models with live weather feeds
   - Update predictions as departure approaches

4. **Advanced techniques**:
   - LSTM/GRU for time-series modeling
   - Attention mechanisms for feature weighting
   - Ensemble stacking

## Contributors

- **Steven Lam** - Data acquisition, research, documentation
- **Zachary Diringer** - Cancellation prediction, linear regression models, air-time prediction
- **Kyle Murrah** - Data acquisition, research, documentation
- **Pramukh Koushik** - Classification models (Logistic, SVM, KNN), ROC/confusion matrix analysis

## License

This project is for educational purposes as part of a machine learning course at Northeastern University.

## References

1. Kaggle Airline Delay Dataset: https://www.kaggle.com/datasets/sriharshaeedala/airline-delay
2. WeatherAPI.com: https://www.weatherapi.com/
3. Related academic papers cited in final report

## Acknowledgments

- Northeastern University ECE Department
- WeatherAPI.com for weather data access
- Kaggle for flight data availability