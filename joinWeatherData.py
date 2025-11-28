import pandas as pd
from pathlib import Path

# File paths
ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

weather_data = CLEAN_DIR / "weather_data.xlsx"
flight_data = CLEAN_DIR / "cleaned_bos_flights_1.csv"

# Load datasets
weather_df = pd.read_excel(weather_data)
flight_df = pd.read_csv(flight_data)

# Ensure datetime columns
weather_df['time'] = pd.to_datetime(weather_df['time'])
flight_df['DEP_DATETIME'] = pd.to_datetime(flight_df['DEP_DATETIME'])

# Round flight departure time to the nearest hour (use lowercase 'h')
flight_df['DEP_HOUR'] = flight_df['DEP_DATETIME'].dt.round('h')

# Round weather timestamps to hour to match flight data
weather_df['DATE_HOUR'] = weather_df['time'].dt.round('h')

# Merge on the hourly timestamp
merged_df = pd.merge(
    flight_df,
    weather_df,
    left_on='DEP_HOUR',
    right_on='DATE_HOUR',
    how='left'
)

# Drop redundant weather datetime column
merged_df = merged_df.drop(columns=['DATE_HOUR'])

# Save merged dataset
output_file = CLEAN_DIR / "merged_flight_weather_data.csv"
merged_df.to_csv(output_file, index=False)

print("Saved:", output_file)