import pandas as pd
from pathlib import Path

def clean_flight_data(df):
    # Remove rows where departure is not Boston
    df = df[df['ORIGIN'] == 'BOS']

    # Standardize the time format to HH:MM
    # Combine date and time into datetime
# Combine date and time rounded to nearest hour
    df['DEP_DATETIME'] = df.apply(
    lambda row: (
        (pd.to_datetime(row['FL_DATE']) + pd.Timedelta(hours=round(int(row['CRS_DEP_TIME']) / 100)))
        .strftime('%Y-%m-%d %H:00')
        if pd.notnull(row['CRS_DEP_TIME']) else pd.NaT
    ),axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize airline names
    df['airline'] = df['AIRLINE'].str.title().str.strip()

    return df

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

input_file = CLEAN_DIR / "filtered_bos_flights.csv"
output_file = CLEAN_DIR / "cleaned_bos_flights_1.csv"

df = pd.read_csv(input_file)
cleaned_df = clean_flight_data(df)
cleaned_df.to_csv(output_file, index=False)

print("Saved:", output_file)