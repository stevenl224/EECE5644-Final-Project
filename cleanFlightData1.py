import pandas as pd
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

filepath = "C:/Users/zacha/EECE5644-Final-Project/filtered_bos_flights.csv"
df = pd.read_csv(filepath)
cleaned_df = clean_flight_data(df)

cleaned_df.to_csv("C:/Users/zacha/EECE5644-Final-Project/cleaned_bos_flights_1.csv", index=False)