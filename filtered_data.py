import pandas as pd
# if DES is equal to BOS or ORI is equal to BOS, keep row

def filter_bos_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows where the destination (DEST_CITY) 
    or origin (ORIGIN_CITY) is 'Boston, MA', and delay values are valid and positive.
    """
    df = df.dropna(subset=['ARR_DELAY', 'DEP_DELAY', 'DELAY_DUE_WEATHER'])
    add_row_filter = (
        (df['DEP_DELAY'] > 0.0) | 
        (df['ARR_DELAY'] > 0.0) | 
        (df['DELAY_DUE_WEATHER'] > 0.0)
    )
    filtered_df = df[((df['DEST_CITY'] == 'Boston, MA') | (df['ORIGIN_CITY'] == 'Boston, MA')) & add_row_filter]
    return filtered_df

if __name__ == "__main__":
    data = pd.read_csv('flights_sample_3m.csv')
    filtered_df = filter_bos_flights(data)
    print(filtered_df)
    filtered_df.to_csv('filtered_bos_delay_flights.csv', index=False)
