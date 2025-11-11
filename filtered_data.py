import pandas as pd
# if DES is equal to BOS or ORI is equal to BOS, keep row

def filter_bos_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows where the destination (DES) 
    is 'BOS' or the origin (ORI) is 'BOS'.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing flight data with 'DES' and 'ORI' columns.

    Returns:
    pd.DataFrame: A filtered DataFrame containing only the rows where DES is 'BOS' or ORI is 'BOS'.
    """
    filtered_df = df[(df['DEST_CITY'] == 'Boston, MA') | (df['ORIGIN_CITY'] == 'Boston, MA')]
    return filtered_df

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('flights_sample_3m.csv')
    df = pd.DataFrame(data)
    filtered_df = filter_bos_flights(df)
    filtered_df.to_csv('filtered_bos_flights.csv', index=False)