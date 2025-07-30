import pandas as pd

def loading_house_data(csv_path):
    '''    
    Wczytuje dane mieszkaniowe z pliku

    Parameters:
    csv_path (str): ściezka do pliku .csv

    Returns:
    pd.DataFrame: DataFrame z danymi mieszkaniowymi
    '''
    return pd.read_csv(csv_path)