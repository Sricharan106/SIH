import pandas as pd


def load_network_data(file_path):
    """
    Load network traffic CSV and perform basic cleaning.
    """

    df = pd.read_csv(file_path)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Replace infinite values
    df = df.replace([float("inf"), float("-inf")], None)

    # Drop completely empty rows
    df = df.dropna(how="all")

    return df
