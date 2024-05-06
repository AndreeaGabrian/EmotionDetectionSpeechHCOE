import numpy as np
import pandas as pd

# Load Excel file into DataFrame
# df = pd.read_excel('data_with_sma.xlsx')


def extract_miliseconds_emo_type_from_xlsx_person(filename, person, emotion):
    """
    It extracts emotion value given by face emotion recognition alg at each millisecond
    for person A or B
    :param emotion: emotion name: Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral
    :param filename: the name of the big xlsx file wit all correlated results
    :param person: "A" or "B"
    :return: a dataframe with just the requested emotion values at each millisecond for the corresponding person
    """
    # Load Excel file into DataFrame
    df = pd.read_excel(filename, header=None)
    # Drop the first row
    df = df.drop(df.index[0])
    new_header = df.iloc[0].tolist()  # Get the first row as the new header
    df = df[1:]  # Delete the first row (old header)
    df.columns = new_header  # Assign the new header
    p = 0
    if person == 'B':
        p = 1
    return df[emotion].iloc[:, p]  # return the emotion column for person A or B


def pearson_correlation(x, y):  # it should be calculated between raw or sma values
    """
    Calculate Pearson correlation coefficient between two data series.

    Args:
        x: dataframe column
        y: dataframe column
        x and y must have the same length

    Returns:
        Pearson correlation coefficient between x and y.
    """
    return x.corr(y)  # from pandas library pearson correlation


# Function to calculate SMA3
def calculate_sma(data, sma_type_value):
    """
    Calculate sma3, sma5 or sma10 according to sma_type_value
    :param data: a dataframe column with emotion values from frames
    :param sma_type_value: int, 3000 for sma3, 5000 for sma5, 10000 for sma10
    :return: a list of calculated sma values
    """
    sma3_values = []
    for i in range(len(data)):
        start_index = max(0, i - sma_type_value - 1)  # Start index for the sum
        end_index = i + 1  # End index for the sum
        sum_emo = sum(data[start_index:end_index])  # Calculate sum of EMO values
        non_zero_count = sum(1 for x in data[start_index:end_index] if x != 0)  # Count non-zero values
        if non_zero_count != 0:
            sma3_values.append(sum_emo / non_zero_count)  # Calculate SMA value and append to list
        else:
            sma3_values.append(0)  # If all values are zero, set SMA3 to 0
    return sma3_values


# Function to calculate "raw" metric
def calculate_raw(data):
    raw_values = []
    for i in range(len(data)):
        sum_emo = sum(data[:i+1])  # Calculate sum of EMO values for all previous milliseconds
        non_zero_count = sum(1 for x in data[:i+1] if x != 0)  # Count non-zero values
        if non_zero_count != 0:
            raw_values.append(sum_emo / non_zero_count)  # Calculate "raw" metric and append to list
        else:
            raw_values.append(0)  # If all values are zero, set "raw" metric to 0
    return raw_values


# # Add "raw" column to DataFrame
# df['raw'] = calculate_raw(df['EMO'])
# print("done")
#
# # Add SMA3 column to DataFrame
# df['SMA3'] = calculate_sma(df['EMO'], 3000)
# print("done")
#
# # Add SMA5 column to DataFrame
# df['SMA5'] = calculate_sma(df['EMO'], 5000)
# print("done")
#
# # Add SMA10 column to DataFrame
# df['SMA10'] = calculate_sma(df['EMO'], 10000)
# print("done")

# df['pearson'] = pearson_correlation(df['raw'],df['SMA3'])
#
# # Write DataFrame to Excel
# df.to_excel('data_with_sma2.xlsx', index=False)

