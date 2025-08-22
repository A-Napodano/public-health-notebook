# my_stats_tools.py

# Include necessary imports here
import math
import numpy as np 
import pandas as pd

def summarize_data(data_list):
    """
    Summarizes the data by calculating various statistics.

    Args:
        data_list (list): A list of numerical values.

    Returns:
        A list of summary statistics.
    """
    if not data_list:
        print("Data list is empty.")
        return None, None
    
     # Step 1: Convert to DataFrame
    df = pd.DataFrame(data_list, columns=['label'])

    # Step 2: Sort the DataFrame
    df.sort_values(by='label', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 3a: Calculate summary statistics
    n = len(df)
    minimum = df['label'].min()
    maximum = df['label'].max()
    data_range = maximum - minimum
    mean = df['label'].mean()
    median = df['label'].median()
    std_dev = df['label'].std()
    variance = df['label'].var()
    cv = (std_dev / mean) if mean != 0 else float('inf')
    skewness = df['label'].skew()
    kurtosis = df['label'].kurtosis()
   
    # Step 3b: Calculate mode(s)
    value_counts = df['label'].value_counts()
    if value_counts.nunique() == 1:
        mode = None
        mode_str = "Mode: No Mode"  # No mode
    else:
        mode = df['label'].mode().tolist()
        if len(mode) == 1:
            mode_str = f"Mode: {mode[0]}"
        else:
            mode_str = f"Modes: {', '.join(map(str, mode))}"

    # Step 3c: Determine quartiles
    q1_index = round(n * 0.25)
    q3_index = round(n * 0.75)
    q1 = df['label'].iloc[q1_index]
    q3 = df['label'].iloc[q3_index]
    iqr = q3 - q1
    
    # Step 4: Print summary statistics
    print("Summary Statistics:")
    print("-------------------")
    print(f"Count: {n}")
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")
    print(f"Range: {data_range}")
    print(f"Mean: {mean:.3f}")
    print(f"Median: {median:.3f}")
    print(mode_str)
    print(f"Variance: {variance:.3f}")
    print(f"Standard Deviation: {std_dev:.3f}")
    print(f"Coefficient of Variation: {cv:.3f}")
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"IQR: {iqr}")
    print()
    print("Shape of Distribution:")
    print("----------------------")
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    # Consider returning the sorted DataFrame and a dictionary of statistics for further use

def frequency_table(data_list, step=None, bin_start=None):
    """
    Generates and prints a frequency table for the given data.

    Args:
        data_list (list): A list of numerical values.
        step (int): Optional argument to specify bin size.
        bin_start (int): Optional argument to specify starting point for bins. Defaults to min(data_list).
    """
    if not data_list:
        print("Data list is empty.")
        return None
    
    # Step 1: Convert to DataFrame, then sort
    df = pd.DataFrame(data_list, columns=['label'])
    df.sort_values(by='label', inplace=True)
    df.reset_index(drop=True, inplace=True)
    data_min = np.min(data_list)
    data_max = np.max(data_list)

    # Step 2: Determine bins
    start = bin_start if bin_start is not None else data_min

    if step is not None and step > 0:
        bins = np.arange(start, data_max + step, step)
    else:
        n = len(data_list)
        k = round(math.ceil(np.log2(n) + 1)) # number of bins
        step = round((np.max(data_list) - np.min(data_list)) / k)
        bins = np.arange(start, data_max + step, step)
    
    # Step 3: Bin the data
    df['binned'] = pd.cut(df['label'], bins=bins, right=False)

    # Step 4: Calculate frequencies
    freq = df['binned'].value_counts().sort_index()
    rel_freq = df['binned'].value_counts(normalize=True).sort_index()
    cum_freq = rel_freq.cumsum() # Cumulative sum of the relative frequency

    # Step 5: Combine into a single table
    freq_table = pd.DataFrame({
        'Frequency': freq,
        'Relative Frequency': rel_freq,
        'Cumulative Frequency': cum_freq
    })
    print("Frequency Distribution Table:")
    print("-----------------------------")
    print(freq_table)

def create_stem_and_leaf(data_list, title="Stem-and-Leaf Display"):
    """
    Generates and prints a simple stem-and-leaf display for 2-digit numbers.

    Args:
        data_list (list): A list of integers to display.
        title (str): An optional title for the plot.
    """
    print(title)
    print("-" * len(title))

    if not data_list:
        print("Data list is empty.")
        return

    stem_leaf = {}
    data_list.sort()

    for num in data_list:
        stem = num // 10
        leaf = num % 10

        if stem not in stem_leaf:
            stem_leaf[stem] = []
        stem_leaf[stem].append(leaf)

    for stem, leaves in sorted(stem_leaf.items()):
        leaf_str = " ".join(map(str, leaves))
        print(f" {stem} | {leaf_str}")

def sturges_step(data_list):
    n = len(data_list)
    k = math.ceil(np.log2(n) + 1) # number of bins
    step = (np.max(data_list) - np.min(data_list)) / k
    return step


# You can add other functions to this file in the future!