# my_stats_tools.py

# Include necessary imports here
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def summarize_data(data, boxplot=False):
    """
    Summarizes the data by calculating various statistics.

    Args:
        data (list, Series, or DataFrame): Numerical values.
            - If list/array/Series: treated as a single column of values.
            - If DataFrame: must contain only one numerical column.
        boxplot (bool): If True, displays a boxplot of the data. Defaults to False.

    Returns:
        tuple: (DataFrame of sorted values, dictionary of summary statistics)
    """
    # Handle empty input
    if (
        data is None
        or (isinstance(data, (list, pd.Series)) and len(data) == 0)
        or (isinstance(data, pd.DataFrame) and data.shape[0] == 0)
    ):
        print("Data is empty.")
        return None, None

    # Step 1: Convert input to DataFrame
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            df = data.copy()
            df.columns = ["label"]
        else:
            raise ValueError(
                "DataFrame must contain exactly one numerical column for summarization."
            )
    else:
        df = pd.DataFrame(data, columns=["lablel"])

    # Step 2: Sort the DataFrame
    df.sort_values(by="label", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 3a: Calculate summary statistics
    n = len(df)
    minimum = df["label"].min()
    maximum = df["label"].max()
    data_range = maximum - minimum
    mean = df["label"].mean()
    median = df["label"].median()
    std_dev = df["label"].std()
    variance = df["label"].var()
    cv = (std_dev / mean) if mean != 0 else float("inf")
    skewness = df["label"].skew()
    kurtosis = df["label"].kurtosis()

    # Step 3b: Calculate mode(s)
    value_counts = df["label"].value_counts()
    if value_counts.nunique() == 1:
        mode = None
        mode_str = "Mode: No Mode"  # No mode
    else:
        mode = df["label"].mode().tolist()
        if len(mode) == 1:
            mode_str = f"Mode: {mode[0]}"
        else:
            mode_str = f"Modes: {', '.join(map(str, mode))}"

    # Step 3c: Determine quartiles
    q1_index = round(n * 0.25)
    q3_index = round(n * 0.75)
    q1 = df["label"].iloc[q1_index]
    q3 = df["label"].iloc[q3_index]
    iqr = q3 - q1

    # Step 4: Print summary statistics
    print("Summary Statistics:")
    print("-------------------")
    print(f"Count: {n}")
    print(f"Minimum: {minimum}")
    print(f"Q1: {q1}")
    print(f"Median: {median:.3f}")
    print(f"Q3: {q3}")
    print(f"Maximum: {maximum}")
    print(f"Range: {data_range}")
    print(f"IQR: {iqr}")
    print(mode_str)
    print(f"Mean: {mean:.3f}")
    print(f"Variance: {variance:.3f}")
    print(f"Standard Deviation: {std_dev:.3f}")
    print(f"Coefficient of Variation: {cv:.3f}")
    print()
    print("Shape of Distribution:")
    print("----------------------")
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")

    # Step 5 (optional): Print a boxplot
    if boxplot is True:
        sns.boxplot(x=df["label"], width=0.3)
        plt.title("Boxplot")
        plt.xlabel("Values")
        plt.show()

    # Consider returning the sorted DataFrame and a dictionary of statistics for further use
    # stats = {
    #     "Count": n,
    #     "Minimum": minimum,
    #     "Q1": q1,
    #     "Median": median,
    #     "Q3": q3,
    #     "Maximum": maximum,
    #     "Range": data_range,
    #     "IQR": iqr,
    #     "Mode": mode,
    #     "Mean": mean,
    #     "Variance": variance,
    #     "Standard Deviation": std_dev,
    #     "Coefficient of Variation": cv,
    #     "Skewness": skewness,
    #     "Kurtosis": kurtosis
    # }
    # return df, stats


def create_frequency_table(data, bin_start=None, step=None, histogram=False):
    """
    Generates and prints a frequency table for the given data.

    Args:
        data (list, Series, or DataFrame): Numerical values.
        bin_start (int): Optional argument to specify starting point for bins. Defaults to min(data_list).
        step (int): Optional argument to specify bin size. Defaults to None, which uses Sturges' formula.
        histogram (bool): If True, displays a histogram of the data. Defaults to False.
    """
    # Handle empty input
    if (
        data is None
        or (isinstance(data, (list, pd.Series)) and len(data) == 0)
        or (isinstance(data, pd.DataFrame) and data.shape[0] == 0)
    ):
        print("Data is empty.")
        return None, None

    # Step 1: Convert input to DataFrame, then sort
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            df = data.copy()
            df.columns = ["label"]
        else:
            raise ValueError(
                "DataFrame must contain exactly one numerical column for summarization."
            )
    else:
        df = pd.DataFrame(data, columns=["label"])
    df.sort_values(by="label", inplace=True)
    df.reset_index(drop=True, inplace=True)
    data_min = np.min(data)
    data_max = np.max(data)

    # Step 2: Determine bins
    start = bin_start if bin_start is not None else data_min

    if step is not None and step > 0:
        bins = np.arange(start, data_max + step, step)
    else:
        n = len(data)
        k = round(math.ceil(np.log2(n) + 1))  # number of bins
        step = round((np.max(data) - np.min(data)) / k)
        bins = np.arange(start, data_max + step, step)

    # Step 3: Bin the data
    df["binned"] = pd.cut(df["label"], bins=bins, right=False)

    # Step 4: Calculate frequencies
    freq = df["binned"].value_counts().sort_index()
    rel_freq = df["binned"].value_counts(normalize=True).sort_index()
    cum_freq = rel_freq.cumsum()  # Cumulative sum of the relative frequency

    # Step 5: Combine into a single table
    freq_table = pd.DataFrame(
        {
            "Frequency": freq,
            "Relative Frequency": rel_freq,
            "Cumulative Frequency": cum_freq,
        }
    )
    print("Frequency Distribution Table:")
    print("-----------------------------")
    print(freq_table)

    # Consider adding a histogram here as an optional argument
    if histogram is True:
        sns.histplot(df["label"], bins=bins, kde=True)
        print("\nHistogram:")
        print("----------")
        plt.show()


def create_contingency_table(df, verbose=False):
    """
    Generates and prints a contingency table for two categorical variables. Useful when interpreting raw data, such as test results, from another source, such as a study or survey.

    Args:
        df (DataFrame): A pandas DataFrame with exactly two categorical columns.
        verbose (bool): If True, prints information on sensitivity, specificity, PPV, and NPV. Defaults to False. These calculations assume a hardcoded DataFrame structure where the first column is the condition status ("Yes" or "No") and the second column is the test result ("Positive" or "Negative").

    Returns:
        A DataFrame that represents a 2x2 contingency table with counts and marginal totals.
    """
    # Handle input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if df.shape[1] != 2:
        raise ValueError("DataFrame must have exactly two columns.")

    # Create the contingency table
    contingency_table = pd.crosstab(
        df.iloc[:, 0], df.iloc[:, 1], margins=True, margins_name="Total"
    )
    print("Contingency Table:")
    print("------------------")
    print(contingency_table)

    # Optional verbose output
    if verbose:
        prob_given_condition = pd.crosstab(
            df.iloc[:, 0], df.iloc[:, 1], normalize="index"
        )
        prob_given_test = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1], normalize="columns")
        print("\nTest Characteristics:")
        print("---------------------")
        print(
            f"Sensitivity (True Positive Rate): {prob_given_condition.loc['Yes', 'Positive']:.2%}"
        )
        print(
            f"Specificity (True Negative Rate): {prob_given_condition.loc['No', 'Negative']:.2%}"
        )
        print(
            f"Positive Predictive Value (PPV): {prob_given_test.loc['Yes', 'Positive']:.2%}"
        )
        print(
            f"Negative Predictive Value (NPV): {prob_given_test.loc['No', 'Negative']:.2%}"
        )


def diagnostic_performance(tp, fn, fp, tn, verbose=False):
    """
    Calculates and prints diagnostic performance metrics from known contingency table results.

    Args:
        tp (int): True Positives
        fn (int): False Negatives
        fp (int): False Positives
        tn (int): True Negatives
        verbose (bool): If True, prints a contingency table. Defaults to False.
    """
    # Validate inputs
    for val in [tp, fn, fp, tn]:
        if not isinstance(val, int) or val < 0:
            raise ValueError("All inputs must be positive integers.")

    # Construct DataFrame
    data = {"Condition Positive": [tp, fn], "Condition Negative": [fp, tn]}
    table = pd.DataFrame(data, index=["Test Positive", "Test Negative"])

    # Add totals
    table.loc["Total"] = table.sum()
    table["Total"] = table.sum(axis=1)

    # Performance metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (
        (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else float("nan")
    )

    # Print results
    if verbose:
        print("Contingency Table:")
        print("------------------")
        print(table)
        print()
    print("Diagnostic Performance Metrics:")
    print("-------------------------------")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2%}")
    print(f"Specificity (True Negative Rate): {specificity:.2%}")
    print(f"Positive Predictive Value (PPV): {ppv:.2%}")
    print(f"Negative Predictive Value (NPV): {npv:.2%}")
    print(f"Accuracy: {accuracy:.2%}")


def create_stem_and_leaf(data_list, title="Stem-and-Leaf Display"):
    """
    Generates and prints a simple stem-and-leaf display.

    Args:
        data_list (list): A list of integers to display.
        title (str): An optional title for the plot.

    Returns:
        A printed stem-and-leaf plot.
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
    """
    Uses Sturges' formula to calculate an appropriate bin width for creating a histogram.

    Args:
        data_list (list): A list of numerical values.

    Returns:
        An integer representing recommended bin width.
    """
    n = len(data_list)
    k = math.ceil(np.log2(n) + 1)  # number of bins
    step = (np.max(data_list) - np.min(data_list)) / k
    return step


# You can add other functions to this file in the future!
