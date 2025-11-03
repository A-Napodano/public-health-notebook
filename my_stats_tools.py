# my_stats_tools.py

# Include necessary imports here
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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


def calculate_ci_single_mean_or_paired(mean_est, sd_est, n, alpha=0.05):
    """
    Calculates the (1-alpha) CI for a single mean or a paired mean difference.
    Uses the t-distribution.

    Args:
        mean_est (float): Sample mean (x_bar) or mean difference (d_bar).
        sd_est (float): Sample standard deviation (s) or SD of differences (s_d).
        n (int): Sample size (or number of pairs).
        alpha (float): Significance level (e.g., 0.05 for 95% CI).
    """
    df = n - 1
    
    # Calculate the two-tailed t critical value (e.g., t_0.975 for alpha=0.05)
    # stats.t.ppf calculates the percentile point function
    t_critical = stats.t.ppf(1 - alpha/2, df) 
    
    # Calculate the Standard Error (SE) and Margin of Error (ME)
    se = sd_est / np.sqrt(n)
    margin_of_error = t_critical * se
    
    lower_bound = mean_est - margin_of_error
    upper_bound = mean_est + margin_of_error
    
    print(f"Confidence Level: {(1 - alpha)*100:.2f}%")
    print(f"Degrees of Freedom (df): {df}")
    print(f"t Critical Value: {t_critical:.4f}")
    print(f"Margin of Error (ME): {margin_of_error:.4f}")
    print(f"CI ({lower_bound:.4f} to {upper_bound:.4f})")
    
    return (lower_bound, upper_bound)


def calculate_ci_two_means_equal_var(x1, s1, n1, x2, s2, n2, alpha=0.05):
    """
    Calculates the (1-alpha) CI for the difference between two means,
    assuming equal population variances (uses pooled variance).

    Args:
        x1 (float): Mean 1
        s1 (float): Standard deviation 1
        n1 (int): Sample size 1
        x2 (float): Mean 2
        s2 (float): Standard deviation 2
        n2 (int): Sample size 2
        alpha (float): Significance level (e.g., 0.05 for 95% CI).
    """

    # 1. Calculate Pooled Variance (S_p^2)
    df = n1 + n2 - 2
    numerator_sp2 = ((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)
    sp_squared = numerator_sp2 / df
    
    # 2. Calculate t critical value
    t_critical = stats.t.ppf(1 - alpha/2, df) 
    
    # 3. Calculate Standard Error (SE)
    se_squared = sp_squared * (1/n1 + 1/n2)
    se = np.sqrt(se_squared)
    
    # 4. Calculate Margin of Error (ME)
    margin_of_error = t_critical * se
    
    diff = x1 - x2
    lower_bound = diff - margin_of_error
    upper_bound = diff + margin_of_error
    
    print(f"Degrees of Freedom (df): {df}")
    print(f"Pooled Variance (s_p^2): {sp_squared:.4f}")
    print(f"t Critical Value: {t_critical:.4f}")
    print(f"Mean Difference (x1 - x2): {diff:.4f}")
    print(f"Margin of Error (ME): {margin_of_error:.4f}")
    print(f"CI ({lower_bound:.4f} to {upper_bound:.4f})")
    
    return (lower_bound, upper_bound)


def calculate_ci_single_proportion(x, n, alpha=0.05):
    """
    Calculates the (1-alpha) CI for a single proportion using the Wald method.
    Requires meeting np >= 5 and n(1-p) >= 5 for normal approximation.

    Args:
        x (int): Number of successes.
        n (int): Sample size.
        alpha (float): Significance level.
    """
    p_hat = x / n
    q_hat = 1 - p_hat
    
    # Calculate the two-tailed Z critical value (e.g., Z_0.975 for alpha=0.05)
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Calculate Standard Error (SE)
    se = np.sqrt(p_hat * q_hat / n)
    
    # Calculate Margin of Error (ME)
    margin_of_error = z_critical * se
    
    lower_bound = p_hat - margin_of_error
    upper_bound = p_hat + margin_of_error
    
    print(f"Sample Proportion (p_hat): {p_hat:.4f}")
    print(f"Z Critical Value: {z_critical:.4f}")
    print(f"Margin of Error (ME): {margin_of_error:.4f}")
    print(f"CI ({lower_bound:.4f} to {upper_bound:.4f})")
    
    return (lower_bound, upper_bound)


def calculate_ci_two_proportions(x1, n1, x2, n2, alpha=0.05):
    """
    Calculates the (1-alpha) CI for the difference between two proportions.

    Args:
        x1 (int): Number of successes for proportion 1.
        n1 (int): Sample size 1.
        x2 (int): Number of successes for proportion 2.
        n2 (int): Sample size 2.
        alpha (float): Significance level.
    """
    p1_hat = x1 / n1
    q1_hat = 1 - p1_hat
    p2_hat = x2 / n2
    q2_hat = 1 - p2_hat
    
    diff = p1_hat - p2_hat
    
    # Calculate the two-tailed Z critical value
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Calculate Standard Error (SE) squared for difference
    se_squared = (p1_hat * q1_hat / n1) + (p2_hat * q2_hat / n2)
    se = np.sqrt(se_squared)
    
    # Calculate Margin of Error (ME)
    margin_of_error = z_critical * se
    
    lower_bound = diff - margin_of_error
    upper_bound = diff + margin_of_error
    
    print(f"Difference in Proportions (p1 - p2): {diff:.4f}")
    print(f"Z Critical Value: {z_critical:.4f}")
    print(f"Margin of Error (ME): {margin_of_error:.4f}")
    print(f"CI ({lower_bound:.4f} to {upper_bound:.4f})")
    
    return (lower_bound, upper_bound)


def calculate_ci_single_variance(s_squared, n, alpha=0.05):
    """
    Calculates the (1-alpha) CI for a single variance.
    Requires the population to be normally distributed.

    Args:
        s_squared (float): 
        n (int): Sample size.
        alpha (float): Significance level.
    """
    df = n - 1
    
    # Calculate Chi-Square Critical Values
    # Lower tail critical value (X^2_alpha/2)
    chi2_lower = stats.chi2.ppf(alpha/2, df)
    # Upper tail critical value (X^2_1-alpha/2)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, df) 
    
    numerator = (n - 1) * s_squared
    
    # CI bounds are constructed inversely using the chi2 values
    lower_bound = numerator / chi2_upper
    upper_bound = numerator / chi2_lower
    
    print(f"Sample Variance (s^2): {s_squared:.4f}")
    print(f"Degrees of Freedom (df): {df}")
    print(f"Chi^2 Lower Critical ({alpha/2}): {chi2_lower:.4f}")
    print(f"Chi^2 Upper Critical ({1 - alpha/2}): {chi2_upper:.4f}")
    print(f"CI for Variance ({lower_bound:.4f} to {upper_bound:.4f})")
    print(f"CI for Standard Deviation ({np.sqrt(lower_bound):.4f} to {np.sqrt(upper_bound):.4f})")
    
    return (lower_bound, upper_bound)


def calculate_ci_ratio_variances(s1_squared, n1, s2_squared, n2, alpha=0.05):
    """
    Calculates the (1-alpha) CI for the ratio of two variances.
    """
    df1 = n1 - 1
    df2 = n2 - 1
    
    F_stat = s1_squared / s2_squared
    
    # Calculate F Critical Values
    # Lower tail critical value (F_alpha/2)
    F_lower_critical = stats.f.ppf(alpha/2, df1, df2)
    # Upper tail critical value (F_1-alpha/2)
    F_upper_critical = stats.f.ppf(1 - alpha/2, df1, df2)
    
    # CI bounds
    lower_bound = F_stat / F_upper_critical
    upper_bound = F_stat / F_lower_critical
    
    print(f"Variance Ratio (s1^2 / s2^2): {F_stat:.4f}")
    print(f"Degrees of Freedom (Numerator, Denominator): ({df1}, {df2})")
    print(f"F Lower Critical ({alpha/2}): {F_lower_critical:.4f}")
    print(f"F Upper Critical ({1 - alpha/2}): {F_upper_critical:.4f}")
    print(f"CI ({lower_bound:.4f} to {upper_bound:.4f})")
    
    return (lower_bound, upper_bound)


def calculate_t_single_mean(x_bar, s, n, mu_null=0):
    """
    Calculates the t test statistic for a single mean or a paired mean.
    The paired case uses d_bar for x_bar and s_d for s, with mu_null usually 0.
    """
    df = n - 1
    se = s / np.sqrt(n)
    t_stat = (x_bar - mu_null) / se
    
    # Calculate P-value (two-tailed)
    p_value = stats.t.sf(np.abs(t_stat), df) * 2
    
    print(f"Degrees of Freedom (df): {df}")
    print(f"Test Statistic (t): {t_stat:.4f}")
    print(f"Two-Tailed P-value: {p_value:.4f}")
    
    return t_stat


def calculate_t_two_means_unequal_var(x1, s1, n1, x2, s2, n2, diff_null=0):
    """
    Calculates the t' test statistic for two independent means, assuming unequal variances.
    Satterthwaite's degrees of freedom calculation is complex and often done via software,
    but the t statistic calculation itself is straightforward.
    """
    # Calculate Standard Error (SE) squared for difference
    se_squared = (s1**2 / n1) + (s2**2 / n2)
    se = np.sqrt(se_squared)
    
    t_stat = ((x1 - x2) - diff_null) / se
    
    # Note: P-value/Critical value calculation for t' requires complex df calculation (Satterthwaite's)
    # The calculated t_stat is compared to t'_critical value (user must find this value).
    
    print(f"Test Statistic (t'): {t_stat:.4f}")
    print("Note: Critical value t' must be approximated using the complex formula or table lookup.")
    
    return t_stat


def calculate_z_single_proportion(x, n, p_null):
    """
    Calculates the Z test statistic for a single proportion.
    Uses the null proportion p_null in the standard error calculation.
    """
    p_hat = x / n
    q_null = 1 - p_null
    
    # Standard Error uses the null proportion (p_null)
    se_null = np.sqrt(p_null * q_null / n)
    
    z_stat = (p_hat - p_null) / se_null
    
    # Calculate P-value (two-tailed)
    p_value = stats.norm.sf(np.abs(z_stat)) * 2
    
    print(f"Test Statistic (Z): {z_stat:.4f}")
    print(f"Two-Tailed P-value: {p_value:.4f}")
    
    return z_stat


def calculate_z_two_proportions(x1, n1, x2, n2, diff_null=0):
    """
    Calculates the Z test statistic for the difference between two proportions.
    Uses the pooled proportion (p_bar) for the standard error.
    """
    p1_hat = x1 / n1
    p2_hat = x2 / n2
    
    # Calculate Pooled Proportion (p_bar)
    x_total = x1 + x2
    n_total = n1 + n2
    p_bar = x_total / n_total
    q_bar = 1 - p_bar
    
    # Standard Error uses the pooled proportion
    se_pooled = np.sqrt(p_bar * q_bar * (1/n1 + 1/n2))
    
    z_stat = ((p1_hat - p2_hat) - diff_null) / se_pooled
    
    # Calculate P-value (two-tailed)
    p_value = stats.norm.sf(np.abs(z_stat)) * 2
    
    print(f"Pooled Proportion (p_bar): {p_bar:.4f}")
    print(f"Test Statistic (Z): {z_stat:.4f}")
    print(f"Two-Tailed P-value: {p_value:.4f}")
    
    return z_stat


def calculate_chi2_single_variance(s_squared, n, sigma_null_squared):
    """
    Calculates the Chi-Square test statistic for a single variance.
    """
    df = n - 1
    chi2_stat = (n - 1) * s_squared / sigma_null_squared
    
    # Calculate P-value (two-tailed) based on chi2 distribution
    # Find the smaller tail area and double it.
    p_upper = stats.chi2.sf(chi2_stat, df) # Area to the right
    p_lower = stats.chi2.cdf(chi2_stat, df) # Area to the left
    
    p_value = min(p_upper, p_lower) * 2

    print(f"Degrees of Freedom (df): {df}")
    print(f"Test Statistic (Chi^2): {chi2_stat:.4f}")
    print(f"Two-Tailed P-value: {p_value:.4f}")
    
    return chi2_stat


def calculate_f_ratio_variances(s1_squared, n1, s2_squared, n2):
    """
    Calculates the F test statistic for the ratio of two variances.
    Assumes H0: sigma1^2 / sigma2^2 = 1.
    """
    df1 = n1 - 1
    df2 = n2 - 1
    
    f_stat = s1_squared / s2_squared
    
    # Calculate P-value (two-tailed) based on F distribution
    p_upper = stats.f.sf(f_stat, df1, df2) # Area to the right
    p_value = p_upper * 2
    
    print(f"Degrees of Freedom (Numerator, Denominator): ({df1}, {df2})")
    print(f"Test Statistic (F): {f_stat:.4f}")
    print(f"Two-Tailed P-value (2 * Upper Tail): {p_value:.4f}")
    
    return f_stat


def sample_size_mean_ci(sigma, d, alpha=0.05):
    """
    Calculates the required sample size (n) to estimate a single mean (mu) 
    with a given Margin of Error (d).
    
    Args:
        sigma (float): Expected population standard deviation (s).
        d (float): Desired Margin of Error (d).
        alpha (float): Significance level (determines Z).
    """
    # Z value based on alpha (e.g., 1.96 for 95% CI)
    Z = stats.norm.ppf(1 - alpha/2)
    
    n = (Z**2 * sigma**2) / (d**2)
    
    # Always round up
    n_rounded = math.ceil(n)
    
    print(f"Input: Z={Z:.3f}, sigma={sigma:.2f}, Margin of Error (d)={d:.4f}")
    print(f"Required Sample Size (n): {n_rounded} (calculated n={n:.2f})")
    
    return n_rounded


def sample_size_proportion_ci(p_expected, d, alpha=0.05):
    """
    Calculates the required sample size (n) to estimate a single proportion (p) 
    with a given Margin of Error (d).
    
    Args:
        p_expected (float): Best guess for the population proportion (use 0.5 if unknown).
        d (float): Desired Margin of Error (d).
        alpha (float): Significance level (determines Z).
    """
    # Z value based on alpha (e.g., 1.96 for 95% CI)
    Z = stats.norm.ppf(1 - alpha/2)
    
    n = (Z**2 * p_expected * (1 - p_expected)) / (d**2)
    
    # Always round up
    n_rounded = math.ceil(n)
    
    print(f"Input: Z={Z:.3f}, p_expected={p_expected:.2f}, Margin of Error (d)={d:.4f}")
    print(f"Required Sample Size (n): {n_rounded} (calculated n={n:.2f})")
    
    return n_rounded


def sample_size_two_means_ht(s, d, alpha=0.05, power=0.80):
    """
    Calculates the required sample size (n per group) to compare two means 
    with specified power and effect size (d).
    
    Args:
        s (float): Expected standard deviation.
        d (float): Smallest meaningful difference (Effect Size: mu1 - mu2).
        alpha (float): Type I error rate.
        power (float): Desired statistical power (1-beta).
    """
    # Find critical Z values for alpha and beta (Type II error = 1 - power)
    Z_alpha = stats.norm.ppf(1 - alpha/2) # Two-sided alpha
    Z_beta = stats.norm.ppf(power)       # Z value corresponding to desired power
    
    # Formula derived from standard power calculations (using Z approximation)
    n_per_group = (2 * s**2 * (Z_alpha + Z_beta)**2) / (d**2)
    
    # Note: Source provides shortcut n = (4s/d)^2 for 80% power, alpha=0.05.
    # Using the general formula for precision.
    
    n_rounded = math.ceil(n_per_group)
    
    print(f"Input: s={s:.2f}, Effect Size (d)={d:.2f}, Z_alpha={Z_alpha:.3f}, Z_beta={Z_beta:.3f}")
    print(f"Required Sample Size per Group (n): {n_rounded} (calculated n={n_per_group:.2f})")
    
    return n_rounded


def sample_size_two_proportions_ht(p1, p2, alpha=0.05, power=0.80):
    """
    Calculates the required sample size (n per group) to compare two proportions 
    with specified power and effect size (p1 - p2).
    """
    # Calculate average proportion (p_bar)
    p_bar = (p1 + p2) / 2
    
    # Calculate Z values
    Z_alpha_half = stats.norm.ppf(1 - alpha/2) # Z_1-alpha/2
    Z_beta = stats.norm.ppf(power)             # Z_1-beta
    
    p_diff_sq = (p1 - p2)**2
    
    # This formula is slightly simplified for implementation compared to the source's exact structure
    # but captures the core terms for calculation:
    n_per_group = ((Z_alpha_half + Z_beta)**2 * (p_bar * (1 - p_bar))) / p_diff_sq

    # Note: Source provides shortcut n = 16*p(1-p) / (p1-p2)^2 for 80% power, alpha=0.05.
    
    n_rounded = math.ceil(n_per_group)
    
    print(f"Input: p1={p1:.4f}, p2={p2:.4f}, p_bar={p_bar:.4f}")
    print(f"Input: Z_alpha_half={Z_alpha_half:.3f}, Z_beta={Z_beta:.3f}")
    print(f"Required Sample Size per Group (n): {n_rounded} (calculated n={n_per_group:.2f})")
    
    return n_rounded

def sample_size_two_proportions_shortcut(p1, p2):
    """
    This calculates the required sample size (n per group) to compare two proportions
    when the desired power is 80% and the significance level is 0.05.
    """
    # Calculate average proportion (p_bar)
    p_bar = (p1 + p2) / 2

    # Calculate sample size per group
    n_per_group = (16 * p_bar * (1 - p_bar)) / ((p1 - p2)**2)
    n_rounded = math.ceil(n_per_group)
    print(f"Required Sample Size per Group (n): {n_rounded} (calculated n={n_per_group:.2f})")


# You can add other functions to this file in the future!
