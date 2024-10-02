import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def psm_compute_standardized_mean_difference_binary(treated, control, covariate):
    """
    Compute the standardized mean difference (SMD) for a binary covariate.

    The standardized mean difference is calculated as the difference in proportions
    between the treated and control groups, divided by the pooled standard deviation
    of the binary covariate across the two groups.

    Parameters:
    treated (pandas.DataFrame): The treated group data.
    control (pandas.DataFrame): The control group data.
    covariate (str): The name of the binary covariate for which to compute the SMD.

    Returns:
    float: The standardized mean difference (SMD) for the covariate.
    """
    prop_treated = treated[covariate].mean()
    prop_control = control[covariate].mean()
    pooled_prop = (prop_treated + prop_control) / 2
    standardized_mean_difference = (prop_treated - prop_control) / np.sqrt(
        pooled_prop * (1 - pooled_prop)
    )
    return standardized_mean_difference


def psm_compute_standardized_mean_difference_continuous(treated, control, covariate):
    """
    Compute the standardized mean difference (SMD) for a continuous covariate.

    The standardized mean difference is calculated as the difference in means
    between the treated and control groups, divided by the pooled standard deviation
    of the covariate across the two groups.

    Parameters:
    treated (pandas.DataFrame): The treated group data.
    control (pandas.DataFrame): The control group data.
    covariate (str): The name of the covariate for which to compute the SMD.

    Returns:
    float: The standardized mean difference (SMD) for the covariate.
    """
    mean_treated = treated[covariate].mean()
    mean_control = control[covariate].mean()
    std_treated = treated[covariate].std()
    std_control = control[covariate].std()
    pooled_std = np.sqrt((std_treated**2 + std_control**2) / 2)
    standardized_mean_difference = (mean_treated - mean_control) / pooled_std
    return standardized_mean_difference


def psm_encode_categorical_variables(df, columns):
    """
    Encodes categorical variables in the given DataFrame using one-hot encoding.

    Parameters:
        df (pd.DataFrame): Input DataFrame with categorical columns to encode.
        columns (list): List of column names to encode.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical variables.
    """
    # Perform one-hot encoding on specified columns
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    return df_encoded


def psm_get_course_codes(df):
    """
    Returns the unique course codes from the 'course_code' column in the provided DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'course_code' column.

    Returns:
        np.ndarray: Array of unique course codes.
    """
    return df["course_code"].unique()


def psm_get_matches_per_course(df_psm, course_codes, columns_to_match_on):
    """
    Performs propensity score matching for each unique course in the provided DataFrame.
    Matches treated and control groups based on the specified matching columns using propensity scores.

    Parameters:
        df_psm (pd.DataFrame): DataFrame containing the data for PSM analysis.
        course_codes (list): List of unique course codes to iterate through.
        columns_to_match_on (list): List of columns to be used for propensity score modeling.

    Returns:
        pd.DataFrame: A DataFrame containing matched pairs of treated and control groups for each course.
        list: A list of course codes that did not have both control and treatment groups.
    """
    matched_pairs = []
    courses_without_both_control_treatment = []

    for course in course_codes:
        # Subset data for the current course
        df_course = df_psm[df_psm["course_code"] == course]

        # Check if both treatment and control groups are present
        if df_course["did_use_spark"].nunique() < 2:
            print(f"Course {course} did not have both control and treatment groups.")
            courses_without_both_control_treatment.append(course)
            continue  # Skip if only one group is present

        # Prepare data for propensity score model
        X = df_course[columns_to_match_on]
        y = df_course["did_use_spark"]

        # Fit logistic regression model for propensity scores
        model = LogisticRegression(solver="liblinear")
        model.fit(X, y)

        # Get propensity scores
        df_course = df_course.copy()  # Avoid SettingWithCopyWarning
        df_course["propensity_score"] = model.predict_proba(X)[:, 1]

        # Separate treated and control groups with propensity scores
        treated = df_course[df_course["did_use_spark"]].reset_index(drop=True)
        control = df_course[~df_course["did_use_spark"]].reset_index(drop=True)

        # Use Nearest Neighbors to find matches based on propensity scores
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            control[["propensity_score"]]
        )
        distances, indices = nbrs.kneighbors(treated[["propensity_score"]])

        # Get matched control observations
        control_matches = control.loc[indices.flatten()].reset_index(drop=True)

        # Add prefixes to columns to distinguish between treated and control
        treated = treated.add_prefix("treated_")
        control_matches = control_matches.add_prefix("control_")

        # Concatenate treated and control matches side by side
        matched_course = pd.concat([treated, control_matches], axis=1)
        matched_pairs.append(matched_course)

    # Combine matched pairs from all courses
    df_psm_matches = pd.concat(matched_pairs, ignore_index=True)

    return df_psm_matches, courses_without_both_control_treatment


def psm_plot_standardized_mean_differences(smd_before, smd_after):
    """
    Plot the standardized mean differences (SMD) for covariates before and after matching.

    This function generates a scatter plot showing the absolute standardized mean differences
    for covariates in the treated and control groups before and after matching or adjustment.
    It is used to assess covariate balance, and a threshold line at 0.1 is included to indicate
    acceptable balance levels.

    This type of plot is occasionally referred to as a "love plot" in the context of propensity
    score matching or causal inference, named after Thomas Love. It visualizes how well covariates
    are balanced after the matching process compared to before matching.

    Parameters:
    smd_before (dict): A dictionary containing the SMDs for each covariate before matching.
                       Keys are covariate names, and values are the SMDs.
    smd_after (dict): A dictionary containing the SMDs for each covariate after matching.
                      Keys are covariate names, and values are the SMDs.

    Returns:
    None: Displays the plot.
    """
    covariates = list(smd_before.keys())
    smd_before_values = [abs(smd_before[cov]) for cov in covariates]
    smd_after_values = [abs(smd_after[cov]) for cov in covariates]

    fig, ax = plt.subplots(figsize=(8, len(covariates) * 0.5))
    y_pos = np.arange(len(covariates))

    ax.scatter(smd_before_values, y_pos, label="Before Matching", color="red")
    ax.scatter(smd_after_values, y_pos, label="After Matching", color="blue")
    ax.axvline(0.1, color="grey", linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Covariate Balance Before and After Matching")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def psm_plot_standardized_mean_differences_all_three(
    smd_before, smd_after, smd_after_filtering
):
    # Renaming the covariates for better readability
    covariate_names = {
        "age_at_entry": "Age",
        "gender_M": "Gender (Male)",
        "ethnicity_Asian": "Ethnicity (Asian)",
        "ethnicity_Black or African American": "Ethnicity (Black or African American)",
        "ethnicity_Hispanics of any race": "Ethnicity (Hispanics of any race)",
        "ethnicity_Native Hawaiian or Other Pacific Islander": "Ethnicity (Native Hawaiian or Other Pacific Islander)",
        "ethnicity_Race and Ethnicity unknown": "Ethnicity (Race and Ethnicity unknown)",
        "ethnicity_Two or more races": "Ethnicity (Two or more races)",
        "ethnicity_White": "Ethnicity (White)",
    }

    # Map the renamed covariates to their SMD values
    covariates = [covariate_names.get(cov, cov) for cov in smd_before.keys()]
    smd_before_values = [abs(smd_before[cov]) for cov in smd_before.keys()]
    smd_after_values = [abs(smd_after[cov]) for cov in smd_after.keys()]
    smd_after_filtering_values = [
        abs(smd_after_filtering[cov]) for cov in smd_after_filtering.keys()
    ]

    fig, ax = plt.subplots(figsize=(10, len(covariates) * 0.5))
    y_pos = np.arange(len(covariates))

    # Plot SMDs for each stage with sequential colors
    ax.scatter(
        smd_before_values, y_pos, label="Before Matching", color="#c6dbef"
    )  # Light Blue
    ax.scatter(
        smd_after_values, y_pos, label="After Matching", color="#6baed6"
    )  # Medium Blue
    ax.scatter(
        smd_after_filtering_values,
        y_pos,
        label="After Age Difference Post-Filtering",
        color="#08519c",
    )  # Dark Blue

    # Vertical line for SMD threshold
    ax.axvline(0.1, color="grey", linestyle="--")

    # Plot labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title(
        "Covariate Balance Before Matching, After Matching, and After Age Difference Post-Filtering"
    )
    ax.legend()
    plt.gca().invert_yaxis()  # Invert y-axis for traditional Love plot orientation
    plt.tight_layout()
    plt.show()
