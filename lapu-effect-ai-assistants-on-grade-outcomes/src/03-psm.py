import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from dotenv import load_dotenv

import utils

# Define the project root
PROJECT_ROOT = "lapu-effect-ai-assistants-on-grade-outcomes"

utils.add_source_root_to_system_path(PROJECT_ROOT)

from src import misc, preprocessing, psm  # noqa: E402

# Load environment variables
dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path)

# Get configs path
config_path = utils.get_configs_path(PROJECT_ROOT)

# Load configs
with open(config_path, "r") as config_file:
    configs = yaml.safe_load(config_file)

# Get file ID
file_id = configs.get("file_id")

# Read Excel file from Drive
df = misc.read_excel_from_drive(file_id)

# =============================================
# 1. Data Preparation
# =============================================

# Clean data
df = preprocessing.clean_data(df)

# Create binary did_use_spark field
df = preprocessing.create_did_use_spark_field(df)

# Select fields
df_psm = misc.select_psm_fields(df)

# Drop missing GPAs
df_psm = preprocessing.drop_missing_gpas(df_psm)

# Drop rows with missing genders
df_psm = preprocessing.drop_missing_genders(df_psm)

# Encode categorical variables
categorical_variables_to_encode = ["gender", "ethnicity"]
df_psm = psm.psm_encode_categorical_variables(df_psm, categorical_variables_to_encode)

# Identify the new one-hot encoded columns
one_hot_columns = df_psm.columns.difference(
    ["age_at_entry", "did_use_spark", "course_code"]
)

# Remove 'course_gpa' and 'id' from one_hot_columns if present
one_hot_columns = [col for col in one_hot_columns if col not in ["course_gpa", "id"]]

# Define columns to match on
columns_to_match_on = ["age_at_entry"] + list(one_hot_columns)

# =============================================
# 2. Assessing Balance Before Matching
# =============================================

# Split the data into treated and control groups before matching
treated_pre = df_psm[df_psm["did_use_spark"]]
control_pre = df_psm[~df_psm["did_use_spark"]]

# List of covariates (after encoding)
covariates = [
    col
    for col in df_psm.columns
    if col not in ["id", "did_use_spark", "course_code", "course_gpa"]
]

smd_before = {}
for covariate in covariates:
    if df_psm[covariate].nunique() == 2:
        # Binary variable
        smd = psm.psm_compute_standardized_mean_difference_binary(
            treated_pre, control_pre, covariate
        )
    else:
        # Continuous variable
        smd = psm.psm_compute_standardized_mean_difference_continuous(
            treated_pre, control_pre, covariate
        )
    smd_before[covariate] = smd

print("\nStandardized Mean Differences Before Matching:")
for covariate, smd in smd_before.items():
    print(f"{covariate}: {smd:.4f}")

# =============================================
# 3. Propensity Score Matching
# =============================================

course_codes = psm.psm_get_course_codes(df_psm)
df_psm_matches, courses_without_both_control_treatment = psm.psm_get_matches_per_course(
    df_psm, course_codes, columns_to_match_on
)

# =============================================
# 4. Assessing Balance After Matching
# =============================================

# Extract treated and control data from matched pairs
treated_post = df_psm_matches[
    [col for col in df_psm_matches.columns if col.startswith("treated_")]
]
control_post = df_psm_matches[
    [col for col in df_psm_matches.columns if col.startswith("control_")]
]

# Remove the 'treated_' and 'control_' prefixes
treated_post.columns = [col.replace("treated_", "") for col in treated_post.columns]
control_post.columns = [col.replace("control_", "") for col in control_post.columns]

# Ensure indices align
treated_post = treated_post.reset_index(drop=True)
control_post = control_post.reset_index(drop=True)

smd_after = {}
for covariate in covariates:
    if treated_post[covariate].nunique() == 2:
        # Binary variable
        smd = psm.psm_compute_standardized_mean_difference_binary(
            treated_post, control_post, covariate
        )
    else:
        # Continuous variable
        smd = psm.psm_compute_standardized_mean_difference_continuous(
            treated_post, control_post, covariate
        )
    smd_after[covariate] = smd

print("\nStandardized Mean Differences After Matching:")
for covariate, smd in smd_after.items():
    print(f"{covariate}: {smd:.4f}")

# =============================================
# 5. Visualizing Balance with Love Plot (just before matching and after matching, no age filtering)
# =============================================

# Plot standardized mean differences (e.g., Love plot)
psm.psm_plot_standardized_mean_differences(smd_before, smd_after)

# Compute the absolute age difference between matched pairs
df_psm_matches["age_diff"] = abs(
    df_psm_matches["treated_age_at_entry"] - df_psm_matches["control_age_at_entry"]
)

# Plot the distribution of age differences
plt.figure(figsize=(10, 6))
sns.histplot(df_psm_matches["age_diff"], bins=20, kde=False)
plt.title("Distribution of Age Differences Between Matched Pairs")
plt.xlabel("Age Difference")
plt.ylabel("Frequency")
plt.grid(axis="y")
plt.show()

# Observation:
# The standardized mean difference for age_at_entry is very high (> limit of 0.1)
# So we filter for matched pairs close in age

# =============================================
# 6. Filtering Age Differences
# =============================================

# Set the age difference threshold
age_diff_threshold = 10

# Identify pairs within the threshold
df_psm_matches_filtered = df_psm_matches[
    df_psm_matches["age_diff"] <= age_diff_threshold
].reset_index(drop=True)

print(f"Number of matched pairs after filtering: {len(df_psm_matches_filtered)}")

# =============================================
# 7. Re-assessing Balance After Matching
# =============================================

# Extract treated and control data from filtered matches
treated_filtered = df_psm_matches_filtered[
    [col for col in df_psm_matches_filtered.columns if col.startswith("treated_")]
]
control_filtered = df_psm_matches_filtered[
    [col for col in df_psm_matches_filtered.columns if col.startswith("control_")]
]

# Remove prefixes
treated_filtered.columns = [
    col.replace("treated_", "") for col in treated_filtered.columns
]
control_filtered.columns = [
    col.replace("control_", "") for col in control_filtered.columns
]

# Recompute SMDs after filtering
smd_after_filtering = {}
for covariate in covariates:
    if treated_filtered[covariate].nunique() == 2:
        # Binary variable
        smd = psm.psm_compute_standardized_mean_difference_binary(
            treated_filtered, control_filtered, covariate
        )
    else:
        # Continuous variable
        smd = psm.psm_compute_standardized_mean_difference_continuous(
            treated_filtered, control_filtered, covariate
        )
    smd_after_filtering[covariate] = smd

print("\nStandardized Mean Differences After Removing Poor Age Matches:")
for covariate, smd in smd_after_filtering.items():
    print(f"{covariate}: {smd:.4f}")

# =============================================
# 8. Visualizing Balance with Love Plot (Before, After Matching, and After Age Filtering)
# =============================================

# Create Love plot
psm.psm_plot_standardized_mean_differences_all_three(
    smd_before, smd_after, smd_after_filtering
)

# Save the plot
plot_path = os.path.join(
    "lapu-effect-ai-assistants-on-grade-outcomes/figures/03-psm-love-plot.png"
)
plt.savefig(plot_path, bbox_inches="tight")

# =============================================
# 9. Write DataFrame to storage
# =============================================

df_psm_matches_filtered.to_csv(
    "lapu-effect-ai-assistants-on-grade-outcomes/data/df-psm-matches-filtered.csv",
    index=False,
)
