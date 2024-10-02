import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from dotenv import load_dotenv
from scipy.stats import permutation_test

import utils

PROJECT_ROOT = "lapu-effect-ai-assistants-on-grade-outcomes"

utils.add_source_root_to_system_path(PROJECT_ROOT)

from src import misc, preprocessing  # noqa: E402

dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path)

config_path = utils.get_configs_path("lapu-effect-ai-assistants-on-grade-outcomes")

with open(config_path, "r") as config_file:
    configs = yaml.safe_load(config_file)

file_id = configs.get("file_id")

df = misc.read_excel_from_drive(file_id)

# =============================================
# Data Preparation
# =============================================

df = preprocessing.clean_data(df)

# Create binary did_use_spark field
df = preprocessing.create_did_use_spark_field(df)

df_gpas_with_usage = misc.select_gpas_and_binary_usage(df)

df_gpas_with_usage = preprocessing.drop_missing_gpas(df_gpas_with_usage)

group_treatment, group_control = preprocessing.create_treatment_and_control_groups(
    df_gpas_with_usage
)

# =============================================
# Permutation Test
# =============================================


# Define the test statistic function
def difference_in_means(x, y):
    return np.mean(x) - np.mean(y)


# Perform the permutation test
result = permutation_test(
    data=(group_treatment, group_control),
    statistic=difference_in_means,
    permutation_type="independent",
    alternative="greater",  # One-sided test
    n_resamples=10000,
    random_state=42,  # For reproducibility
)

print("Permutation Test Results:")
print(f"Observed Difference in Means: {result.statistic:.7f}")
print(f"p-value: {result.pvalue:.7f}")

# =============================================
# Descriptive Statistics
# =============================================

print("\nDescriptive Statistics:")

n1 = len(group_treatment)
n2 = len(group_control)

print("\nGroup Treatment (Used Spark):")
print(f"Median GPA: {np.median(group_treatment)}")
print(f"Mean GPA: {group_treatment.mean():.2f}")
print(f"Standard Deviation: {np.std(group_treatment, ddof=1):.2f}")
print(f"Sample Size: {n1}")

print("\nGroup Control (Did Not Use Spark):")
print(f"Median GPA: {np.median(group_control)}")
print(f"Mean GPA: {group_control.mean():.2f}")
print(f"Standard Deviation: {np.std(group_control, ddof=1):.2f}")
print(f"Sample Size: {n2}")

# =============================================
# Data Visualization
# =============================================

# Combine data for plotting
df_plot = df_gpas_with_usage[["course_gpa", "did_use_spark"]].copy()
df_plot["Group"] = df_plot["did_use_spark"].map(
    {True: "Used Spark", False: "Did Not Use Spark"}
)

# Create box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Group", y="course_gpa", data=df_plot)
plt.title("Distribution of Course GPA by Group")
plt.xlabel("Group")
plt.ylabel("Course GPA")
plt.show()
