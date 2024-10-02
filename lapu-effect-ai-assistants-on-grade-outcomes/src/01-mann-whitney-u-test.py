import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from dotenv import load_dotenv
from scipy import stats

import utils

# Define the project root
PROJECT_ROOT = "lapu-effect-ai-assistants-on-grade-outcomes"

utils.add_source_root_to_system_path(PROJECT_ROOT)

from src import misc, preprocessing  # noqa: E402

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
# Data Preparation
# =============================================

# Clean data
df = preprocessing.clean_data(df)

# Create binary did_use_spark field
df = preprocessing.create_did_use_spark_field(df)

# Select fields
df_gpas_with_usage = misc.select_gpas_and_binary_usage(df)

# Drop missing GPAs
df_gpas_with_usage = preprocessing.drop_missing_gpas(df_gpas_with_usage)

# Define control and treatment groups
group_treatment, group_control = preprocessing.create_treatment_and_control_groups(
    df_gpas_with_usage
)

# =============================================
# One-Sided Mann-Whitney U Test
# =============================================

# Perform the one-sided Mann-Whitney U test
u_statistic_one_sided, p_value_one_sided = stats.mannwhitneyu(
    group_treatment, group_control, alternative="greater"
)

# Display the results
print("One-Sided Mann-Whitney U Test Results (Testing if Treatment > Control):")
print(f"U-statistic: {u_statistic_one_sided}")
print(f"p-value: {p_value_one_sided:.7f}")

# Calculate Effect Size for One-Sided Test
n1 = len(group_treatment)
n2 = len(group_control)
effect_size_one_sided = u_statistic_one_sided / (n1 * n2)
print(f"Effect Size (Rank-Biserial Correlation): {effect_size_one_sided:.7f}")

# =============================================
# Two-Sided Mann-Whitney U Test
# =============================================

# Perform the two-sided Mann-Whitney U test
u_statistic_two_sided, p_value_two_sided = stats.mannwhitneyu(
    group_treatment, group_control, alternative="two-sided"
)

# Display the results
print("\nTwo-Sided Mann-Whitney U Test Results:")
print(f"U-statistic: {u_statistic_two_sided}")
print(f"p-value: {p_value_two_sided:.7f}")

# =============================================
# Descriptive Statistics
# =============================================

print("\nDescriptive Statistics:")

print("\nGroup Treatment (Used Spark):")
print(f"Median GPA: {np.median(group_treatment)}")
print(f"Mean GPA: {group_treatment.mean():.2f}")
print(f"Standard Deviation: {group_treatment.std():.2f}")
print(f"Sample Size: {n1}")

print("\nGroup Control (Did Not Use Spark):")
print(f"Median GPA: {np.median(group_control)}")
print(f"Mean GPA: {group_control.mean():.2f}")
print(f"Standard Deviation: {group_control.std():.2f}")
print(f"Sample Size: {n2}")

# =============================================
# Data Visualization
# =============================================

# Combine the data for plotting
df_plot = df_gpas_with_usage[["course_gpa", "did_use_spark"]].copy()
df_plot["Group"] = df_plot["did_use_spark"].map(
    {True: "Used Spark", False: "Did Not Use Spark"}
)

# Create the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Group", y="course_gpa", data=df_plot)
plt.title("Distribution of Course GPA by Group")
plt.xlabel("Group")
plt.ylabel("Course GPA")
plt.show()
