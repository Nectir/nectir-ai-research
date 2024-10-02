import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv

import utils

# Define the project root
PROJECT_ROOT = "lapu-effect-ai-assistants-on-grade-outcomes"

utils.add_source_root_to_system_path(PROJECT_ROOT)

from src import misc

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
df = misc.clean_data(df)

# Create binary did_use_spark field
df = misc.create_did_use_spark_field(df)

# Select fields
df_gpas_with_usage = misc.select_gpas_and_binary_usage(df)

# Drop missing GPAs
df_gpas_with_usage = misc.drop_missing_gpas(df_gpas_with_usage)

# Define control and treatment groups
group_treatment, group_control = misc.create_treatment_and_control_groups(
    df_gpas_with_usage
)

# =============================================
# Visualize GPA Distributions
# =============================================

# Assuming group_treatment and group_control are dataframes that have been created
df_gpas_with_usage["group"] = df_gpas_with_usage["did_use_spark"].apply(
    lambda x: "Treatment" if x else "Control"
)

# Ensure the control group is on the left side by reordering the 'group' column
df_gpas_with_usage["group"] = pd.Categorical(
    df_gpas_with_usage["group"], categories=["Control", "Treatment"], ordered=True
)

# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(
    x="group", y="course_gpa", data=df_gpas_with_usage, inner="quartile", bw_method=0.2
)

# Set y-axis limits
plt.ylim(0, 4)

# Add titles and labels
plt.title("GPA Distributions by Spark Usage (Control vs. Treatment)", fontsize=16)
plt.xlabel("Group", fontsize=12)
plt.ylabel("GPA", fontsize=12)

# Save the plot
plot_path = os.path.join(
    "lapu-effect-ai-assistants-on-grade-outcomes/figures/00-gpa-distributions-control-treatment.png"
)
plt.savefig(plot_path)

# Show the plot
plt.show()

# =============================================
# Descriptive Statistics
# =============================================

# Calculate total number of student-course combinations
total_combinations = df_gpas_with_usage.shape[0]

# Calculate total number of unique students in the dataset
total_unique_students = df_gpas_with_usage["student_id"].nunique()

# Calculate sample sizes for control and treatment groups (student-course combinations)
control_combinations = group_control.shape[0]
treatment_combinations = group_treatment.shape[0]

# Calculate the number of unique students in control and treatment groups
control_unique_students = df_gpas_with_usage[
    df_gpas_with_usage["did_use_spark"] == False
]["student_id"].nunique()
treatment_unique_students = df_gpas_with_usage[
    df_gpas_with_usage["did_use_spark"] == True
]["student_id"].nunique()

# Calculate mean and standard deviation of GPA for control and treatment groups (student-course combinations)
control_mean_gpa = group_control.mean()
control_std_gpa = group_control.std()

treatment_mean_gpa = group_treatment.mean()
treatment_std_gpa = group_treatment.std()

# Calculate overall mean and standard deviation of GPA across all student-course combinations
overall_mean_gpa = df_gpas_with_usage["course_gpa"].mean()
overall_std_gpa = df_gpas_with_usage["course_gpa"].std()

# Create a summary DataFrame to report the descriptive statistics for student-course combinations and unique students
summary_df = pd.DataFrame(
    {
        "Group": [
            "Control (student-course)",
            "Treatment (student-course)",
            "Overall (student-course)",
        ],
        "Sample Size (Student-Course Combinations)": [
            control_combinations,
            treatment_combinations,
            total_combinations,
        ],
        "Unique Students": [
            control_unique_students,
            treatment_unique_students,
            total_unique_students,
        ],
        "Mean GPA": [control_mean_gpa, treatment_mean_gpa, overall_mean_gpa],
        "Standard Deviation GPA": [control_std_gpa, treatment_std_gpa, overall_std_gpa],
    }
)

# Print descriptive statistics for control and treatment groups (student-course combinations and unique students)
print("Descriptive Statistics (Student-Course Combinations and Unique Students)")
print("=======================================================================")
print(f"Total student-course combinations: {total_combinations}")
print(f"Total unique students: {total_unique_students}")
print(
    f"Control group: {control_combinations} combinations, {control_unique_students} unique students, Mean GPA = {control_mean_gpa:.2f}, SD = {control_std_gpa:.2f}"
)
print(
    f"Treatment group: {treatment_combinations} combinations, {treatment_unique_students} unique students, Mean GPA = {treatment_mean_gpa:.2f}, SD = {treatment_std_gpa:.2f}"
)
print(f"Overall Mean GPA = {overall_mean_gpa:.2f}, SD = {overall_std_gpa:.2f}")

# Calculate the total number of unique courses in the dataset
total_courses = df_gpas_with_usage["course_code"].nunique()

# Print the results
print(f"Total number of courses: {total_courses}")
