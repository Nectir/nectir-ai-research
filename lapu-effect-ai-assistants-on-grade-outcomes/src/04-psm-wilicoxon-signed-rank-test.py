import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

# =============================================
# 1. Reading data
# =============================================

df_psm_matches_filtered = pd.read_csv(
    "lapu-effect-ai-assistants-on-grade-outcomes/data/df-psm-matches-filtered.csv"
)

# =============================================
# 2. Calculating GPA Differences
# =============================================

# Calculate the differences in GPA
df_psm_matches_filtered["gpa_diff"] = (
    df_psm_matches_filtered["treated_course_gpa"]
    - df_psm_matches_filtered["control_course_gpa"]
)

# =============================================
# 3. Wilcoxon Signed-Rank Test (Adjusting for Ties)
# =============================================

# Perform the Wilcoxon Signed-Rank Test
statistic, p_value = wilcoxon(
    df_psm_matches_filtered["gpa_diff"],
    zero_method="wilcox",  # Drops zero differences
    correction=False,  # No continuity correction
    alternative="greater",  # One-sided test
    mode="approx",  # Approximate p-value computation (appropriate when n > 25)
)

print("Wilcoxon Signed-Rank Test Results (ties are automatically handled):")
print(f"Statistic (W): {statistic}")
print(f"p-value: {p_value:.7f}")

# =============================================
# 4. Calculate Effect Size (r)
# =============================================

# Filter out zero differences
non_zero_diffs = df_psm_matches_filtered["gpa_diff"][
    df_psm_matches_filtered["gpa_diff"] != 0
]

# Number of non-zero differences (sample size)
n = len(non_zero_diffs)

# Calculate mean and standard deviation of ranks (adjusted for ties)
mean_rank = n * (n + 1) / 4
std_rank = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

# Compute Z-score using Wilcoxon statistic, mean, and standard deviation of ranks
z = (statistic - mean_rank) / std_rank

# Calculate effect size (r) using Z-score and sample size
effect_size = z / np.sqrt(n)

print(f"Effect Size (r): {effect_size:.7f}")

# =============================================
# 5. Descriptive Statistics
# =============================================

print("\nDescriptive Statistics of Matched Pairs:")
print(f"Number of Matched Pairs: {len(df_psm_matches_filtered)}")

print("\nTreated Group (Used Spark):")
print(f"Mean GPA: {df_psm_matches_filtered['treated_course_gpa'].mean():.2f}")
print(f"Median GPA: {df_psm_matches_filtered['treated_course_gpa'].median()}")
print(f"Standard Deviation: {df_psm_matches_filtered['treated_course_gpa'].std():.2f}")

print("\nControl Group (Did Not Use Spark):")
print(f"Mean GPA: {df_psm_matches_filtered['control_course_gpa'].mean():.2f}")
print(f"Median GPA: {df_psm_matches_filtered['control_course_gpa'].median()}")
print(f"Standard Deviation: {df_psm_matches_filtered['control_course_gpa'].std():.2f}")

# =============================================
# 6. Data Visualization
# =============================================

# Histogram of GPA Differences
plt.figure(figsize=(10, 6))
sns.histplot(df_psm_matches_filtered["gpa_diff"], bins=20, kde=False)
plt.title("Distribution of GPA Differences (Treated - Control)")
plt.xlabel("GPA Difference")
plt.ylabel("Frequency")
plt.grid(axis="y")
plt.show()

# Box Plot of Treated vs Control GPAs
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_psm_matches_filtered[["treated_course_gpa", "control_course_gpa"]])
plt.title("Course GPA Distribution by Group")
plt.xticks([0, 1], ["Treated (Used Spark)", "Control (Did Not Use Spark)"])
plt.ylabel("Course GPA")
plt.grid(axis="y")
plt.show()
