import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import permutation_test

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
# 3. Permutation Test for Paired Data
# =============================================


# Define the test statistic function
def mean_difference(x):
    return np.mean(x)


# Perform the permutation test
result = permutation_test(
    data=(df_psm_matches_filtered["gpa_diff"].values,),
    statistic=mean_difference,
    permutation_type="samples",  # Indicates observations are from paired but distinct samples
    alternative="greater",  # One-sided test
    n_resamples=10000,
    random_state=42,
)

print("\nPermutation Test Results:")
print(f"Observed Mean Difference: {result.statistic:.7f}")
print(f"p-value: {result.pvalue:.7f}")

# =============================================
# 4. Visualization of Permutation Test Results
# =============================================

# Extract the distribution of permuted statistics
permuted_stats = (
    result.null_distribution
)  # Distribution of mean differences from permutations
observed_stat = result.statistic  # Observed mean difference

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the distribution of permuted statistics
sns.histplot(permuted_stats, bins=30, kde=False, color="lightblue")
plt.axvline(
    observed_stat,
    color="red",
    linestyle="--",
    label=f"Observed mean GPA difference = {observed_stat:.3f}",
)
plt.title(
    "Permutation Test Results: GPA Differences for Matched Student-Course Pairs (Spark Usage vs. No Usage)"
)
plt.xlabel("GPA Difference")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(
    "lapu-effect-ai-assistants-on-grade-outcomes/figures/04-distribution-permuted-gpa-differences.png"
)
plt.savefig(plot_path)

# Show the plot
plt.show()
