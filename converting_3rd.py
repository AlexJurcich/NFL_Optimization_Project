import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Data Loading and Preprocessing ---

# Load data
plays = pd.read_csv('Data/plays.csv')

# Filter for third-down pass plays
third_down_plays = plays.loc[
    (plays["down"] == 3) &
    (plays["passResult"].notna())
    ].copy()


# Categorize passes as 'First Down' (beyond line to gain) or 'Short'
def categorize_passes(row):
    if row["passLength"] >= row["yardsToGo"]:
        return "First Down"
    else:
        return "Short"


# Apply categorization function
third_down_plays['pass_category'] = third_down_plays.apply(categorize_passes, axis=1)

# Determine if the play resulted in a conversion
third_down_plays['converted'] = third_down_plays['yardsGained'] >= third_down_plays['yardsToGo']

# --- Conversion Rate Calculation ---

# Calculate conversion rates by pass category
conversion_rates = third_down_plays.groupby('pass_category')['converted'].mean().reset_index()
conversion_rates.columns = ['Pass Location', 'Conversion Rate']

# Print conversion rates
print("--- Conversion Rates ---")
print(conversion_rates)

# --- Visualization ---

# Prepare data for plotting (using calculated rates for consistency)
plot_data = pd.DataFrame({
    "Pass Location": conversion_rates['Pass Location'],
    "Conversion Rate": conversion_rates['Conversion Rate']
})

custom_palette = {
    "First Down": "#55A868",  # Turf Green
    "Short": "#88CCEE"       # Complementary light blue
}

# Create bar plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x="Pass Location", y="Conversion Rate", data=plot_data, palette=custom_palette)

# Add labels and title
plt.xlabel("Pass Location")
plt.ylabel("Conversion Rate")
plt.title("Third Down Conversion Rate: Pass Beyond vs. Short of Line to Gain")

# Display values on bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

# Adjust y-axis limit
plt.ylim(0, 0.6)
plt.tight_layout()
plt.show()

# --- Hypothesis Testing (Two-Proportion Z-test) ---

# Get counts and observations for the z-test
n1 = third_down_plays[third_down_plays['pass_category'] == 'First Down'].shape[0]
n2 = third_down_plays[third_down_plays['pass_category'] == 'Short'].shape[0]
x1 = \
    third_down_plays[
        (third_down_plays["pass_category"] == "First Down") & (third_down_plays["converted"] == True)].shape[0]
x2 = third_down_plays[(third_down_plays["pass_category"] == "Short") & (third_down_plays["converted"] == True)].shape[0]

# Perform z-test
count = np.array([x1, x2])
nobs = np.array([n1, n2])
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

# Print z-test results
print("\n--- Hypothesis Testing (Two-Proportion Z-test) ---")
print(f"Z-statistic = {z_stat:.4f}")
print(f"P-value = {p_value:.10e}")

# --- Pass Distribution Analysis ---

# Count total passes by category
pass_counts = third_down_plays["pass_category"].value_counts().reset_index()
pass_counts.columns = ["Pass Location", "Total Passes"]

# Print pass distribution
print("\n--- Pass Distribution ---")
print(pass_counts)

# --- Field Position Analysis ---

# Create field position bins (10-yard increments)
third_down_plays["field_position_bin"] = (third_down_plays["absoluteYardlineNumber"] // 10) * 10

# Count passes by field position and category
pass_counts_by_field = third_down_plays.groupby(["field_position_bin", "pass_category"]).size().unstack(fill_value=0)
pass_counts_by_field["Total Passes"] = pass_counts_by_field.sum(axis=1)
pass_counts_by_field["First Down %"] = pass_counts_by_field["First Down"] / pass_counts_by_field["Total Passes"]
pass_counts_by_field["Short %"] = pass_counts_by_field["Short"] / pass_counts_by_field["Total Passes"]
pass_counts_by_field = pass_counts_by_field.reset_index()

# Print field position distribution
print("\n--- Pass Distribution by Field Position (10-yard bins) ---")
print(pass_counts_by_field)

# --- Chi-Square Test (Field Position) ---

# Create contingency table for chi-square test
contingency_table = third_down_plays.pivot_table(
    index="field_position_bin", columns="pass_category", aggfunc="size", fill_value=0
)
contingency_table = contingency_table[contingency_table.sum(axis=1) > 0]  # Filter out empty rows

# Perform Chi-Square Test if enough data
if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test (Field Position and Pass Category) ---")
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("\nExpected Frequencies:")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
else:
    print("\n--- Chi-Square Test (Field Position and Pass Category) ---")
    print("Not enough variation in field position or pass categories for Chi-Square test.")

# --- Yards to Go Distribution ---

# Count passes by yards to go and category
yardage_to_go_counts = third_down_plays.groupby(["yardsToGo", "pass_category"]).size().unstack(fill_value=0)
yardage_to_go_counts["Total Passes"] = yardage_to_go_counts.sum(axis=1)
yardage_to_go_counts["First Down %"] = yardage_to_go_counts["First Down"] / yardage_to_go_counts["Total Passes"]
yardage_to_go_counts["Short %"] = yardage_to_go_counts["Short"] / yardage_to_go_counts["Total Passes"]
yardage_to_go_counts = yardage_to_go_counts.reset_index()

# Print yards to go distribution
print("\n--- Pass Distribution by Yards to Go ---")
print(yardage_to_go_counts)
