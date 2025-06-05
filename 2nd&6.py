import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind, shapiro, fisher_exact, kruskal, mannwhitneyu, chi2_contingency

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Data Loading & Initial Preprocessing ---

plays = pd.read_csv("Data/plays.csv")

# Filter for 2nd & 6 plays with valid EPA
filtered_plays = plays[(plays["down"] == 2) & (plays["yardsToGo"] == 6)].copy()
filtered_plays = filtered_plays[filtered_plays["expectedPointsAdded"].notnull()]

# Remove specific outlier/garbage time situations
filtered_plays = filtered_plays[filtered_plays["quarter"] != 5]
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"]
filtered_plays["score_difference"] = filtered_plays["preSnapHomeScore"] - filtered_plays["preSnapVisitorScore"]
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) & (filtered_plays["score_difference"].abs() > 17))
]
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) &
      (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) < 5)) &
      (filtered_plays["score_difference"].abs() >= 10) &
      ((filtered_plays["preSnapHomeTeamWinProbability"] >= 0.7) |
       (filtered_plays["preSnapVisitorTeamWinProbability"] >= 0.7)))
]
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 2) &
      (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) == 0 and int(x.split(":")[1]) < 5)))
]


# --- Play Type Classification ---

def classify_play(row):
    if row["playAction"]:
        return "Play Action"
    elif row["isDropback"]:
        return "Straight Dropback"
    return "Run"


filtered_plays["PlayType"] = filtered_plays.apply(classify_play, axis=1)

# --- Midfield vs. Non-Midfield Split ---

midfield = filtered_plays[filtered_plays["absoluteYardlineNumber"].between(40, 60)].copy()
non_midfield = filtered_plays[~filtered_plays["absoluteYardlineNumber"].between(40, 60)]

# --- Average EPA Bar Plots ---

# Midfield EPA bar plot
plt.figure(figsize=(8, 6))
sns.barplot(data=midfield.groupby("PlayType")["expectedPointsAdded"].mean().reset_index(name="Avg_EPA"),
            x="PlayType", y="Avg_EPA", palette="Set2")
plt.title("Average EPA on 2nd & 6 (40–60 Yardline) by Play Type")
plt.xlabel("Play Type")
plt.ylabel("Average EPA")
plt.tight_layout()
plt.show()

# Non-Midfield EPA bar plot
plt.figure(figsize=(8, 6))
sns.barplot(data=non_midfield.groupby("PlayType")["expectedPointsAdded"].mean().reset_index(name="Avg_EPA"),
            x="PlayType", y="Avg_EPA", palette="Set2")
plt.title("Average EPA on 2nd & 6 (Outside 40–60) by Play Type")
plt.xlabel("Play Type")
plt.ylabel("Average EPA")
plt.tight_layout()
plt.show()


# --- T-Tests for EPA Comparison ---

def run_t_tests(df, label):
    pa = df[df["PlayType"] == "Play Action"]["expectedPointsAdded"]
    dropback = df[df["PlayType"] == "Straight Dropback"]["expectedPointsAdded"]
    run = df[df["PlayType"] == "Run"]["expectedPointsAdded"]
    print(f"\n--- T-Test Results for {label} ---")
    print(f"Play Action vs Straight Dropback: p = {ttest_ind(pa, dropback, equal_var=False).pvalue:.4f}")
    print(f"Play Action vs Run:               p = {ttest_ind(pa, run, equal_var=False).pvalue:.4f}")
    print(f"Straight Dropback vs Run:         p = {ttest_ind(dropback, run, equal_var=False).pvalue:.4f}")


run_t_tests(filtered_plays, "All 2nd & 6 Plays")
run_t_tests(midfield, "Midfield (40–60)")
run_t_tests(non_midfield, "Outside Midfield")

# --- Play Type Distribution Plot ---

plays_2nd_6 = filtered_plays.copy()

# Plot play type distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=plays_2nd_6, x="PlayType", palette="Set2", order=["Run", "Straight Dropback", "Play Action"])
plt.title("Play Type Distribution on 2nd & 6")
plt.xlabel("Play Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Print play type counts
print("\n--- Play Type Counts on 2nd & 6 ---")
print(plays_2nd_6["PlayType"].value_counts())

# --- Shapiro-Wilk Normality Test on EPA ---

print("\n--- Shapiro-Wilk Normality Test on EPA by Play Type (2nd & 6) ---")
for play_type in plays_2nd_6["PlayType"].unique():
    epa_data = plays_2nd_6[plays_2nd_6["PlayType"] == play_type]["expectedPointsAdded"].dropna()
    if len(epa_data) > 3:
        stat, p = shapiro(epa_data)
        print(f"{play_type}: p = {p:.4f} | {'Reject' if p < 0.05 else 'Fail to Reject'} H₀ (normal distribution)")
    else:
        print(f"{play_type}: Not enough data for Shapiro-Wilk test.")

# --- EPA by Pass Depth Analysis ---

# Filter for pass plays with valid passResult and EPA
pass_plays = plays_2nd_6[plays_2nd_6["passResult"].notna() & plays_2nd_6["expectedPointsAdded"].notnull()].copy()


# Function to label pass depth
def label_depth(length):
    if pd.isna(length):
        return "Unknown"
    elif length <= 0:
        return "Behind LOS"
    elif length <= 5:
        return "Short"
    elif length <= 15:
        return "Intermediate"
    return "Deep"


# Apply pass depth labeling
pass_plays["PassDepth"] = pass_plays["passLength"].apply(label_depth)

# Calculate average EPA and count by pass depth
epa_by_depth = pass_plays.groupby("PassDepth")["expectedPointsAdded"].agg(["mean", "count"]).reset_index()
epa_by_depth.rename(columns={"mean": "Avg_EPA", "count": "Count"}, inplace=True)
print("\n--- Average EPA on 2nd & 6 by Pass Depth ---")
print(epa_by_depth)

# Plot average EPA by pass depth
plt.figure(figsize=(10, 6))
sns.barplot(data=epa_by_depth, x="PassDepth", y="Avg_EPA", palette="Set2",
            order=["Behind LOS", "Short", "Intermediate", "Deep", "Unknown"])
plt.title("Average EPA on 2nd & 6 by Pass Depth")
plt.ylabel("Average EPA")
plt.xlabel("Pass Depth Category")
plt.tight_layout()
plt.show()

# --- Kruskal-Wallis Test on Pass Depth (excluding certain categories) ---

# Filter for relevant pass depths for Kruskal-Wallis test
filtered_depths_for_kruskal = pass_plays[pass_plays["PassDepth"].isin(["Short", "Intermediate", "Deep"])].copy()
depth_groups = [
    filtered_depths_for_kruskal[filtered_depths_for_kruskal["PassDepth"] == d]["expectedPointsAdded"].dropna() for d in
    ["Short", "Intermediate", "Deep"]]

# Perform Kruskal-Wallis test if groups are not empty
if all(len(group) > 0 for group in depth_groups):
    stat, p_val = kruskal(*depth_groups)
    print(f"\n--- Kruskal-Wallis Test (EPA by Pass Depth, Excluding Behind LOS & Unknown) ---")
    print(f"Test Statistic = {stat:.4f}, p-value = {p_val:.4f}")
else:
    print("\n--- Kruskal-Wallis Test (EPA by Pass Depth, Excluding Behind LOS & Unknown) ---")
    print("Not enough data in all specified pass depth categories for Kruskal-Wallis test.")

# --- Turnover Rate by Dropback Type ---

# Filter for dropback plays
dropbacks = filtered_plays[
    (filtered_plays["down"] == 2) & (filtered_plays["yardsToGo"] == 6) & (filtered_plays["isDropback"])].copy()
dropbacks["DropbackType"] = dropbacks["playAction"].apply(lambda x: "Play Action" if x else "Straight Dropback")

# Calculate turnover rate (interceptions)
turnovers = dropbacks[dropbacks["passResult"] == "IN"]
total_dropbacks_by_type = dropbacks["DropbackType"].value_counts()
interceptions_by_type = turnovers["DropbackType"].value_counts()

# Create DataFrame for turnover rates
turnover_rate_df = pd.DataFrame({
    'DropbackType': total_dropbacks_by_type.index,
    'Turnover Rate': (interceptions_by_type / total_dropbacks_by_type).fillna(0)
}).reset_index(drop=True)

print("\n--- Turnover Rate (Interceptions) by Dropback Type ---")
print(turnover_rate_df)

# --- First Down Rate on Play Action Plays ---

# Filter for play action plays
pa_plays = dropbacks[dropbacks["DropbackType"] == "Play Action"].copy()
# Determine if play action resulted in a first down
pa_plays["FirstDown"] = pa_plays["yardsGained"] >= 6
# Calculate first down rate
first_down_rate = pa_plays["FirstDown"].mean()

print(f"\n--- First Down Rate for Play Action on 2nd & 6 ---")
print(f"First Down Rate: {first_down_rate:.3%} ({pa_plays['FirstDown'].sum()}/{len(pa_plays)})")

# --- 3rd Down Conversion Rate After Incompletions ---

# Filter for incomplete passes on 2nd & 6
incompletes = dropbacks[dropbacks["passResult"] == "I"].copy()
incompletes["DropbackType"] = incompletes["playAction"].apply(lambda x: "Play Action" if x else "Straight Dropback")

# Sort filtered_plays to get next play information
filtered_plays_sorted_for_next_play = filtered_plays.sort_values(by=["gameId", "playId"]).copy()
filtered_plays_sorted_for_next_play["nextPlayYardsGained"] = filtered_plays_sorted_for_next_play.groupby("gameId")[
    "yardsGained"].shift(-1)
filtered_plays_sorted_for_next_play["nextPlayDown"] = filtered_plays_sorted_for_next_play.groupby("gameId")[
    "down"].shift(-1)
filtered_plays_sorted_for_next_play["nextPlayYardsToGo"] = filtered_plays_sorted_for_next_play.groupby("gameId")[
    "yardsToGo"].shift(-1)

# Merge next play data into incompletes DataFrame
merged_incompletes = pd.merge(
    incompletes,
    filtered_plays_sorted_for_next_play[
        ["gameId", "playId", "nextPlayYardsGained", "nextPlayDown", "nextPlayYardsToGo"]],
    on=["gameId", "playId"],
    how="left"
)

# Define 3rd down conversion: next play is 3rd down and yards gained >= yards to go
merged_incompletes["converted_3rd_down"] = (
        (merged_incompletes["nextPlayDown"] == 3) &
        (merged_incompletes["nextPlayYardsGained"] >= merged_incompletes["nextPlayYardsToGo"])
)

# Summarize conversion rates
conversion_summary_after_incomplete = merged_incompletes.groupby("DropbackType")["converted_3rd_down"].agg(
    ["mean", "count", "sum"]).reset_index()
conversion_summary_after_incomplete.columns = ["DropbackType", "Conversion Rate", "Total Incompletions", "Converted"]
conversion_summary_after_incomplete["Conversion Rate"] = conversion_summary_after_incomplete["Conversion Rate"].apply(
    lambda x: f"{x:.2%}")

print("\n--- 3rd Down Conversion After 2nd & 6 Incomplete Pass ---")
print(conversion_summary_after_incomplete)

# Fisher's Exact Test on 3rd down conversion success after incomplete
if 'Play Action' in conversion_summary_after_incomplete['DropbackType'].values and \
        'Straight Dropback' in conversion_summary_after_incomplete['DropbackType'].values:

    pa_converted = \
    conversion_summary_after_incomplete[conversion_summary_after_incomplete['DropbackType'] == 'Play Action'][
        'Converted'].iloc[0]
    pa_not_converted = \
    conversion_summary_after_incomplete[conversion_summary_after_incomplete['DropbackType'] == 'Play Action'][
        'Total Incompletions'].iloc[0] - pa_converted

    sb_converted = \
    conversion_summary_after_incomplete[conversion_summary_after_incomplete['DropbackType'] == 'Straight Dropback'][
        'Converted'].iloc[0]
    sb_not_converted = \
    conversion_summary_after_incomplete[conversion_summary_after_incomplete['DropbackType'] == 'Straight Dropback'][
        'Total Incompletions'].iloc[0] - sb_converted

    fisher_table = [[int(pa_converted), int(pa_not_converted)], [int(sb_converted), int(sb_not_converted)]]

    odds_ratio, p_val = fisher_exact(fisher_table, alternative='two-sided')
    print("\n--- Fisher's Exact Test — 3rd Down Conversion After Incomplete 2nd & 6 ---")
    print(f"Odds Ratio = {odds_ratio:.4f}")
    print(f"p-value = {p_val:.4f}")
else:
    print("\n--- Fisher's Exact Test ---")
    print("Not enough data for both Play Action and Straight Dropback for Fisher's Exact Test.")

# --- Conversion & Touchdown Rate by Play Type (Overall) ---

# Re-filter 2nd & 6 plays from original 'plays' for comprehensive next-play analysis
sequence_df = plays[(plays["down"] == 2) & (plays["yardsToGo"] == 6)].copy()
sequence_df["PlayType"] = sequence_df.apply(classify_play, axis=1)
sequence_df = sequence_df.sort_values(by=["gameId", "playId"]).copy()
sequence_df["playIndex"] = sequence_df.groupby("gameId").cumcount()

# Get next play metadata from the full sorted plays DataFrame
full_sorted = plays.sort_values(by=["gameId", "playId"]).copy()
full_sorted["playIndex"] = full_sorted.groupby("gameId").cumcount()
full_sorted["nextPlayYardsGained"] = full_sorted.groupby("gameId")["yardsGained"].shift(-1)
full_sorted["nextPlayDescription"] = full_sorted.groupby("gameId")["playDescription"].shift(-1)
full_sorted["nextEPA"] = full_sorted.groupby("gameId")["expectedPointsAdded"].shift(-1)
full_sorted["nextPlayDown"] = full_sorted.groupby("gameId")["down"].shift(-1)
full_sorted["nextPlayYardsToGo"] = full_sorted.groupby("gameId")["yardsToGo"].shift(-1)

# Merge next play data into sequence_df
sequence_df = sequence_df.merge(
    full_sorted[["gameId", "playIndex", "nextPlayYardsGained", "nextPlayDescription", "nextEPA", "nextPlayDown",
                 "nextPlayYardsToGo"]],
    on=["gameId", "playIndex"],
    how="left"
)

# Define conversion (yards gained on current play vs. yards to go)
sequence_df["converted"] = sequence_df["yardsGained"] >= sequence_df["yardsToGo"]

# Determine if the current play resulted in a touchdown or if the immediate next play description contains "touchdown"
sequence_df["touchdown"] = sequence_df["playDescription"].str.lower().str.contains("touchdown") | \
                           sequence_df["nextPlayDescription"].str.lower().str.contains("touchdown")

# Summarize conversion and touchdown rates
conversion_summary_overall = sequence_df.groupby("PlayType").agg(
    Play_Count=("playId", "count"),
    Avg_Yards_Gained=("yardsGained", "mean"),
    Conversion_Rate=("converted", "mean"),
    TD_Rate=("touchdown", "mean")
).reset_index()

conversion_summary_overall["Conversion_Rate"] = conversion_summary_overall["Conversion_Rate"].apply(
    lambda x: f"{x:.2%}")
conversion_summary_overall["TD_Rate"] = conversion_summary_overall["TD_Rate"].apply(lambda x: f"{x:.2%}")
conversion_summary_overall["Avg_Yards_Gained"] = conversion_summary_overall["Avg_Yards_Gained"].round(2)

print("\n--- Play Outcomes (Avg Yards, Conversion %, TD %) on 2nd & 6 ---")
print(conversion_summary_overall)

# --- EPA Delta from Play Type (Next Play EPA - Current EPA) ---

# Calculate EPA delta
sequence_df["epa_delta"] = sequence_df["nextEPA"] - sequence_df["expectedPointsAdded"]
# Summarize average EPA delta by play type
epa_delta_summary = sequence_df.groupby("PlayType")["epa_delta"].mean().reset_index()
epa_delta_summary["epa_delta"] = epa_delta_summary["epa_delta"].round(3)

print("\n--- Average EPA Delta (Next Play EPA – 2nd & 6 EPA) by Play Type ---")
print(epa_delta_summary)

# --- Midfield-Only Play Outcome Analysis ---

# Filter for midfield plays in the sequence_df
midfield_sequence = sequence_df[sequence_df["absoluteYardlineNumber"].between(40, 60)].copy()

# Summarize conversion and touchdown rates for midfield plays
midfield_summary_outcomes = midfield_sequence.groupby("PlayType").agg(
    Play_Count=("playId", "count"),
    Avg_Yards_Gained=("yardsGained", "mean"),
    Conversion_Rate=("converted", "mean"),
    TD_Rate=("touchdown", "mean")
).reset_index()

midfield_summary_outcomes["Conversion_Rate"] = midfield_summary_outcomes["Conversion_Rate"].apply(lambda x: f"{x:.2%}")
midfield_summary_outcomes["TD_Rate"] = midfield_summary_outcomes["TD_Rate"].apply(lambda x: f"{x:.2%}")
midfield_summary_outcomes["Avg_Yards_Gained"] = midfield_summary_outcomes["Avg_Yards_Gained"].round(2)

print("\n--- Play Outcomes (Avg Yards, Conversion %, TD %) on 2nd & 6 (Midfield Only) ---")
print(midfield_summary_outcomes)

# --- Kruskal & Mann-Whitney Tests (Yards Gained Midfield) ---

# Prepare groups for non-parametric tests on yards gained at midfield
groups = [
    midfield_sequence[midfield_sequence["PlayType"] == "Play Action"]["yardsGained"].dropna(),
    midfield_sequence[midfield_sequence["PlayType"] == "Run"]["yardsGained"].dropna(),
    midfield_sequence[midfield_sequence["PlayType"] == "Straight Dropback"]["yardsGained"].dropna()
]

# Kruskal-Wallis Test
if all(len(group) > 0 for group in groups):  # Ensure groups are not empty
    stat, p_val = kruskal(*groups)
    print("\n--- Kruskal-Wallis Test on Yards Gained (2nd & 6 Midfield) ---")
    print(f"Test Statistic = {stat:.4f}")
    print(f"p-value = {p_val:.4f}")
else:
    print("\n--- Kruskal-Wallis Test on Yards Gained (2nd & 6 Midfield) ---")
    print("Not enough data in all play type categories for Kruskal-Wallis test.")

# Pairwise Mann-Whitney U tests
pa_yards = groups[0]
run_yards = groups[1]
dropback_yards = groups[2]

print("\n--- Pairwise Mann-Whitney U Tests (Yards Gained on 2nd & 6 Midfield) ---")
# Only run tests if both groups have data
if len(pa_yards) > 0 and len(dropback_yards) > 0:
    print(f"Play Action vs Straight Dropback: p = {mannwhitneyu(pa_yards, dropback_yards).pvalue:.4f}")
else:
    print("Not enough data for Play Action vs Straight Dropback Mann-Whitney U test.")
if len(pa_yards) > 0 and len(run_yards) > 0:
    print(f"Play Action vs Run:               p = {mannwhitneyu(pa_yards, run_yards).pvalue:.4f}")
else:
    print("Not enough data for Play Action vs Run Mann-Whitney U test.")
if len(dropback_yards) > 0 and len(run_yards) > 0:
    print(f"Straight Dropback vs Run:         p = {mannwhitneyu(dropback_yards, run_yards).pvalue:.4f}")
else:
    print("Not enough data for Straight Dropback vs Run Mann-Whitney U test.")

print("\n--- Median Yards Gained (Midfield 2nd & 6) ---")
if len(pa_yards) > 0:
    print(f"Play Action:       {pa_yards.median():.2f}")
else:
    print("Play Action: No data for median.")
if len(run_yards) > 0:
    print(f"Run:               {run_yards.median():.2f}")
else:
    print("Run: No data for median.")
if len(dropback_yards) > 0:
    print(f"Straight Dropback: {dropback_yards.median():.2f}")
else:
    print("Straight Dropback: No data for median.")

# --- Chi-Square Test: Success Rates (Midfield) ---

# Define success based on yards gained >= 6
midfield_sequence["Success"] = midfield_sequence["yardsGained"] >= 6
# Create contingency table for chi-square test
success_table = pd.crosstab(midfield_sequence["PlayType"], midfield_sequence["Success"])

# Perform Chi-Square Test if table is not empty
if not success_table.empty and success_table.shape[0] > 1 and success_table.shape[1] > 1:
    chi2, p, dof, _ = chi2_contingency(success_table)
    print("\n--- Chi-Square Test — Success Rate (2nd & 6 Midfield) ---")
    print("Contingency Table:")
    print(success_table)
    print(f"Chi-Square p-value = {p:.4f}")
else:
    print("\n--- Chi-Square Test — Success Rate (2nd & 6 Midfield) ---")
    print("Not enough data for Chi-Square Test on success rates (Midfield).")

# --- 2nd Down Yards to Go Distribution (Overall) ---

# Filter for all 2nd down plays and valid yardsToGo
second_down_all = plays[plays["down"] == 2].copy()
second_down_all = second_down_all[second_down_all["yardsToGo"].notnull()]

# Count play occurrences by yards to go
yards_to_go_counts = second_down_all["yardsToGo"].value_counts().sort_index().reset_index()
yards_to_go_counts.columns = ["YardsToGo", "PlayCount"]

print("\n--- 2nd Down Yards to Go Distribution ---")
print(yards_to_go_counts)

# Plot 2nd down yards to go distribution
plt.figure(figsize=(12, 6))
sns.barplot(data=yards_to_go_counts, x="YardsToGo", y="PlayCount", palette="Set2")
plt.title("Play Counts by Yards to Go on 2nd Down")
plt.xlabel("Yards to Go")
plt.ylabel("Number of Plays")
plt.tight_layout()
plt.show()
