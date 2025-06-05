import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Data Loading & Initial Preprocessing ---
plays = pd.read_csv("Data/plays.csv")

# Filter out outlier and garbage time plays
filtered_plays = plays.copy()
filtered_plays = filtered_plays[filtered_plays["quarter"] != 5]
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"]

# Calculate pre-snap score difference
filtered_plays["score_difference"] = (
        filtered_plays["preSnapHomeScore"] - filtered_plays["preSnapVisitorScore"]
)

# Remove garbage time in Q4 based on large score difference (> 17 points)
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) & (filtered_plays["score_difference"].abs() > 17))
]

# Remove more specific garbage time scenarios in Q4
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) &
      (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) < 5)) &
      (filtered_plays["score_difference"].abs() >= 10) &
      ((filtered_plays["preSnapHomeTeamWinProbability"] >= 0.7) |
       (filtered_plays["preSnapVisitorTeamWinProbability"] >= 0.7)))
]

# Remove end-of-half plays in Q2 (under 5 seconds on clock)
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 2) &
      (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) == 0 and int(x.split(":")[1]) < 5)))
]

# Filter for run plays only (where isDropback is False)
run_plays = filtered_plays[filtered_plays["isDropback"] == False].copy()


# --- Classify Formation Type (Shotgun vs. Under Center) ---
def classify_formation(formation):
    if pd.isna(formation):
        return "Unknown"
    elif "SHOTGUN" in formation.upper():
        return "Shotgun"
    else:
        return "Under Center"


run_plays["formation_type"] = run_plays["offenseFormation"].apply(classify_formation)
run_plays = run_plays[run_plays["formation_type"] != "Unknown"]  # Remove plays with unknown formations


# --- Classify Run Scheme (Gap vs. Zone) ---
def label_scheme(concept):
    if concept in ["COUNTER", "POWER", "PULL LEAD", "TRAP"]:
        return "Gap"
    elif concept in ["INSIDE ZONE", "OUTSIDE ZONE"]:
        return "Zone"
    else:
        return "Other"


# Apply scheme labeling to plays with valid run concepts
scheme_data = run_plays.copy()
scheme_data = scheme_data[scheme_data["pff_runConceptPrimary"].notna()]
scheme_data["run_scheme"] = scheme_data["pff_runConceptPrimary"].apply(label_scheme)
scheme_data = scheme_data[scheme_data["run_scheme"].isin(["Gap", "Zone"])]  # Keep only Gap and Zone schemes

# --- Summarize Yards Gained by Scheme and Formation ---

scheme_summary = (
    scheme_data.groupby(["run_scheme", "formation_type"])["yardsGained"]
    .agg(["count", "mean", "std"])
    .rename(columns={"count": "Play Count", "mean": "Avg Yards Gained", "std": "STD"})
    .reset_index()
)

# Pivot table to compare average yards gained side-by-side
pivot = scheme_summary.pivot(index="run_scheme", columns="formation_type", values="Avg Yards Gained").reset_index()
pivot["Avg Diff (Shotgun - UC)"] = pivot["Shotgun"] - pivot["Under Center"]

# Display results
print("===== Gap vs Zone Performance by Formation (All Downs) =====")
print(pivot)

# --- Overall Gap vs Zone Run Performance (All Formations) ---

overall_scheme_summary = (
    scheme_data.groupby("run_scheme")["yardsGained"]
    .agg(["count", "mean", "std"])
    .rename(columns={"count": "Play Count", "mean": "Avg Yards Gained", "std": "STD"})
    .reset_index()
)

print("===== Overall Gap vs Zone Run Performance (All Formations) =====")
print(overall_scheme_summary)

# --- Mann-Whitney U Test: Gap vs Zone Yards Gained ---

# Extract yards gained for Gap and Zone schemes
gap_yards = scheme_data[scheme_data["run_scheme"] == "Gap"]["yardsGained"]
zone_yards = scheme_data[scheme_data["run_scheme"] == "Zone"]["yardsGained"]

# Run Mann-Whitney U test (non-parametric, two-sided)
u_stat, p_val = mannwhitneyu(gap_yards, zone_yards, alternative='two-sided')
print(f"\nMann-Whitney U Test:\nU = {u_stat:.0f}, p-value = {p_val:.4f}")

# --- Explosive Run Rate: Gap vs Zone ---

# Add explosive run flag (yards gained >= 10)
scheme_data["explosive"] = scheme_data["yardsGained"] >= 10

# Group by run scheme to count total and explosive plays
explosive_summary = (
    scheme_data.groupby("run_scheme")["explosive"]
    .agg(["count", "sum"])
    .rename(columns={"count": "Total Plays", "sum": "Explosive Plays"})
    .reset_index()
)

# Calculate explosive rate percentage
explosive_summary["Explosive Rate (%)"] = (explosive_summary["Explosive Plays"] / explosive_summary[
    "Total Plays"]) * 100

print("===== Explosive Run Rate: Gap vs Zone =====")
print(explosive_summary)

# Create contingency table for Chi-Square test
table = [
    [explosive_summary[explosive_summary['run_scheme'] == 'Gap']['Explosive Plays'].iloc[0],
     explosive_summary[explosive_summary['run_scheme'] == 'Gap']['Total Plays'].iloc[0] -
     explosive_summary[explosive_summary['run_scheme'] == 'Gap']['Explosive Plays'].iloc[0]],
    [explosive_summary[explosive_summary['run_scheme'] == 'Zone']['Explosive Plays'].iloc[0],
     explosive_summary[explosive_summary['run_scheme'] == 'Zone']['Total Plays'].iloc[0] -
     explosive_summary[explosive_summary['run_scheme'] == 'Zone']['Explosive Plays'].iloc[0]]
]

# Perform Chi-Square test for independence between scheme and explosiveness
chi2, p, dof, expected = chi2_contingency(table)
print(f"\nChi-Square Test:\nChi2 = {chi2:.2f}, p-value = {p:.4f}")

# --- Success Rate by Run Scheme (Overall & by Down) ---

success_data = scheme_data.copy()


# Define success based on NFL advanced statistics (different criteria per down)
def is_success(row):
    if row["down"] == 1:
        return row["yardsGained"] >= 4
    elif row["down"] == 2:
        return row["yardsGained"] >= 0.5 * row["yardsToGo"]
    elif row["down"] in [3, 4]:
        return row["yardsGained"] >= row["yardsToGo"]
    else:
        return False  # Safety net for any unexpected values


# Apply success flag to each play
success_data["success"] = success_data.apply(is_success, axis=1)

# Group by run scheme to summarize overall success rates
success_summary = (
    success_data.groupby("run_scheme")["success"]
    .agg(["count", "sum"])
    .rename(columns={"count": "Total Plays", "sum": "Successful Plays"})
    .reset_index()
)

# Calculate success rate percentage
success_summary["Success Rate (%)"] = (success_summary["Successful Plays"] / success_summary["Total Plays"]) * 100

# Show results
print("===== Success Rate by Run Scheme (Gap vs Zone) =====")
print(success_summary)

# Group by run scheme and down to analyze success rates by down
success_by_down = (
    success_data.groupby(["run_scheme", "down"])["success"]
    .agg(["count", "sum"])
    .rename(columns={"count": "Total Plays", "sum": "Successful Plays"})
    .reset_index()
)

# Add success rate percentage
success_by_down["Success Rate (%)"] = (success_by_down["Successful Plays"] / success_by_down["Total Plays"]) * 100

# Display success rates by run scheme and down
print("===== Success Rate by Run Scheme and Down =====")
print(success_by_down)

# --- Chi-Square Test for Success Rate by Down (Gap vs Zone) ---

# Loop through each unique down to perform chi-square tests
for down in sorted(success_data["down"].unique()):
    gap_success = success_data[(success_data["run_scheme"] == "Gap") & (success_data["down"] == down)]["success"]
    zone_success = success_data[(success_data["run_scheme"] == "Zone") & (success_data["down"] == down)]["success"]

    # Build contingency table for current down: [[Gap Success, Gap Fail], [Zone Success, Zone Fail]]
    table = [
        [gap_success.sum(), gap_success.count() - gap_success.sum()],
        [zone_success.sum(), zone_success.count() - zone_success.sum()]
    ]

    # Perform Chi-Square test for independence
    chi2, p, dof, expected = chi2_contingency(table)

    # Print results for the current down
    print(f"Down: {down}")
    print(f"  Chi² = {chi2:.2f}, p = {p:.4f}")
    print("")

# --- Plot: Success Rate on 1st Down (Gap vs Zone) ---
# Filter for 1st down plays only
first_down_data = success_data[success_data["down"] == 1]

# Group by run scheme to summarize success rates on 1st down
first_down_summary = (
    first_down_data.groupby("run_scheme")["success"]
    .agg(["count", "sum"])
    .rename(columns={"count": "Total Plays", "sum": "Successful Plays"})
    .reset_index()
)

# Calculate success rate percentage for 1st down
first_down_summary["Success Rate (%)"] = (first_down_summary["Successful Plays"] / first_down_summary[
    "Total Plays"]) * 100

# Create bar plot for 1st down success rates
plt.figure(figsize=(6, 4))
sns.barplot(data=first_down_summary, x="run_scheme", y="Success Rate (%)", palette=["#4C72B0", "#55A868"])

# Annotate bar values with percentage
for i, row in first_down_summary.iterrows():
    plt.text(i, row["Success Rate (%)"] + 1, f"{row['Success Rate (%)']:.1f}%", ha="center", fontsize=12,
             fontweight="bold")

# Set plot titles and labels
plt.title("Success Rate on 1st Down: Gap vs Zone Runs", fontsize=14, fontweight="bold")
plt.ylabel("Success Rate (%)")
plt.xlabel("Run Scheme")
plt.ylim(0, max(first_down_summary["Success Rate (%)"]) + 10)
plt.tight_layout()
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.show()

custom_palette_gap_zone = {
    "Gap": "#4C72B0",
    "Zone": "#55A868"
}

# Create bar plot for Average Yards Gained
plt.figure(figsize=(6, 4))
ax = sns.barplot(x="run_scheme", y="Avg Yards Gained", data=overall_scheme_summary, palette=custom_palette_gap_zone)
plt.title("Average Yards Gained: Gap vs Zone Runs", fontsize=14, fontweight="bold")
plt.xlabel("Run Scheme")
plt.ylabel("Average Yards Gained")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=12, fontweight="bold")

plt.ylim(0, overall_scheme_summary["Avg Yards Gained"].max() + 1.0)
plt.tight_layout()
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.show()
