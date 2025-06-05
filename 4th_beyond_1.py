import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Configuration ---

# Display all columns without truncation
pd.set_option('display.max_columns', None)

# --- Data Loading and Initial Preparation ---

# Load data for fourth down plays beyond 1 yard
fourth_down_plays_beyond_1 = pd.read_csv("Data/fourth_down_plays_beyond_1.csv")

# Perform initial quality check
print("--- Initial Data Quality Check ---")
print(fourth_down_plays_beyond_1[['yardsToGo', 'down']].count())


# Function to categorize conversion attempt outcome
def categorize_try(row):
    if row['yardsGained'] >= row['yardsToGo']:
        return "First Down"
    else:
        return "Short"


# Apply function to categorize conversion attempts
fourth_down_plays_beyond_1.loc[:, 'successful_conversion'] = fourth_down_plays_beyond_1.apply(categorize_try, axis=1)

# Determine conversion success directly (boolean)
fourth_down_plays_beyond_1['converted'] = (
        fourth_down_plays_beyond_1['yardsGained'] >= fourth_down_plays_beyond_1['yardsToGo']
)

# Calculate and print initial conversion rates
conversion_rates_initial = fourth_down_plays_beyond_1.groupby('converted').size().reset_index(name="Total Attempts")
conversion_rates_initial["Conversion Rate"] = (
        conversion_rates_initial["Total Attempts"] / conversion_rates_initial["Total Attempts"].sum()
)
conversion_rates_initial["converted"] = conversion_rates_initial["converted"].map({True: "First Down", False: "Short"})
conversion_rates_initial.columns = ['Outcome', 'Total Attempts', 'Conversion Rate']

print("\n--- Initial Conversion Rates (Before Outlier Removal) ---")
print(conversion_rates_initial)

# --- Outlier and Garbage Time Removal ---

# Create a copy to avoid SettingWithCopyWarning
filtered_plays = fourth_down_plays_beyond_1.copy()

# Remove overtime plays (quarter 5)
filtered_plays = filtered_plays[filtered_plays["quarter"] != 5]

# Remove plays nullified by penalty
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"]

# Calculate score difference
filtered_plays["score_difference"] = filtered_plays["preSnapHomeScore"] - filtered_plays["preSnapVisitorScore"]

# Remove garbage time in 4th quarter (large score difference)
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) & (filtered_plays["score_difference"].abs() > 17))
]

# Remove garbage time in 4th quarter (late clock, significant score diff, high win probability)
filtered_plays = filtered_plays[
    ~(
            (filtered_plays["quarter"] == 4) &
            (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) < 5)) &
            (filtered_plays["score_difference"].abs() >= 10) &
            (
                    (filtered_plays["preSnapHomeTeamWinProbability"] >= 0.7) |
                    (filtered_plays["preSnapVisitorTeamWinProbability"] >= 0.7)
            )
    )
]

# Remove end-of-half plays in 2nd quarter (last 5 seconds)
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 2) & (
        filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) == 0 and int(x.split(":")[1]) < 5)))
]

# Print size after filtering
print(f"\n--- Filtered Data Size: {filtered_plays.shape[0]} plays ---")

# Recheck conversion rates after initial filtering
conversion_rates_after_filter1 = filtered_plays.groupby("converted").size().reset_index(name="Total Attempts")
conversion_rates_after_filter1["Conversion Rate"] = (
        conversion_rates_after_filter1["Total Attempts"] / conversion_rates_after_filter1["Total Attempts"].sum()
)
conversion_rates_after_filter1["converted"] = conversion_rates_after_filter1["converted"].map(
    {True: "First Down", False: "Short"})
conversion_rates_after_filter1.columns = ["Outcome", "Total Attempts", "Conversion Rate"]

print("\n--- Conversion Rates (After Initial Filtering) ---")
print(conversion_rates_after_filter1)

# --- Further Filtering: Limit yardsToGo ---

# Filter out plays where yardsToGo is greater than 6
filtered_plays = filtered_plays[filtered_plays["yardsToGo"] <= 6].copy()

# Confirm filtering
print(f"\n--- Remaining plays after filtering yardsToGo <= 6: {filtered_plays.shape[0]} ---")

# Recheck conversion rates after all filtering
conversion_rates_final = filtered_plays.groupby("converted").size().reset_index(name="Total Attempts")
conversion_rates_final["Conversion Rate"] = (
        conversion_rates_final["Total Attempts"] / conversion_rates_final["Total Attempts"].sum()
)
conversion_rates_final["converted"] = conversion_rates_final["converted"].map({True: "First Down", False: "Short"})
conversion_rates_final.columns = ["Outcome", "Total Attempts", "Conversion Rate"]

print("\n--- Final Conversion Rates (After All Filtering) ---")
print(conversion_rates_final)

# --- Exploratory Data Analysis (EDA) ---

# Filter for successful and failed conversions
converted_plays = filtered_plays[filtered_plays['converted'] == True].copy()
failed_plays = filtered_plays[filtered_plays['converted'] == False].copy()

# Calculate and print average EPA for converted and failed plays
epa_after_conversion = converted_plays['expectedPointsAdded'].mean()
epa_failed = failed_plays['expectedPointsAdded'].mean()

print(f"\n--- Expected Points Added (EPA) Analysis ---")
print(f"Average EPA after 4th down conversion: {epa_after_conversion:.2f}")
print(f"Average EPA after failed 4th down attempt: {epa_failed:.2f}")

# Plot EPA distribution for converted vs. failed plays
plt.figure(figsize=(8, 5))
sns.kdeplot(data=converted_plays, x='expectedPointsAdded', label='Converted', fill=True)
sns.kdeplot(data=failed_plays, x='expectedPointsAdded', label='Failed', fill=True)
plt.title('EPA Distribution: 4th Down Conversion vs. Failure')
plt.xlabel('Expected Points Added')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Compare average EPA by yards to go (from 6 to 2)
print("\n--- Average EPA by Yards to Go (6 to 2) ---")
for yards in range(6, 1, -1):
    subset = filtered_plays[filtered_plays["yardsToGo"] == yards]
    if subset.empty:
        print(f"No data for {yards} Yards to Go.")
        continue

    converted = subset[subset["converted"] == True]
    failed = subset[subset["converted"] == False]

    print(f"\n--- {yards} Yards to Go ---")
    print(f"Avg EPA (Converted): {converted['expectedPointsAdded'].mean():.2f}")
    print(f"Avg EPA (Failed): {failed['expectedPointsAdded'].mean():.2f}")

# Plot EPA distribution for each yards to go (6 to 2)
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
fig.suptitle("EPA Distribution by 4th Down Conversion Outcome\n(Yards to Go: 6 to 2)", fontsize=16)

for idx, yards in enumerate(range(6, 1, -1)):
    ax = axes[idx]
    subset = filtered_plays[filtered_plays["yardsToGo"] == yards]

    if subset.empty:
        ax.set_title(f"{yards} Yards to Go (No Data)")
        ax.axis('off')
        continue

    converted = subset[subset["converted"] == True]
    failed = subset[subset["converted"] == False]

    sns.kdeplot(data=converted, x="expectedPointsAdded", ax=ax, fill=True, label="Converted", color="blue")
    sns.kdeplot(data=failed, x="expectedPointsAdded", ax=ax, fill=True, label="Failed", color="orange")

    ax.set_title(f"{yards} Yards to Go")
    ax.set_xlabel("EPA")
    if idx == 0:
        ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- Field Position Analysis ---

# Create 10-yard bins for field position
filtered_plays["field_position_bin_10"] = (filtered_plays["absoluteYardlineNumber"] // 10) * 10

# Count plays per 10-yard bin
field_bin_counts_10 = (
    filtered_plays["field_position_bin_10"]
    .value_counts()
    .sort_index()
    .reset_index()
)
field_bin_counts_10.columns = ["Field Position Bin (10-yard)", "Number of 4th Down Attempts"]

# Print field position counts
print("\n--- Number of 4th Down Attempts by 10-Yard Field Position Bin ---")
print(field_bin_counts_10)

# Look at specific deep territory plays (10-19 yardline) for detailed info
deep_territory_plays = filtered_plays[
    (filtered_plays["absoluteYardlineNumber"] >= 10) &
    (filtered_plays["absoluteYardlineNumber"] < 20)
    ].copy()

deep_territory_info = deep_territory_plays[[
    "gameClock", "quarter", "score_difference", "converted", "yardsToGo",
    "down", "expectedPointsAdded", 'preSnapHomeTeamWinProbability', 'homeTeamWinProbabilityAdded'
]]

print("\n--- Detailed Info for Plays between 10-19 Yardline ---")
print(deep_territory_info)

# --- Drive Scoring Analysis ---

# Filter for 4th down plays on opponent's side of the field (absoluteYardlineNumber <= 50)
fourth_down_opponent_side = filtered_plays[
    (filtered_plays["down"] == 4) &
    (filtered_plays["absoluteYardlineNumber"] <= 50)
    ].copy()

# Group by yardsToGo and summarize attempts/conversions
summary_by_ytg_opponent = fourth_down_opponent_side.groupby("yardsToGo").agg(
    Attempts=("converted", "count"),
    Conversions=("converted", "sum")
).reset_index()

# Calculate conversion rate for opponent's side
summary_by_ytg_opponent["Conversion Rate"] = summary_by_ytg_opponent["Conversions"] / summary_by_ytg_opponent[
    "Attempts"]
summary_by_ytg_opponent = summary_by_ytg_opponent.sort_values(by="yardsToGo")

print("\n--- Conversion Summary by Yards to Go (Opponent's Side) ---")
print(summary_by_ytg_opponent)

# Track if a score occurs on the same drive after a 4th down attempt (within opponent's 30)
fourth_down_opponent_30 = filtered_plays[
    (filtered_plays["down"] == 4) &
    (filtered_plays["absoluteYardlineNumber"] <= 30)
    ].copy()

# Sort plays for sequential analysis
filtered_plays_sorted_for_drive = filtered_plays.sort_values(by=["gameId", "playId"]).reset_index(drop=True)
filtered_plays_sorted_for_drive["row_id"] = filtered_plays_sorted_for_drive.index

# Merge row_id into the 4th down plays
fourth_down_opponent_30 = fourth_down_opponent_30.merge(
    filtered_plays_sorted_for_drive[["gameId", "playId", "row_id"]],
    on=["gameId", "playId"],
    how="left"
)

# Loop through 4th down plays to check for subsequent scores on the same drive
scored_after = []
for idx, row in fourth_down_opponent_30.iterrows():
    game_id = row["gameId"]
    start_idx = row["row_id"]
    offense = row["possessionTeam"]
    initial_home = row["preSnapHomeScore"]
    initial_visitor = row["preSnapVisitorScore"]

    future_plays = filtered_plays_sorted_for_drive[
        (filtered_plays_sorted_for_drive["gameId"] == game_id) &
        (filtered_plays_sorted_for_drive["row_id"] > start_idx)
        ]

    scored = False
    for _, future in future_plays.iterrows():
        # Break if possession changes (drive ends)
        if future["possessionTeam"] != offense:
            break
        # Check if score increased for either team
        if (future["preSnapHomeScore"] > initial_home) or (future["preSnapVisitorScore"] > initial_visitor):
            scored = True
            break
    scored_after.append(scored)

# Add scoring outcome to the summary
fourth_down_opponent_30["scored_after"] = scored_after

# Summarize scoring rates by yardsToGo
drive_score_summary = fourth_down_opponent_30.groupby("yardsToGo").agg(
    Attempts=("converted", "count"),
    Conversions=("converted", "sum"),
    Scored=("scored_after", "sum")
).reset_index()

drive_score_summary["Conversion Rate"] = drive_score_summary["Conversions"] / drive_score_summary["Attempts"]
drive_score_summary["Scored Rate"] = drive_score_summary["Scored"] / drive_score_summary["Attempts"]

print("\n--- Drive Scoring Summary by Yards to Go (Opponent's 30-Yard Line) ---")
print(drive_score_summary)

# --- Offensive Rhythm and Conversion Rates ---

# Reload full plays data (cleaned for garbage time)
plays_full_cleaned = pd.read_csv("Data/plays.csv")
plays_full_cleaned = plays_full_cleaned[plays_full_cleaned["quarter"] != 5]
plays_full_cleaned = plays_full_cleaned[plays_full_cleaned["playNullifiedByPenalty"] == "N"]
plays_full_cleaned["score_difference"] = plays_full_cleaned["preSnapHomeScore"] - plays_full_cleaned[
    "preSnapVisitorScore"]
plays_full_cleaned = plays_full_cleaned[
    ~((plays_full_cleaned["quarter"] == 4) & (plays_full_cleaned["score_difference"].abs() > 17))
]
plays_full_cleaned = plays_full_cleaned[
    ~((plays_full_cleaned["quarter"] == 4) &
      (plays_full_cleaned["gameClock"].apply(lambda x: int(x.split(":")[0]) < 5)) &
      (plays_full_cleaned["score_difference"].abs() >= 10) &
      ((plays_full_cleaned["preSnapHomeTeamWinProbability"] >= 0.7) |
       (plays_full_cleaned["preSnapVisitorTeamWinProbability"] >= 0.7)))
]
plays_full_cleaned = plays_full_cleaned[
    ~((plays_full_cleaned["quarter"] == 2) & (
        plays_full_cleaned["gameClock"].apply(lambda x: int(x.split(":")[0]) == 0 and int(x.split(":")[1]) < 5)))
]

# Create yards gained bins for all plays
plays_full_cleaned["yardsGained_bin"] = (plays_full_cleaned["yardsGained"] // 0.5) * 0.5

# Aggregate play count by yards gained bins
avg_yards_buckets_all_plays = plays_full_cleaned.groupby("yardsGained_bin").size().reset_index(name="Play Count")

# Get conversion rates by yardsToGo from the specific filtered_plays (4th down data)
conversion_by_ytg = filtered_plays.groupby("yardsToGo").agg(
    Attempts=("converted", "count"),
    Conversions=("converted", "sum")
).reset_index()
conversion_by_ytg["Conversion Rate"] = conversion_by_ytg["Conversions"] / conversion_by_ytg["Attempts"]

# Merge for comparison
comparison_yards_buckets_conversions = pd.merge(
    avg_yards_buckets_all_plays,
    conversion_by_ytg,
    left_on="yardsGained_bin",
    right_on="yardsToGo",
    how="inner"
)

print("\n--- Comparison: Average Yards Gained (All Plays) vs. 4th Down Conversion ---")
print(comparison_yards_buckets_conversions[[
    "yardsGained_bin", "Play Count", "Attempts", "Conversions", "Conversion Rate"
]])

# Calculate average yards gained on previous plays in the same drive
filtered_plays_sorted_for_avg_yards = filtered_plays.sort_values(by=["gameId", "playId"]).reset_index(drop=True)

avg_yards_before_all = []
for idx, row in filtered_plays_sorted_for_avg_yards.iterrows():
    gid = row["gameId"]
    pid = row["playId"]
    team = row["possessionTeam"]

    earlier_plays = filtered_plays_sorted_for_avg_yards[
        (filtered_plays_sorted_for_avg_yards["gameId"] == gid) &
        (filtered_plays_sorted_for_avg_yards["playId"] < pid) &
        (filtered_plays_sorted_for_avg_yards["possessionTeam"] == team)
        ]
    avg_yards = earlier_plays["yardsGained"].mean() if not earlier_plays.empty else np.nan
    avg_yards_before_all.append(avg_yards)

filtered_plays_sorted_for_avg_yards["avg_yards_before_all"] = avg_yards_before_all
filtered_plays_sorted_for_avg_yards["avg_yards_bin"] = (
        (filtered_plays_sorted_for_avg_yards["avg_yards_before_all"] // 0.5) * 0.5)

# Group by average yards bin and calculate conversion rate
conversion_by_avg_yards_all = filtered_plays_sorted_for_avg_yards.groupby("avg_yards_bin").agg(
    Attempts=("converted", "count"),
    Conversions=("converted", "sum")
).reset_index()

conversion_by_avg_yards_all["Conversion Rate"] = (
        conversion_by_avg_yards_all["Conversions"] / conversion_by_avg_yards_all["Attempts"]
)

print("\n--- 4th Down Conversion Rate by Average Yards Gained on Prior Plays in Drive ---")
print(conversion_by_avg_yards_all)

# --- Final Hypothesis Testing (EPA and WPA) ---

# Perform one-sample t-test for EPA
epa_values = filtered_plays["expectedPointsAdded"].dropna()
t_stat_epa, p_val_epa = ttest_1samp(epa_values, popmean=0, alternative='less')

# Calculate WPA
filtered_plays.loc[:, "wpa"] = np.where(
    filtered_plays["possessionTeam"] == filtered_plays["yardlineSide"],
    filtered_plays["homeTeamWinProbabilityAdded"],
    filtered_plays["visitorTeamWinProbilityAdded"]
)
wpa_values = filtered_plays["wpa"].dropna()
t_stat_wpa, p_val_wpa = ttest_1samp(wpa_values, popmean=0, alternative='less')

# Print overall test results
print("\n--- Final Test: Does going for it (YardsToGo > 1) provide positive offensive value? ---")
print(f"EPA — t-stat: {t_stat_epa:.3f}, p-value: {p_val_epa:.3f}")
print(f"WPA — t-stat: {t_stat_wpa:.3f}, p-value: {p_val_wpa:.3f}")

# Filter for 4th-and-2 plays specifically
plays_2yd = filtered_plays[filtered_plays["yardsToGo"] == 2].copy()

# EPA values for 4th-and-2
epa_2yd = plays_2yd["expectedPointsAdded"].dropna()

# Calculate WPA for 4th-and-2 plays
plays_2yd.loc[:, "wpa"] = np.where(  # Use .loc for explicit assignment
    plays_2yd["possessionTeam"] == plays_2yd["yardlineSide"],
    plays_2yd["homeTeamWinProbabilityAdded"],
    plays_2yd["visitorTeamWinProbilityAdded"]
)
wpa_2yd = plays_2yd["wpa"].dropna()

# One-sample t-tests for 4th-and-2
t_stat_epa_2, p_val_epa_2 = ttest_1samp(epa_2yd, popmean=0, alternative='greater')
t_stat_wpa_2, p_val_wpa_2 = ttest_1samp(wpa_2yd, popmean=0, alternative='greater')

# Print 4th-and-2 specific test results
print("\n--- 4th-and-2 Test: Does going for it specifically on 4th-and-2 help? ---")
print(f"EPA — t-stat: {t_stat_epa_2:.3f}, p-value: {p_val_epa_2:.3f}")
print(f"WPA — t-stat: {t_stat_wpa_2:.3f}, p-value: {p_val_wpa_2:.3f}")