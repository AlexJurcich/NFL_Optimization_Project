import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Data Loading & Initial Preprocessing ---

plays = pd.read_csv("Data/plays.csv")

# Filter out outlier and garbage time plays
filtered_plays = plays.copy()
filtered_plays = filtered_plays[filtered_plays["quarter"] != 5] # Remove overtime plays
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"] # Remove plays nullified by penalty

# Calculate pre-snap score difference
filtered_plays["score_difference"] = (
        filtered_plays["preSnapHomeScore"] - filtered_plays["preSnapVisitorScore"]
)

# Remove garbage time in Q4 based on large score difference (> 17 points)
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) & (filtered_plays["score_difference"].abs() > 17))
]

# Remove more specific garbage time scenarios in Q4 (clock under 5 min, score lopsided, high win probability)
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

# Isolate ONLY 4th and 1 plays for this analysis
filtered_plays = filtered_plays[(filtered_plays["down"] == 4) & (filtered_plays["yardsToGo"] == 1)].copy()

# --- Conversion Rate by Offensive Team ---

# Create the 'converted' column (True if yards gained >= yards to go)
filtered_plays["converted"] = filtered_plays["yardsGained"] >= filtered_plays["yardsToGo"]

# Group by possession team and calculate mean conversion rate and total attempts
team_conversion = filtered_plays.groupby("possessionTeam")["converted"].agg(['mean', 'count']).reset_index()
team_conversion.columns = ["Team", "Conversion Rate", "Total Attempts"]

# Sort teams by conversion rate in descending order
team_conversion = team_conversion.sort_values(by="Conversion Rate", ascending=False)

# Display the DataFrame
print(team_conversion)

# Optional: Visualize conversion rates by team
plt.figure(figsize=(12, 6))
sns.barplot(x="Conversion Rate", y="Team", data=team_conversion, palette="viridis")

plt.title("4th and 1 Conversion Rate by Team")
plt.xlabel("Conversion Rate")
plt.ylabel("Team")
plt.xlim(0, 1) # Set x-axis limit from 0 to 1 for rates
plt.grid(axis='x', linestyle='--') # Add horizontal grid lines
plt.tight_layout()
plt.show()


# --- Classify Play Type for 4th & 1 (QB Sneak, Pass, or Run) ---
def classify_play(row):
    if row["qbSneak"] == 1:
        return "QB Sneak"
    elif row["isDropback"]:
        return "Pass"
    else:
        return "Run"

filtered_plays["play_type"] = filtered_plays.apply(classify_play, axis=1)

# Aggregate success rates by classified play type
play_type_summary = filtered_plays.groupby("play_type")["converted"].agg(['mean', 'count']).reset_index()
play_type_summary.columns = ["Play Type", "Conversion Rate", "Total Attempts"]

# Sort and display the summary
play_type_summary = play_type_summary.sort_values(by="Conversion Rate", ascending=False)
print(play_type_summary)

# Plot 4th and 1 conversion rate by play type
plt.figure(figsize=(8, 5))
sns.barplot(data=play_type_summary, x="Conversion Rate", y="Play Type", palette="deep")

plt.title("4th and 1 Conversion Rate by Play Type")
plt.xlabel("Conversion Rate")
plt.ylabel("Play Type")
plt.xlim(0, 1) # Set x-axis limit from 0 to 1 for rates
plt.grid(axis='x', linestyle='--') # Add horizontal grid lines
plt.tight_layout()
plt.show()

# --- QB Sneak Performance by Team ---

# Step 1: Filter only QB Sneak plays
qb_sneaks = filtered_plays[filtered_plays["play_type"] == "QB Sneak"].copy()

# Step 2: Group by team and count successful and failed QB sneaks
qb_sneak_summary = qb_sneaks.groupby(["possessionTeam", "converted"]).size().unstack(fill_value=0).reset_index()

# Step 3: Rename columns and calculate total attempts and conversion rate
qb_sneak_summary.columns = ["Team", "Failed", "Successful"]
qb_sneak_summary["Total Attempts"] = qb_sneak_summary["Failed"] + qb_sneak_summary["Successful"]
qb_sneak_summary["Conversion Rate"] = qb_sneak_summary["Successful"] / qb_sneak_summary["Total Attempts"]

# Step 4: Sort by conversion rate and display
qb_sneak_summary = qb_sneak_summary.sort_values(by="Conversion Rate", ascending=False)
print(qb_sneak_summary)

# --- QB Sneak Usage Rate by Team ---

# Count all 4th-and-1 plays per team
all_plays = filtered_plays.groupby("possessionTeam").size().reset_index(name="Total_4th_and_1")

# Count QB sneak attempts per team
sneak_counts = qb_sneaks.groupby("possessionTeam").size().reset_index(name="QB_Sneak_Attempts")

# Merge total 4th-and-1 attempts with QB sneak attempts
sneak_usage = pd.merge(all_plays, sneak_counts, on="possessionTeam", how="left")
sneak_usage["QB_Sneak_Attempts"] = sneak_usage["QB_Sneak_Attempts"].fillna(0).astype(int) # Fill NaNs with 0 and convert to int
sneak_usage["Sneak Rate"] = sneak_usage["QB_Sneak_Attempts"] / sneak_usage["Total_4th_and_1"] # Calculate sneak rate

# Sort by sneak rate and display
sneak_usage = sneak_usage.sort_values(by="Sneak Rate", ascending=False)
print(sneak_usage)

# --- 4th & 1 Conversion Rate After Failed 3rd & 1 ---

# Filter for 3rd down and 1 plays
third_and_1 = plays[(plays["down"] == 3) & (plays["yardsToGo"] == 1)].copy()
third_and_1["converted"] = third_and_1["yardsGained"] >= 1 # Mark if 3rd & 1 was converted

# Identify failed 3rd-and-1 plays
failed_third_and_1 = third_and_1[third_and_1["converted"] == False].copy()

# All 4th-and-1 plays (from the original 'plays' DataFrame, not the pre-filtered one)
fourth_and_1 = plays[(plays["down"] == 4) & (plays["yardsToGo"] == 1)].copy()

# Shift playId of 3rd-down failures by +1 to match potential subsequent 4th-down attempt
failed_third_and_1["nextPlayId"] = failed_third_and_1["playId"] + 1

# Merge to find 4th-and-1 attempts that immediately follow a failed 3rd-and-1
after_fail = pd.merge(
    failed_third_and_1[["gameId", "nextPlayId", "possessionTeam"]],
    fourth_and_1[["gameId", "playId", "possessionTeam", "yardsGained"]],
    left_on=["gameId", "nextPlayId", "possessionTeam"],
    right_on=["gameId", "playId", "possessionTeam"],
    how="inner" # Use inner merge to keep only matching sequences
)

# Calculate 4th-and-1 conversion result for these specific plays
after_fail["converted"] = after_fail["yardsGained"] >= 1

total_attempts = after_fail.shape[0]
successful_attempts = after_fail["converted"].sum()
conversion_rate = successful_attempts / total_attempts

print(f"Total 4th-and-1 attempts after failed 3rd-and-1: {total_attempts}")
print(f"Successful conversions: {successful_attempts}")
print(f"Conversion Rate: {conversion_rate:.2%}")

# --- Team Behavior After Failed 3rd & 1 (Go-for-it vs. No Attempt) ---

# Count failed 3rd-and-1 plays by team
third_1_fail_counts = failed_third_and_1.groupby("possessionTeam").size().reset_index(name="Failed_3rd_and_1")

# Count how many of those resulted in a 4th-and-1 go-for-it attempt
fourth_1_after_fail_counts = after_fail.groupby("possessionTeam").size().reset_index(name="4th_and_1_Attempts")

# Merge counts to analyze team behavior
team_behavior = pd.merge(third_1_fail_counts, fourth_1_after_fail_counts, on="possessionTeam", how="left")
team_behavior["4th_and_1_Attempts"] = team_behavior["4th_and_1_Attempts"].fillna(0).astype(int)
team_behavior["No_Attempt"] = team_behavior["Failed_3rd_and_1"] - team_behavior["4th_and_1_Attempts"]
team_behavior["No_Attempt_Rate"] = team_behavior["No_Attempt"] / team_behavior["Failed_3rd_and_1"]

print(team_behavior.sort_values(by="No_Attempt_Rate", ascending=False))

# --- 4th & 1 Conversion Rate by Yard Line ---

# Only include 4th-and-1 go-for-it attempts (already filtered in 'filtered_plays' at the start)
# Group by yardline bucket (using 5-yard increments for smoother visualization)
filtered_plays["yard_bin"] = (filtered_plays["absoluteYardlineNumber"] // 5) * 5

# Group by yard bin and count total attempts at each location
yardline_summary = filtered_plays.groupby("yard_bin").size().reset_index(name="Attempts")

# Calculate conversion rate at each yard line
conversion_by_yard = filtered_plays.groupby("yard_bin")["converted"].mean().reset_index(name="Conversion Rate")

# Merge both for plotting
heatmap_data = pd.merge(yardline_summary, conversion_by_yard, on="yard_bin")

# Plot 4th-and-1 conversion rate by yard line
plt.figure(figsize=(14, 5))
sns.barplot(data=heatmap_data, x="yard_bin", y="Conversion Rate", palette="YlGnBu")

plt.title("4th-and-1 Conversion Rate by Yard Line")
plt.xlabel("Absolute Yardline (0 = Own End Zone, 100 = Opponent End Zone)")
plt.ylabel("Conversion Rate")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- EPA Comparison for 4th & 1 Decisions (Go-for-it vs. Punt vs. Field Goal) ---

# Filter for 4th & 1 plays from the original 'plays' DataFrame for comprehensive decision analysis
# Go-for-it EPA: Filter for plays that are runs or passes on 4th & 1
go_attempts = plays[
    (plays["down"] == 4) &
    (plays["yardsToGo"] == 1) &
    (plays["playDescription"].str.contains("run|pass", case=False, na=False)) # Crude filter for go-for-it plays
]
go_epa = go_attempts["expectedPointsAdded"].mean() # Calculate mean EPA for go-for-it attempts

# Punt EPA: Filter for plays that are punts on 4th & 1
punt_attempts = plays[
    (plays["down"] == 4) &
    (plays["yardsToGo"] == 1) &
    (plays["playDescription"].str.contains("punt", case=False, na=False))
]
punt_epa = punt_attempts["expectedPointsAdded"].mean()

# Field Goal EPA: Filter for plays that are field goals on 4th & 1
fg_attempts = plays[
    (plays["down"] == 4) &
    (plays["yardsToGo"] == 1) &
    (plays["playDescription"].str.contains("field goal", case=False, na=False))
]
fg_epa = fg_attempts["expectedPointsAdded"].mean()

print(f"Go for it EPA: {go_epa:.3f}")
print(f"Punt EPA: {punt_epa:.3f}")
print(f"Field Goal EPA: {fg_epa:.3f}")