import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind, shapiro, mannwhitneyu, chi2_contingency
from statsmodels.stats.weightstats import ztest

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Data Loading & Initial Preprocessing ---

plays = pd.read_csv("Data/plays.csv")
player_plays = pd.read_csv("Data/player_play.csv")

# Identify motion at player level from player_play data
player_plays["motion_indicator"] = (
    player_plays["inMotionAtBallSnap"].fillna(False) |
    player_plays["shiftSinceLineset"].fillna(False) |
    player_plays["motionSinceLineset"].fillna(False)
)

# Aggregate motion flags to the play level (True if any player had motion)
motion_flags = player_plays.groupby(['gameId', 'playId'])['motion_indicator'].any().reset_index()
motion_flags.rename(columns={'motion_indicator': 'hadMotion'}, inplace=True)

# Merge motion data with the main plays DataFrame
plays = plays.merge(motion_flags, on=["gameId", "playId"], how="left")
plays["hadMotion"] = plays["hadMotion"].fillna(False) # Fill NaN from merge with False (no motion detected)

# --- Apply Outlier Filters ---

# Remove overtime plays (quarter 5)
filtered_plays = plays[plays["quarter"] != 5].copy()

# Remove plays nullified by penalty
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"]

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

# --- Isolate Run Plays ---
run_plays = filtered_plays[filtered_plays["isDropback"] == False].copy()

# --- Split Into Motion and No-Motion Run Plays ---
motion_run_plays = run_plays[run_plays["hadMotion"] == True]
no_motion_run_plays = run_plays[run_plays["hadMotion"] == False]

print(f"Motion run plays: {len(motion_run_plays)}")
print(f"No-Motion run plays: {len(no_motion_run_plays)}")

# --- Remove Explosive Plays (15+ Yards) ---
# Filter out explosive runs (15+ yards) for analysis of typical run play effectiveness
motion_filtered = motion_run_plays[motion_run_plays["yardsGained"] < 15]
no_motion_filtered = no_motion_run_plays[no_motion_run_plays["yardsGained"] < 15]

# --- Average Yards Gained Comparison (Excluding Explosive Plays) ---
avg_motion_yards = motion_filtered["yardsGained"].mean()
avg_no_motion_yards = no_motion_filtered["yardsGained"].mean()

print(f"Average Yards Gained (Motion, <15 yds): {avg_motion_yards:.2f}")
print(f"Average Yards Gained (No Motion, <15 yds): {avg_no_motion_yards:.2f}")

# --- Compute Avg Yards Gained by YardsToGo & Down ---

# Filter for 1st, 2nd, and 3rd downs
plot_data = run_plays[run_plays["down"].isin([1, 2, 3])].copy()
# Group by down, yardsToGo, and motion status, then calculate mean yards gained
grouped = plot_data.groupby(['down', 'yardsToGo', 'hadMotion'])['yardsGained'].mean().reset_index()
# Filter for yardsToGo less than 11 (i.e., 10 or less) for clearer plotting
grouped = grouped[grouped["yardsToGo"] < 11]

# --- Plot: Average Yards Gained on Run Plays by Yards to Go (Motion vs No Motion) ---
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

for i, down in enumerate([1, 2, 3]):
    ax = axes[i]
    subset = grouped[grouped["down"] == down]

    # Plot line for each motion status
    sns.lineplot(data=subset, x="yardsToGo", y="yardsGained", hue="hadMotion", marker="o", ax=ax)

    # Set subplot titles and labels
    ax.set_title(f"Down {down}")
    ax.set_xlabel("Yards to Go")
    ax.set_ylabel("Avg Yards Gained" if i == 0 else "") # Only show Y-label on the first subplot

    # Customize legend (only for the first subplot)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        label_map = {True: "Motion", False: "No Motion"}
        new_labels = [label_map[eval(l)] if l in ["True", "False"] else l for l in labels]
        ax.legend(handles=handles, labels=new_labels, title="Motion")
    else:
        ax.get_legend().remove() # Remove legends from other subplots

plt.suptitle("Average Yards Gained on Run Plays by Yards to Go (Motion vs No Motion)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()

# --- Field Position Analysis (10-yard bins) ---

# Bin field position into 10-yard increments (e.g., 0-9, 10-19, etc.) based on absoluteYardlineNumber
run_plays["field_bin"] = (run_plays["absoluteYardlineNumber"] // 10) * 10

# Group by field position bin and motion status to calculate mean yards gained
field_summary = run_plays.groupby(["field_bin", "hadMotion"])["yardsGained"].mean().reset_index()

# Define labels and colors for plotting
label_map = {False: "No Motion", True: "Motion"}
palette = {False: "#1f77b4", True: "#ff7f0e"}  # Blue for No Motion, Orange for Motion

# Plot average yards gained by field position
plt.figure(figsize=(10, 6))
for motion_value in [False, True]:
    subset = field_summary[field_summary["hadMotion"] == motion_value]
    sns.lineplot(
        data=subset,
        x="field_bin",
        y="yardsGained",
        marker="o",
        label=label_map[motion_value],
        color=palette[motion_value]
    )

plt.title("Average Yards Gained on Run Plays by Field Position (10-yard bins)")
plt.xlabel("Field Position (Absolute Yard Line)")
plt.ylabel("Average Yards Gained")
plt.legend(title="Motion")
plt.xticks(range(0, 110, 10)) # Set x-axis ticks for 0 to 100 in steps of 10
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot: Yards to Go by Down (10–40 Yard Line Only) ---

# Filter run plays to the 10-40 yard line range (e.g., own 10 to opponent 40)
run_plays_10_40 = run_plays[
    (run_plays["absoluteYardlineNumber"] >= 10) &
    (run_plays["absoluteYardlineNumber"] <= 40)
]

# Filter for 1st, 2nd, and 3rd downs
plot_data_filtered = run_plays_10_40[run_plays_10_40["down"].isin([1, 2, 3])]
# Group by down, yardsToGo, and motion status, then calculate mean yards gained
grouped_filtered = plot_data_filtered.groupby(
    ['down', 'yardsToGo', 'hadMotion']
)["yardsGained"].mean().reset_index()

grouped_filtered = grouped_filtered[grouped_filtered['yardsToGo'] < 11] # Filter for yardsToGo less than 11

# Create subplots for each down
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

for i, down in enumerate([1, 2, 3]):
    ax = axes[i]
    subset = grouped_filtered[grouped_filtered["down"] == down]

    sns.lineplot(data=subset, x="yardsToGo", y="yardsGained", hue="hadMotion", marker="o", ax=ax)

    ax.set_title(f"Down {down}")
    ax.set_xlabel("Yards to Go")
    ax.set_ylabel("Avg Yards Gained" if i == 0 else "") # Only show Y-label on the first subplot

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        label_map = {True: "Motion", False: "No Motion"}
        new_labels = [label_map[eval(l)] if l in ["True", "False"] else l for l in labels]
        ax.legend(handles=handles, labels=new_labels, title="Motion")
    else:
        ax.get_legend().remove()

plt.suptitle("Avg Yards Gained on Run Plays by Yards to Go (10–40 Yard Line Only)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- Hypothesis Tests (40 Inward, Down & Distance Specific) ---

print("\n--- Mann-Whitney U Tests: Motion vs No Motion (Yards Gained) ---")

# Filter run plays that are on the opponent's side of the 40-yard line or closer to the goal line
goal_side_runs = run_plays[run_plays["absoluteYardlineNumber"] <= 40]

# Define specific down and distance situations for testing
situations = {
    "1st & 10": (goal_side_runs["down"] == 1) & (goal_side_runs["yardsToGo"] == 10),
    "2nd Down (<11 YTG)": (goal_side_runs["down"] == 2) & (goal_side_runs["yardsToGo"] < 11),
    "3rd Down (<3 YTG)": (goal_side_runs["down"] == 3) & (goal_side_runs["yardsToGo"] < 3),
}

# Iterate through situations and perform Mann-Whitney U tests
for label, condition in situations.items():
    subset = goal_side_runs[condition]
    motion = subset[subset["hadMotion"] == True]["yardsGained"]
    no_motion = subset[subset["hadMotion"] == False]["yardsGained"]

    # Only run test if both groups have data
    if len(motion) > 0 and len(no_motion) > 0:
        # Perform Mann-Whitney U test (alternative='greater' for Motion > No Motion)
        stat, p = mannwhitneyu(motion, no_motion, alternative='greater')
        print(f"\n--- {label} ---")
        print(f"Motion Mean Yards:     {motion.mean():.2f} (n={len(motion)})")
        print(f"No Motion Mean Yards: {no_motion.mean():.2f} (n={len(no_motion)})")
        print(f"Mann-Whitney U Stat:   {stat:.2f}")
        print(f"P-Value (Motion > No Motion): {p:.4f}")
    else:
        print(f"\n--- {label} ---")
        print("Not enough data for both groups to run test.")

# --- Grouped Test (All Situations Combined) ---
# Combine conditions for 1st & 10, 2nd <11 YTG, and 3rd <3 YTG, restricted to opponent 40-yard line inward
combined_condition = (
    ((run_plays["down"] == 1) & (run_plays["yardsToGo"] == 10)) |
    ((run_plays["down"] == 2) & (run_plays["yardsToGo"] < 11)) |
    ((run_plays["down"] == 3) & (run_plays["yardsToGo"] < 3))
) & (run_plays["absoluteYardlineNumber"] <= 40)

combined_subset = run_plays[combined_condition]

motion_group = combined_subset[combined_subset["hadMotion"] == True]["yardsGained"]
no_motion_group = combined_subset[combined_subset["hadMotion"] == False]["yardsGained"]

# Perform Mann-Whitney U Test on the combined data
print("\n--- Combined Test (1st & 10, 2nd <11, 3rd <3 from Opp 40 Inward) ---")
if len(motion_group) > 0 and len(no_motion_group) > 0:
    stat, p = mannwhitneyu(motion_group, no_motion_group, alternative='greater')
    print(f"Motion Mean Yards:     {motion_group.mean():.2f} (n={len(motion_group)})")
    print(f"No Motion Mean Yards: {no_motion_group.mean():.2f} (n={len(no_motion_group)})")
    print(f"Mann-Whitney U Stat:   {stat:.2f}")
    print(f"P-Value (Motion > No Motion): {p:.4f}")
else:
    print("Not enough data to run combined test.")

# --- Welch's T-Test ---
# Perform Welch's t-test (which does not assume equal variance) for the combined groups
t_stat, t_p = ttest_ind(motion_group, no_motion_group, equal_var=False, alternative='greater')

print("\n--- Welch's T-Test ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (Motion > No Motion): {t_p:.4f}")

# --- Z-Test ---
# Perform Z-test (assuming large enough sample sizes for approximation)
z_stat, z_p = ztest(motion_group, no_motion_group, alternative='larger')

print("\n--- Z-Test ---")
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value (Motion > No Motion): {z_p:.4f}")
