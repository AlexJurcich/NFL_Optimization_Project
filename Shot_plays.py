import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# # --- Data Loading (Batch) ---
# tracking_weeks = []
# for week in range(1, 5):
#     df = pd.read_csv(f"Data/tracking_week_{week}.csv")
#     tracking_weeks.append(df)
#
# tracking = pd.concat(tracking_weeks, ignore_index=True)
# tracking.to_csv("Data/tracking_all_weeks.csv", index=False)

# --- Load & Preprocess Main Data ---

# Load a sampled version of tracking data for performance
tracking = pd.read_csv('Data/tracking_sampled_11pct.csv')

# Load plays data
plays = pd.read_csv("Data/plays.csv")

# Remove outliers and extreme win-probability situations from plays data
filtered_plays = plays[plays["quarter"] != 5]  # Remove overtime plays
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"]  # Remove plays nullified by penalty
filtered_plays["score_difference"] = filtered_plays["preSnapHomeScore"] - filtered_plays[
    "preSnapVisitorScore"]  # Calculate score difference
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) & (filtered_plays["score_difference"].abs() > 17))  # Remove Q4 blowouts
]
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 4) &
      (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) < 5)) &
      (filtered_plays["score_difference"].abs() >= 10) &
      ((filtered_plays["preSnapHomeTeamWinProbability"] >= 0.7) |
       (filtered_plays["preSnapVisitorTeamWinProbability"] >= 0.7)))
    # Remove Q4 garbage time with lopsided win probability
]
filtered_plays = filtered_plays[
    ~((filtered_plays["quarter"] == 2) &
      (filtered_plays["gameClock"].apply(lambda x: int(x.split(":")[0]) == 0 and int(x.split(":")[1]) < 5)))
    # Remove end-of-half plays in Q2
]

# --- Identify Shot Plays ---
# Filter for pass plays with valid pass results
pass_plays = filtered_plays[filtered_plays["passResult"].notna()]
# Keep only relevant events from tracking data for pass plays (pass_arrived or pass_outcome_caught)
tracking_passes = tracking[tracking["event"].isin(["pass_arrived", "pass_outcome_caught"])].copy()

# Merge tracking data with filtered pass plays to ensure valid plays
tracking_passes = tracking_passes.merge(pass_plays[["gameId", "playId"]], on=["gameId", "playId"])


# Normalize player x-position relative to line of scrimmage (x = 0 at LOS)
# Assumes line of scrimmage is the minimum x-coordinate at snap for the offensive team
def get_relative_depth(df):
    if df["playDirection"].iloc[0] == "right":
        return df["x"] - df["x"].min()
    else:
        return df["x"].max() - df["x"]


# Apply function to calculate relative depth for each player on each play
tracking_passes["relative_depth"] = tracking_passes.groupby(["gameId", "playId"]).apply(get_relative_depth).reset_index(
    drop=True)

# Count number of offensive players 20+ yards downfield per play
deep_targets = tracking_passes[tracking_passes["relative_depth"] >= 20]
deep_counts = deep_targets.groupby(["gameId", "playId"]).size().reset_index(name="deep_count")

# Label plays with 2 or more offensive players 20+ yards deep as 'shot plays'
shot_plays = deep_counts[deep_counts["deep_count"] >= 2]
shot_plays["is_shot_play"] = True

# Merge 'is_shot_play' flag back into the main filtered_plays DataFrame
filtered_plays = filtered_plays.merge(shot_plays[["gameId", "playId", "is_shot_play"]], on=["gameId", "playId"],
                                      how="left")
filtered_plays["is_shot_play"] = filtered_plays["is_shot_play"].fillna(False)

# Pick an example shot play to visualize (optional, for debugging/illustration)
example_play = shot_plays.iloc[0]
game_id = example_play["gameId"]
play_id = example_play["playId"]

# Filter tracking data for this example play and relevant pass events
example_snapshot = tracking[
    (tracking["gameId"] == game_id) &
    (tracking["playId"] == play_id) &
    (tracking["event"].isin(["pass_arrived", "pass_outcome_caught"]))
    ]

# --- Compare Average Yards Gained (Overall) ---

print(filtered_plays['is_shot_play'].value_counts())

# Filter for shot plays that are dropbacks
shot_plays = filtered_plays[
    (filtered_plays['is_shot_play'] == True) &
    (filtered_plays['isDropback'] == True)
    ]

# Filter for non-shot plays that are dropbacks
not_shot_plays = filtered_plays[
    (filtered_plays['is_shot_play'] == False) &
    (filtered_plays['isDropback'] == True)
    ]

# Calculate and print average yards gained for both categories
avg_yards_gained_shot = np.round(shot_plays['yardsGained'].mean(), 3)
avg_yards_gained_not_shot = np.round(not_shot_plays['yardsGained'].mean(), 3)

print(f"Avg yards gained om shot plays: {avg_yards_gained_shot}")
print(f"Avg yards gained on regular passes: {avg_yards_gained_not_shot}")

# --- Compare Average Yards Gained (Midfield) --
# -
# Filter plays occurring between the 40 and 60-yard lines (midfield)
midfield_sequence = filtered_plays[filtered_plays["absoluteYardlineNumber"].between(40, 60)]

# Filter for shot plays that are dropbacks within midfield
shot_plays = midfield_sequence[
    (midfield_sequence['is_shot_play'] == True) &
    (midfield_sequence['isDropback'] == True)
    ]

# Filter for non-shot plays that are dropbacks within midfield
not_shot_plays = midfield_sequence[
    (midfield_sequence['is_shot_play'] == False) &
    (midfield_sequence['isDropback'] == True)
    ]

# Calculate and print average yards gained for both categories in midfield
avg_yards_gained_shot = np.round(shot_plays['yardsGained'].mean(), 3)
avg_yards_gained_not_shot = np.round(not_shot_plays['yardsGained'].mean(), 3)

print(f"Avg yards gained om shot plays (midfield): {avg_yards_gained_shot}")
print(f"Avg yards gained on regular passes (midfield): {avg_yards_gained_not_shot}")

# --- Compare Average Yards Gained (2nd Down at Midfield) ---

# Filter plays occurring on 2nd down and between the 40 and 60-yard lines
midfield_sequence_2nd = filtered_plays[(filtered_plays["absoluteYardlineNumber"].between(40, 60)) &
                                       (filtered_plays['down'] == 2)
                                       ]

# Filter for shot plays that are dropbacks on 2nd down in midfield
shot_plays = midfield_sequence_2nd[
    (midfield_sequence_2nd['is_shot_play'] == True) &
    (midfield_sequence_2nd['isDropback'] == True)
    ]

# Filter for non-shot plays that are dropbacks on 2nd down in midfield
not_shot_plays = midfield_sequence_2nd[
    (midfield_sequence_2nd['is_shot_play'] == False) &
    (midfield_sequence_2nd['isDropback'] == True)
    ]

# Calculate and print average yards gained for both categories on 2nd down at midfield
avg_yards_gained_shot = np.round(shot_plays['yardsGained'].mean(), 3)
avg_yards_gained_not_shot = np.round(not_shot_plays['yardsGained'].mean(), 3)

print(f"Avg yards gained om shot plays (2nd @ midfield): {avg_yards_gained_shot}")
print(f"Avg yards gained on regular passes (2nd @ midfield): {avg_yards_gained_not_shot}")

# --- Summary so far ---
'''
Looking at the avg yards gain for shot plays compared to regular passes, we see that the avg is higher for shot plays.
This isn't surprising beacuse we are throwing the ball further down the field most of the time. In order ot see when
the best time to run a shot play we need to compare the EPA (Expected Points added) after the play.
'''

# --- Compare Average EPA Gained (Overall) ---
print(filtered_plays['is_shot_play'].value_counts())

# Filter for shot plays that are dropbacks
shot_plays = filtered_plays[
    (filtered_plays['is_shot_play'] == True) &
    (filtered_plays['isDropback'] == True)
    ]

# Filter for non-shot plays that are dropbacks
not_shot_plays = filtered_plays[
    (filtered_plays['is_shot_play'] == False) &
    (filtered_plays['isDropback'] == True)
    ]

# Calculate and print average EPA gained for both categories
avg_EPA_gained_shot = np.round(shot_plays['expectedPointsAdded'].mean(), 3)
avg_EPA_gained_not_shot = np.round(not_shot_plays['expectedPointsAdded'].mean(), 3)

print(f"\nAvg EPA gained om shot plays: {avg_EPA_gained_shot}")
print(f"Avg EPA gained on regular passes: {avg_EPA_gained_not_shot}")

# Perform T-test for EPA comparison
t_stat, p_val = ttest_ind(
    shot_plays['expectedPointsAdded'].dropna(),
    not_shot_plays['expectedPointsAdded'].dropna(),
    equal_var=False
)

# Calculate completion rates for shot vs. non-shot plays
shot_completion_rate = (shot_plays['passResult'] == 'C').mean()
non_shot_completion_rate = (not_shot_plays['passResult'] == 'C').mean()

print(f"Shot Play Completion %: {shot_completion_rate:.3f}")
print(f"Non-Shot Completion %: {non_shot_completion_rate:.3f}")

print(f't stat: {np.round(t_stat, 4)}')
print(f'p_val: {np.round(p_val, 4)}')

# --- Compare Average EPA Gained (Midfield) ---
# Filter plays occurring between the 40 and 60-yard lines (midfield)
midfield_sequence = filtered_plays[filtered_plays["absoluteYardlineNumber"].between(40, 60)]

# Filter for shot plays that are dropbacks within midfield
shot_plays = midfield_sequence[
    (midfield_sequence['is_shot_play'] == True) &
    (midfield_sequence['isDropback'] == True)
    ]

# Filter for non-shot plays that are dropbacks within midfield
not_shot_plays = midfield_sequence[
    (midfield_sequence['is_shot_play'] == False) &
    (midfield_sequence['isDropback'] == True)
    ]

# Calculate and print average EPA gained for both categories in midfield
avg_EPA_gained_shot = np.round(shot_plays['expectedPointsAdded'].mean(), 3)
avg_EPA_gained_not_shot = np.round(not_shot_plays['expectedPointsAdded'].mean(), 3)

print(f"\nAvg EPA gained om shot plays (midfield): {avg_EPA_gained_shot}")
print(f"Avg EPA gained on regular passes (midfield): {avg_EPA_gained_not_shot}")

# Calculate completion rates for shot vs. non-shot plays in midfield
shot_completion_rate = (shot_plays['passResult'] == 'C').mean()
non_shot_completion_rate = (not_shot_plays['passResult'] == 'C').mean()

print(f"Shot Play Completion %: {shot_completion_rate:.3f}")
print(f"Non-Shot Completion %: {non_shot_completion_rate:.3f}")

# Perform T-test for EPA comparison in midfield
t_stat, p_val = ttest_ind(
    shot_plays['expectedPointsAdded'].dropna(),
    not_shot_plays['expectedPointsAdded'].dropna(),
    equal_var=False
)
print(f't stat: {np.round(t_stat, 4)}')
print(f'p_val: {np.round(p_val, 4)}')

# --- Compare Average EPA Gained (2nd Down at Midfield) ---
# Filter plays occurring on 2nd down and between the 40 and 60-yard lines
midfield_sequence_2nd = filtered_plays[(filtered_plays["absoluteYardlineNumber"].between(40, 60)) &
                                       (filtered_plays['down'] == 2)
                                       ]

# Filter for shot plays that are dropbacks on 2nd down in midfield
shot_plays = midfield_sequence_2nd[
    (midfield_sequence_2nd['is_shot_play'] == True) &
    (midfield_sequence_2nd['isDropback'] == True)
    ]

# Filter for non-shot plays that are dropbacks on 2nd down in midfield
not_shot_plays = midfield_sequence_2nd[
    (midfield_sequence_2nd['is_shot_play'] == False) &
    (midfield_sequence_2nd['isDropback'] == True)
    ]

# Calculate and print average EPA gained for both categories on 2nd down at midfield
avg_EPA_gained_shot = np.round(shot_plays['expectedPointsAdded'].mean(), 3)
avg_EPA_gained_not_shot = np.round(not_shot_plays['expectedPointsAdded'].mean(), 3)

# Calculate completion rates for shot vs. non-shot plays on 2nd down at midfield
shot_completion_rate = (shot_plays['passResult'] == 'C').mean()
non_shot_completion_rate = (not_shot_plays['passResult'] == 'C').mean()

print(f"Shot Play Completion %: {shot_completion_rate:.3f}")
print(f"Non-Shot Completion %: {non_shot_completion_rate:.3f}")

print(f"\nAvg EPA gained om shot plays (2nd & midfield): {avg_EPA_gained_shot}")
print(f"Avg EPA gained on regular passes (2nd & midfield): {avg_EPA_gained_not_shot}")

# Perform T-test for EPA comparison on 2nd down at midfield
t_stat, p_val = ttest_ind(
    shot_plays['expectedPointsAdded'].dropna(),
    not_shot_plays['expectedPointsAdded'].dropna(),
    equal_var=False
)

print(f't stat: {np.round(t_stat, 4)}')
print(f'p_val: {np.round(p_val, 4)}')

shot_completion_rate = (shot_plays['passResult'] == 'C').mean()
non_shot_completion_rate = (not_shot_plays['passResult'] == 'C').mean()

print(f"Shot Play Completion %: {shot_completion_rate:.3f}")
print(f"Non-Shot Completion %: {non_shot_completion_rate:.3f}")

# --- Summary so far (Textual Comment) ---
'''
We see from the avg EPA after shot plays is higher compared to normal drop back passes in all of the situations 
we tested. This shows that shot plays are better than normal pass plays. Once again this isn't a surprising result. 
The next step is to find the spots and dnd on the field when the EPA is considerably higher compared to normal passes. 
Furthermore, an offensecan't run shot plays on every play, thus we will search for an optimal number of times a shot 
play should run by looking at the delta of EPA and end result of a game to see if it's possible
'''

# --- Group by Game to Find Shot Play Frequency ---
# Filter down to dropback plays with valid EPA, yards to go, and within midfield
epa_data = filtered_plays[
    (filtered_plays['isDropback'] == True) &
    (filtered_plays['expectedPointsAdded'].notnull()) &
    (filtered_plays['yardsToGo'].notnull()) &
    (filtered_plays["absoluteYardlineNumber"].between(40, 60))
    ]

# Group by yardsToGo and shot play status, then calculate mean EPA
epa_grouped = epa_data.groupby(["yardsToGo", "is_shot_play"])["expectedPointsAdded"].mean().reset_index()
epa_grouped = epa_grouped[epa_grouped['yardsToGo'] < 11]  # Filter for yardsToGo less than 11

# Pivot to get two columns: shot and non-shot EPA per yardsToGo
epa_pivot = epa_grouped.pivot(index="yardsToGo", columns="is_shot_play", values="expectedPointsAdded")
epa_pivot.columns = ["Non-Shot Play", "Shot Play"]  # Rename columns for clarity
epa_pivot = epa_pivot.reset_index()

# Plot EPA by Yards to Go for Shot vs. Non-Shot Plays (Midfield)
plt.figure(figsize=(10, 6))
sns.lineplot(data=epa_pivot, x="yardsToGo", y="Shot Play", label="Shot Plays")
sns.lineplot(data=epa_pivot, x="yardsToGo", y="Non-Shot Play", label="Non-Shot Plays")
plt.title("EPA by Yards to Go: Shot vs. Non-Shot Plays (Midfield)")
plt.xlabel("Yards to First Down")
plt.ylabel("Expected Points Added (EPA)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- T-Test Setup (EPA for Shot vs. Non-Shot, Midfield < 11 Yards to Go) ---

# Filter again to get raw EPA values for shot and non-shot plays under specific conditions
shot_epa = epa_data[
    (epa_data["is_shot_play"] == True) & (epa_data["yardsToGo"] < 11)
    ]["expectedPointsAdded"]

non_shot_epa = epa_data[
    (epa_data["is_shot_play"] == False) & (epa_data["yardsToGo"] < 11)
    ]["expectedPointsAdded"]

# --- Perform T-Test ---
# Perform an independent t-test to compare EPA between shot and non-shot plays
t_stat, p_val = ttest_ind(shot_epa, non_shot_epa, alternative='greater', equal_var=False)

print(f't stat: {np.round(t_stat, 4)}')
print(f'p_val: {np.round(p_val, 4)}')

# Load game metadata
games = pd.read_csv("Data/games.csv")

# Reshape game data into long format: one row per team per game
home = games[["gameId", "homeTeamAbbr", "homeFinalScore"]].copy()
home.columns = ["gameId", "team", "score"]
home["is_home"] = 1  # Indicator for home team

away = games[["gameId", "visitorTeamAbbr", "visitorFinalScore"]].copy()
away.columns = ["gameId", "team", "score"]
away["is_home"] = 0  # Indicator for away team

# Combine home and away teams into one long dataframe
scores = pd.concat([home, away], ignore_index=True)

# Label win/loss for each team based on their final score in the game
scores["game_result"] = scores.groupby("gameId")["score"].transform(lambda x: x == x.max()).astype(int)

# Count the number of shot plays per possession team per game
shot_counts = filtered_plays[
    filtered_plays["isDropback"] & filtered_plays["is_shot_play"]
    ].groupby(["gameId", "possessionTeam"]).size().reset_index(name="num_shot_plays")

# Rename possessionTeam for merge compatibility
shot_counts = shot_counts.rename(columns={"possessionTeam": "team"})

# Merge shot play counts with team outcomes (win/loss)
team_game_outcomes = pd.merge(shot_counts, scores, on=["gameId", "team"])

# Boxplot of shot play counts by game result (win/loss)
plt.figure(figsize=(8, 5))
sns.boxplot(data=team_game_outcomes, x="game_result", y="num_shot_plays")
plt.title("Shot Plays per Game: Winners vs. Losers")
plt.xlabel("Game Result (0 = Loss, 1 = Win)")
plt.ylabel("Number of Shot Plays")
plt.grid(True)
plt.show()

# Perform T-test to compare number of shot plays between winning and losing teams
winners = team_game_outcomes[team_game_outcomes["game_result"] == 1]["num_shot_plays"]
losers = team_game_outcomes[team_game_outcomes["game_result"] == 0]["num_shot_plays"]

t_stat, p_val = ttest_ind(winners, losers, equal_var=False)
print(f't stat: {np.round(t_stat, 4)}')
print(f'p_val: {np.round(p_val, 4)}')

# Sort filtered_plays by gameId and playId for chronological ordering
filtered_plays = filtered_plays.sort_values(by=["gameId", "playId"])

# Reset index after sorting
filtered_plays = filtered_plays.reset_index(drop=True)

# Use a cumulative counter to estimate drive breaks based on change in possession or gameId
filtered_plays["possession_change"] = (filtered_plays["possessionTeam"] !=
                                       filtered_plays["possessionTeam"].shift(1)) | (filtered_plays["gameId"] !=
                                                                                     filtered_plays["gameId"].shift(1))

# Create a drive ID approximation
filtered_plays["drive_group"] = filtered_plays["possession_change"].cumsum()

# Assign play number within each approximated drive
filtered_plays["play_num_in_drive"] = (
        filtered_plays.groupby("drive_group").cumcount() + 1
)

# Filter for shot plays with valid EPA to analyze their position in drives
shot_drive_seq = filtered_plays[
    (filtered_plays["is_shot_play"] == True) &
    (filtered_plays["expectedPointsAdded"].notnull())
    ]

# Group by play number within drive and get average EPA and count
shot_epa_by_play_num = (shot_drive_seq.groupby("play_num_in_drive")["expectedPointsAdded"]
                        .agg(["mean", "count"]).reset_index())

# Plot Average EPA of Shot Plays by Position in Drive
plt.figure(figsize=(10, 6))
sns.lineplot(data=shot_epa_by_play_num, x="play_num_in_drive", y="mean")
plt.title("Average EPA of Shot Plays by Position in Drive")
plt.xlabel("Play Number in Drive")
plt.ylabel("Average EPA")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- EPA by Down and Field Bin (Midfield Only) ---

# Bin field position into 10-yard increments
filtered_plays["field_bin"] = (filtered_plays["absoluteYardlineNumber"] // 10) * 10
# Filter to midfield plays only
filtered_plays = filtered_plays[filtered_plays['field_bin'].between(40, 60)]
# Filter for shot plays with valid EPA
conditions = filtered_plays[filtered_plays["is_shot_play"] & filtered_plays["expectedPointsAdded"].notnull()]

# Group by down and field bin, then calculate mean EPA
summary = conditions.groupby(["down", "field_bin"])["expectedPointsAdded"].mean().reset_index()
summary = summary[summary['down'] < 4]  # Limit to 1st, 2nd, 3rd downs
print(summary)

# Plot comparison of EPA by field position and down
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x='down', y='expectedPointsAdded', hue='field_bin')
plt.xlabel('Down')
plt.ylabel('Expected Points Added')
plt.title('Comparison of EPA by Field Position and Down')
plt.legend(title='Absolute Field Position')
plt.tight_layout()
plt.show()

# Analyze 3rd down at 50-yard line specific instances
third_fifty = conditions[(conditions['field_bin'] == 50) & (conditions['down'] == 3)]
print(f"\nNumber of instances of third down @ midfield: {len(third_fifty)}")

avg_yards_to_go_503 = third_fifty['yardsToGo'].mean()
print(f'\nAvg yards to go on third down @ 50 yard line: {np.round(avg_yards_to_go_503, 4)}')

# --- Causal Inference Question & Method (Textual Comment) ---
'''
From the plot of average EPA by down, we observe that offenses generate the highest value—measured by expected points
added—on 3rd down. This naturally raises the question: Why is 3rd down so effective for shot plays?

One hypothesis is that the EPA boost comes from completing a deep pass that not only advances field position but also
resets the downs. However, this logic should apply to 1st and 2nd down shot plays as well.
So why does 3rd down stand out?

To explore this, I will estimate the **causal effect** of calling a shot play on 3rd down at midfield. Specifically,
I ask:

"Do shot plays on 3rd down at midfield cause better outcomes (higher EPA), or are they simply chosen in favorable
situations?"

To answer this, I’ll use **propensity score weighting** to control for confounding variables like yards-to-go, motion,
and play-action. This approach allows me to simulate a randomized trial using observational data, isolating the true
effect of shot plays in this scenario.
'''

# --- Add Motion Indicator to Filtered Plays for Propensity Modeling ---
# Load player-level motion data
player_plays = pd.read_csv("Data/player_play.csv")

# Create motion indicator using any of the 3 relevant flags (inMotionAtBallSnap, shiftSinceLineset, motionSinceLineset)
player_plays["motion_indicator"] = (
        player_plays["inMotionAtBallSnap"].fillna(False) |
        player_plays["shiftSinceLineset"].fillna(False) |
        player_plays["motionSinceLineset"].fillna(False)
)

# Reduce to play-level motion (True if any player was in motion on the play)
motion_flags = player_plays.groupby(["gameId", "playId"])["motion_indicator"].any().reset_index()

# Merge motion flag with filtered_plays DataFrame
filtered_plays = filtered_plays.merge(motion_flags, on=["gameId", "playId"], how="left")
filtered_plays["motion_indicator"] = filtered_plays["motion_indicator"].fillna(False).astype(
    int)  # Convert boolean to int (0/1)

# --- Propensity Score Weighting for Causal Effect Analysis ---

# Filter to 3rd down midfield plays with valid EPA for causal analysis
third_mid = filtered_plays[
    (filtered_plays["down"] == 3) &
    (filtered_plays["absoluteYardlineNumber"].between(40, 60)) &
    (filtered_plays["expectedPointsAdded"].notnull())
    ].copy()

# Define features (X) to predict treatment (shot play) - these are potential confounders
features = ["yardsToGo", "playAction", "motion_indicator"]
X = third_mid[features].copy()
X["playAction"] = X["playAction"].astype(int)
X["motion_indicator"] = X["motion_indicator"].astype(int)

# Define treatment (T): whether the play was a shot play (1) or not (0)
T = third_mid["is_shot_play"].astype(int)

# Step 1: Fit Propensity Model (Logistic Regression)
model = LogisticRegression(solver='liblinear', random_state=42)  # Using liblinear for small datasets
model.fit(X, T)
third_mid["propensity"] = model.predict_proba(X)[:, 1]  # Get probability of being a shot play

# Step 2: Calculate IPTW (Inverse Probability of Treatment Weights)
# Weight for treated (shot) plays: 1 / propensity
# Weight for control (non-shot) plays: 1 / (1 - propensity)
third_mid["weight"] = np.where(
    third_mid["is_shot_play"] == 1,
    1 / third_mid["propensity"],
    1 / (1 - third_mid["propensity"])
)

# Step 3: Weighted EPA comparison to estimate causal effect
treated = third_mid[third_mid["is_shot_play"] == 1]
control = third_mid[third_mid["is_shot_play"] == 0]

treated_EPA = np.average(treated["expectedPointsAdded"], weights=treated["weight"])
control_EPA = np.average(control["expectedPointsAdded"], weights=control["weight"])

print(f"Weighted EPA (Shot): {treated_EPA:.3f}")
print(f"Weighted EPA (Non-Shot): {control_EPA:.3f}")
print(f"Delta (Treatment Effect): {treated_EPA - control_EPA:.3f}")

# --- Causal Inference Results Summary (Textual Comment) ---
'''
To isolate the true effect of shot plays on 3rd down at midfield, I built a causal inference model using
Inverse Probability of Treatment Weighting (IPTW). The goal was to determine whether the high EPA observed
on these plays is due to favorable conditions or the shot play itself.

I first created a logistic regression model (propensity score model) to estimate the probability of calling
a shot play based on key confounding variables: yards to go, play-action, and motion. This generated a
"propensity score" for each play, indicating how likely it was to be a shot given its situation.

Then, using IPTW, I weighted plays inversely by their probability of receiving the treatment (shot play) to
simulate a randomized experiment.

The results show a large treatment effect:
- Weighted EPA (Shot Plays): 0.995
- Weighted EPA (Non-Shot Plays): -0.093
- **Estimated Causal Effect (Delta): +1.088 EPA**

This suggests that calling a shot play on 3rd down at midfield is not just correlated with better outcomes —
it causes significantly higher offensive value, even after adjusting for situational context.
'''

# --- Completion Percentage for 3rd Down Midfield Shot vs Non-Shot Plays ---
# Filter for 3rd down midfield plays with valid EPA
third_mid = filtered_plays[
    (filtered_plays["down"] == 3) &
    (filtered_plays["absoluteYardlineNumber"].between(40, 60)) &
    (filtered_plays["expectedPointsAdded"].notnull())
    ].copy()

# Separate into shot and non-shot plays
shot_plays_3rd = third_mid[third_mid["is_shot_play"] == True]
non_shot_plays_3rd = third_mid[third_mid["is_shot_play"] == False]

# Calculate and print completion percentages
shot_completion_pct = (shot_plays_3rd["passResult"] == "C").mean()
non_shot_completion_pct = (non_shot_plays_3rd["passResult"] == "C").mean()

print(f"\nShot Play Completion % (3rd @ Midfield): {shot_completion_pct:.3f}")
print(f"Non-Shot Completion % (3rd @ Midfield): {non_shot_completion_pct:.3f}")
