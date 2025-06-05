import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# --- Data Loading & Initial Preprocessing ---
plays = pd.read_csv("Data/plays.csv")

# Filter out outlier and garbage time plays
filtered_plays = plays.copy()
filtered_plays = filtered_plays[filtered_plays["quarter"] != 5]  # Remove overtime plays
filtered_plays = filtered_plays[filtered_plays["playNullifiedByPenalty"] == "N"]  # Remove plays nullified by penalty

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

# --- Classify Play Types (RPO vs. Regular, Pass vs. Run) ---
filtered_plays['is_RPO'] = filtered_plays['pff_runPassOption'] == 1  # Flag for RPO plays
filtered_plays['is_pass'] = filtered_plays['isDropback'] == True  # Flag for pass plays (dropbacks)
filtered_plays['is_run'] = filtered_plays['isDropback'] == False  # Flag for run plays

# Classify play type based on RPO and pass/run flags
filtered_plays['play_type'] = filtered_plays.apply(
    lambda row: 'RPO Pass' if row['is_RPO'] and row['is_pass']
    else 'RPO Run' if row['is_RPO'] and row['is_run']
    else 'Regular Pass' if not row['is_RPO'] and row['is_pass']
    else 'Regular Run', axis=1
)

# --- Define Field Zones for Analysis ---
# Plays starting in the "Starting Zone" (own 25 to own 40-yard line)
zone_starting = filtered_plays[
    (filtered_plays['absoluteYardlineNumber'] >= 25) &
    (filtered_plays['absoluteYardlineNumber'] <= 40)
    ]

# Plays starting in the "Scoring Zone" (opponent 40 to opponent 10-yard line, excluding red zone)
zone_redzone_prep = filtered_plays[
    (filtered_plays['absoluteYardlineNumber'] >= 60) &
    (filtered_plays['absoluteYardlineNumber'] <= 90)
    ]


# --- Plotting Function (Average Yards Gained per Play Type by Down and Zone) ---
def plot_zone_analysis(data, title_suffix):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
    down_labels = ['1st Down', '2nd Down', '3rd Down']
    colors = {
        'RPO Run': 'powderblue',
        'RPO Pass': 'aquamarine',
        'Regular Run': 'tan',
        'Regular Pass': 'hotpink'
    }

    for i, down in enumerate([1, 2, 3]):
        ax = axes[i]
        subset = data[data['down'] == down]

        # Group by play_type and calculate average yards gained
        means = (
            subset.groupby('play_type')['yardsGained']
            .mean()
            .reindex(colors.keys())
        )

        bars = ax.bar(means.index, means.values, color=[colors[pt] for pt in means.index])

        # Add data labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{yval:.1f}", ha='center', fontsize=9)

        ax.set_title(down_labels[i], fontsize=12, weight='bold')
        ax.set_ylabel("Avg Yards Gained" if i == 0 else "")
        ax.set_xticklabels(means.index, rotation=45)
        ax.set_ylim(0, 10)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    fig.suptitle(f"Avg Yards Gained — {title_suffix}", fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# --- Generate Plots for Field Zones ---

# Plot for the "Starting Zone"
plot_zone_analysis(zone_starting, "Starting Zone (25–40 Yard Line)")
# Plot for the "Scoring Zone"
plot_zone_analysis(zone_redzone_prep, "Scoring Zone (60–90 Yard Line)")

# --- Situational Context Analysis of RPO Usage ---

# - RPO Rate by Quarter -
rpo_by_quarter = filtered_plays.groupby(['quarter', 'is_RPO']).size().unstack().fillna(0)
rpo_by_quarter['RPO Rate'] = rpo_by_quarter[True] / (rpo_by_quarter[True] + rpo_by_quarter[False])
print("\nRPO Usage by Quarter:")
print(rpo_by_quarter)


# - RPO Rate by Score State (Leading, Tied, Trailing) -
def score_state(row):
    if row['score_difference'] > 0:
        return 'Leading'
    elif row['score_difference'] < 0:
        return 'Trailing'
    else:
        return 'Tied'


filtered_plays['score_state'] = filtered_plays.apply(score_state, axis=1)

rpo_by_score = filtered_plays.groupby(['score_state', 'is_RPO']).size().unstack().fillna(0)
rpo_by_score['RPO Rate'] = rpo_by_score[True] / (rpo_by_score[True] + rpo_by_score[False])
print("\nRPO Usage by Score State:")
print(rpo_by_score)


# - RPO Rate by Yards to Go Range -
def ytg_bucket(yards):
    if yards <= 3:
        return 'Short (1–3)'
    elif yards <= 6:
        return 'Medium (4–6)'
    else:
        return 'Long (7–10)'


filtered_plays['ytg_range'] = filtered_plays['yardsToGo'].apply(ytg_bucket)

rpo_by_ytg = filtered_plays.groupby(['ytg_range', 'is_RPO']).size().unstack().fillna(0)
rpo_by_ytg['RPO Rate'] = rpo_by_ytg[True] / (rpo_by_ytg[True] + rpo_by_ytg[False])
print("\nRPO Usage by Yards to Go:")
print(rpo_by_ytg)

# - RPO Rate by Down -
rpo_by_down = filtered_plays.groupby(['down', 'is_RPO']).size().unstack().fillna(0)
rpo_by_down['RPO Rate'] = rpo_by_down[True] / (rpo_by_down[True] + rpo_by_down[False])
print("\nRPO Usage by Down:")
print(rpo_by_down)


# - RPO Rate by Win Probability Bin -
def wp_bin(prob):
    if prob < 0.3:
        return '<30%'
    elif prob < 0.7:
        return '30–70%'
    else:
        return '>70%'


filtered_plays['wp_range'] = filtered_plays['preSnapHomeTeamWinProbability'].apply(
    wp_bin)

rpo_by_wp = filtered_plays.groupby(['wp_range', 'is_RPO']).size().unstack().fillna(0)
rpo_by_wp['RPO Rate'] = rpo_by_wp[True] / (rpo_by_wp[True] + rpo_by_wp[False])
print("\nRPO Usage by Win Probability Range:")
print(rpo_by_wp)

# --- Team RPO Usage vs. Wins Analysis ---

# Team win totals from the image (2022 regular season data)
team_wins_2022 = {
    'KC': 14, 'PHI': 14, 'BUF': 13, 'MIN': 13, 'SF': 13, 'CIN': 12, 'DAL': 12,
    'BAL': 10, 'LAC': 10, 'NYG': 9, 'DET': 9, 'JAX': 9, 'MIA': 9, 'PIT': 9, 'SEA': 9,
    'WAS': 8, 'GB': 8, 'NE': 8, 'TB': 8, 'CAR': 7, 'CLE': 7, 'NO': 7, 'ATL': 7,
    'TEN': 7, 'NYJ': 7, 'LV': 6, 'DEN': 5, 'LAR': 5, 'IND': 4, 'ARI': 4, 'HOU': 3, 'CHI': 3
}

# - Recompute RPO Rate by Team -
rpo_by_team = filtered_plays.groupby(['possessionTeam', 'is_RPO']).size().unstack().fillna(0)
rpo_by_team.columns = ['Non-RPO Plays', 'RPO Plays']
rpo_by_team['Total Plays'] = rpo_by_team['Non-RPO Plays'] + rpo_by_team['RPO Plays']
rpo_by_team['RPO Rate'] = rpo_by_team['RPO Plays'] / rpo_by_team['Total Plays']

# Define success as gaining at least yardsToGo
filtered_plays['successful'] = filtered_plays['yardsGained'] >= filtered_plays['yardsToGo']

# Compute RPO Success Rate per team
rpo_success = (
    filtered_plays[filtered_plays['is_RPO']]
    .groupby('possessionTeam')['successful']
    .mean()
    .rename('RPO Success Rate')
)

# Combine RPO usage and success rates into a single DataFrame
team_rpo_profile = rpo_by_team[['RPO Plays', 'Total Plays', 'RPO Rate']]
team_rpo_profile = team_rpo_profile.join(rpo_success)

# Add 2022 season win totals to the team RPO profile
team_rpo_profile['Wins'] = team_rpo_profile.index.map(team_wins_2022)

# - Scatter Plot: RPO Usage Rate vs Team Wins -
plt.figure(figsize=(10, 6))
sns.scatterplot(data=team_rpo_profile, x='RPO Rate', y='Wins')
sns.regplot(data=team_rpo_profile, x='RPO Rate', y='Wins', scatter=False, ci=None, color='gray')  # Add regression line

# Annotate each point with team names
for team, row in team_rpo_profile.iterrows():
    plt.text(row['RPO Rate'] + 0.001, row['Wins'], team, fontsize=8)

plt.title("🏈 RPO Usage Rate vs Team Wins (2022)", fontsize=14, weight='bold')
plt.xlabel("RPO Usage Rate")
plt.ylabel("Team Wins")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Calculate and print correlation between RPO Usage Rate and Wins
corr = team_rpo_profile[['RPO Rate', 'Wins']].corr().iloc[0, 1]
print(f"Correlation between RPO Usage Rate and Wins: {corr:.3f}")

# - Scatter Plot: RPO Success Rate vs Team Wins -
plt.figure(figsize=(10, 6))
sns.scatterplot(data=team_rpo_profile, x='RPO Success Rate', y='Wins')

# Label each point with team names
for team, row in team_rpo_profile.iterrows():
    plt.text(row['RPO Success Rate'] + 0.001, row['Wins'], team, fontsize=8)

plt.title("RPO Success Rate vs Team Wins (2022)", fontsize=14, weight='bold')
plt.xlabel("RPO Success Rate")
plt.ylabel("Team Wins")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print correlation matrix for RPO Rate, RPO Success Rate, and Wins
print(team_rpo_profile[['RPO Rate', 'RPO Success Rate', 'Wins']].corr())
