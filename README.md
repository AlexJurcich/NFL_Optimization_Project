# NFL_Optimization_Project
Leveraging data science to identify optimal play-calling strategies and enhance offensive efficiency in the NFL.
# NFL Offensive Optimization Project

## Description
Leveraging data science to identify optimal play-calling strategies and enhance offensive efficiency in the NFL.

## Table of Contents
1.  [Project Overview & Motivation](#project-overview--motivation)
2.  [Methodology](#methodology)
3.  [Key Findings](#key-findings)
    * [1. Third Down Pass Conversion Rate](#1-third-down-pass-conversion-rate)
    * [2. Run Plays with Motion vs. Without Motion](#2-run-plays-with-motion-vs-without-motion)
    * [3. Shot Plays](#3-shot-plays)
    * [4. Gap vs. Zone Run Plays](#4-gap-vs-zone-run-plays)
    * [5. Optimal Play Selection on 2nd and 6 at Midfield](#5-optimal-play-selection-on-2nd-and-6-at-midfield)
    * [6. 4th and 1](#6-4th-and-1)
4.  [Limitations & Future Work](#limitations--future-work)
5.  [Project Structure](#project-structure)
6.  [Contact](#contact)

---

## Project Overview & Motivation
This project challenges long-standing assumptions in NFL offensive play-calling by combining firsthand football experience with data-driven analysis. As a former collegiate football player at the University of Wisconsin-Eau Claire and a dedicated fan of the sport, I have often shared the reaction of many viewers and players who question the rationale behind certain play calls. The ultimate aim is to offer coaches, analysts, and team personnel clear, data-backed guidance on how to maximize offensive efficiency and scoring, thus contributing to a broader evolution in how NFL offenses operate.

## Methodology
We started by collecting and cleaning detailed play-by-play data from the NFL Big Data Bowl on Kaggle. We defined key performance metrics. We applied formal hypothesis tests to confirm or dispel conventional wisdom on offensive decision-making. Looking forward, our ultimate aim is to create predictive models that quantify the value of adopting more optimal situational plays.

## Key Findings

### 1. Third Down Pass Conversion Rate
This analysis determined that the likelihood of converting a third down is significantly higher when a pass is thrown beyond the line to gain compared to when it is thrown short of it. Passes targeting receivers past the sticks are twice as effective in securing a first down.

### 2. Run Plays with Motion vs. Without Motion
This analysis evaluated whether run plays with motion are more effective than plays without motion. When motion was used on 1st and 10 inside the opponent's 40-yard line, teams gained 5.7 yards on average, compared to 3.9 yards without motion, a difference of nearly 2 full yards.

### 3. Shot Plays
This analysis aimed to optimize when offenses should attempt shot plays. On 3rd down at midfield, shot plays gained an average of 12.0 yards, compared to just 6.2 yards for non-shot plays. A statistical test revealed the difference in EPA was statistically significant (p<0.001).

### 4. Gap vs. Zone Run Plays
This analysis explores whether Gap or Zone run schemes are more effective across all downs in the NFL. Gap runs gained 5.16 yards per carry, compared to Zone runs at 4.34 yards per carry. On 1st down, Gap success rate was 51.1%, while Zone success rate was 45.8%.

### 5. Optimal Play Selection on 2nd and 6 at Midfield
This analysis aimed to evaluate the effectiveness of play action compared to straight dropbacks and run plays on 2nd down with 6 yards to go. At midfield (40-60 yardline), play action produced significantly higher EPA than straight dropbacks ($p=0.0311$).

### 6. 4th and 1
This analysis investigates optimal decision-making on 4th and 1. QB sneaks had the highest conversion rate at 87.1%, far surpassing traditional runs (56.9%) and pass plays (51.2%).

## Limitations & Future Work
One key limitation of this analysis is the absence of explicit field goal and punt attempt data in the dataset. Without complete play classification or special teams outcome data, we are unable to quantify the expected points or win probability associated with those alternative choices. Future work incorporating full play-by-play datasets - including kicking and punting plays - would allow for a more comprehensive cost-benefit comparison between fourth-down options. Opportunities for further analysis include comparing decision-making by coach/team tendencies, assessing defensive coverages most vulnerable to shot plays, and examining coach-level tendencies.

## Project Structure
* `README.md`: This file provides an overview of the project.
* `NFL_Offensive_Project_Slides.pptx`: The non-technical presentation slides.
* `Situations`: Contains the Python scripts used for data loading, preprocessing, analysis, and plot generation.
    * `2nd&6.py`
    * `Motion_vs_NoMotion.py`
    * `RPO_vs_Reg.py` 
    * `Shot_plays.py`
    * `Zone_vs_Gap.py`
    * `4th_and_1.py`
    * `4th_beyond_1`
* `Data` can be found here: https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data
* `NFL_Optimization_Complete.pdf/`: Full analysis report (PDF).

## Contact
Alex Jurcich
MS Data Science Student, Indiana University
jurcichalex@gmail.com |
https://www.linkedin.com/in/alexjurcich/


