# NFL Picks: ML Predictions + Sportsbook Line Comparison

This project builds an end-to-end supervised machine learning pipeline to predict NFL game outcomes (margin + total) and then compare predictions against sportsbook lines (spread, moneyline, total) to compute a simple edge and output a weekly “Play / No Play” spreadsheet.

We treat this as a decision-support tool. The whole point is to show what the model expects, what the sportsbook implies, and where the model disagrees enough to flag potential plays.

---

## Repo Structure
- `build_dataset.py`  
  Loads play-by-play data (2021–2025) and produces a game-level/team-level dataset (one row per team per game). Saves `game_level_data.csv`.

- `make_features.py`  
  Builds rolling features (recent form), matchup difference features, and context features. Produces the modeling dataset `model_dataset.csv`.

- `train_model.py`  
  Trains regression models to predict:
  - margin (home - away)
  - total points (home + away)  
  Selects the best model using validation MAE, saves the trained pipeline + feature list (artifacts).

- `make_picks.py`  
  Pulls odds from The Odds API, builds the current-week feature rows, predicts expected scores, and generates:
  - `weekly_picks.csv`
  - `weekly_picks.xlsx`  
  Also prints a simple winner accuracy number for the latest season.

---

## Setup
### 1) Install dependencies
```Windows Powershell
pip install pandas numpy scikit-learn requests openpyxl nfl_data_py
```
### 2) Set ODDs Api key
```Windows Powershell
setx ODDS_API_KEY "YOUR_KEY_HERE"
```
## Run Model

Step 1: Build game-level dataset
``` Windows Powershell
python build_dataset.py
```
Step 2: Feature Engineering
``` Windows Powershell
python make_features.py
```
Step 3: Train + Backtest
``` Windows Powershell
python train_model.py
```
Step 4: Generate Weekly Picks
``` Windows Powershell
python make_picks.py
```
---
## Lines + implied probability

Sportsbook odds imply a win probability. We compare the Model Home Win % (what the model suggests) vs the Implied Home Win % / Implied Away Win % (what the sportsbook suggests)

## Limitations
NFL outcomes are high variance and even a good model will not be 100% correct. Sportsbook lines are also a strong baseline and it is very difficult to beat them consistently.

---
## Screenshots of Model Predictions (NFL Week 14 and Week 15)
**Week 14**

![Week1 4 Predictions](images/Week14.png)


**Week 15**

![Week 15 Predictions](images/Week15.png)

