import nfl_data_py as nfl
import pandas as pd
import numpy as np

SEASONS = [2021, 2022, 2023, 2024, 2025]
OUTPUT_PATH = "game_level_data.csv"

print("Loading play-by-play...")
pbp = nfl.import_pbp_data(SEASONS)

group_keys = ["season", "week", "game_id", "home_team", "away_team"]

#home and away scores
if ("home_score" in pbp.columns) and ("away_score" in pbp.columns):
    games = (
    pbp.groupby(group_keys)
    .agg(
        game_date=("game_date", "max"),
        home_points=("home_score", "max"),
        away_points=("away_score", "max"),
    )
    .reset_index()
)

#Normalize game_date to datetime if possible
games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

team_games = []
for _, row in games.iterrows():
    for side in ["home", "away"]:
        team = row[f"{side}_team"]
        opp = row["away_team"] if side == "home" else row["home_team"]
        pts_scored = row[f"{side}_points"]
        pts_allowed = row["away_points"] if side == "home" else row["home_points"]

        team_games.append(
            {
                "season": row["season"],
                "week": row["week"],
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "team": team,
                "opponent": opp,
                "is_home": 1 if side == "home" else 0,
                "points_scored": pts_scored,
                "points_allowed": pts_allowed,
            }
        )

df = pd.DataFrame(team_games)

#Helpers
df["margin"] = df["points_scored"] - df["points_allowed"]
df["win"] = (df["margin"] > 0).astype(int)
df["game_total_points"] = df["points_scored"] + df["points_allowed"] #same for both rows per game

#Use only actual offensive plays run or pass
pbp_off = pbp.copy()
pbp_off = pbp_off[pbp_off["posteam"].notna()].copy()

if "play_type" in pbp_off.columns:
    pbp_off = pbp_off[pbp_off["play_type"].isin(["run", "pass"])].copy()

#Normalize numeric columns that have NaNs
for c in ["yards_gained", "passing_yards", "rushing_yards", "epa"]:
    if c in pbp_off.columns:
        pbp_off[c] = pd.to_numeric(pbp_off[c], errors="coerce").fillna(0.0)

#Turnovers committed = interceptions + fumbles lost
if "interception" in pbp_off.columns:
    pbp_off["interception"] = pd.to_numeric(pbp_off["interception"], errors="coerce").fillna(0.0)
else:
    pbp_off["interception"] = 0.0

if "fumble_lost" in pbp_off.columns:
    pbp_off["fumble_lost"] = pd.to_numeric(pbp_off["fumble_lost"], errors="coerce").fillna(0.0)
else:
    pbp_off["fumble_lost"] = 0.0

pbp_off["turnovers_committed"] = pbp_off["interception"] + pbp_off["fumble_lost"]

#Num of plays run
stats = (
    pbp_off.groupby(["game_id", "posteam"])
    .agg(
        yards=("yards_gained", "sum"),
        plays=("yards_gained", "size"),
        passing_yards=("passing_yards", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        turnovers_committed=("turnovers_committed", "sum"),
        epa=("epa", "sum"),
    )
    .reset_index()
)

#Rates
stats["yards_per_play"] = np.where(stats["plays"] > 0, stats["yards"] / stats["plays"], np.nan)
stats["epa_per_play"] = np.where(stats["plays"] > 0, stats["epa"] / stats["plays"], np.nan)


starting_qb = None
if "passer_player_name" in pbp.columns:
    qb_pbp = pbp.copy()
    qb_pbp = qb_pbp[qb_pbp["posteam"].notna()].copy()

    #Dropback definition is pass attempt or sack
    mask = pd.Series(True, index=qb_pbp.index)

    if "pass_attempt" in qb_pbp.columns:
        mask &= (qb_pbp["pass_attempt"].fillna(0).astype(int) == 1)
    else:
        #If there's no pass attempts then use passer_player_name
        mask &= qb_pbp["passer_player_name"].notna()

    if "sack" in qb_pbp.columns:
        #Sacks included as dropbacks
        mask = mask | (qb_pbp["sack"].fillna(0).astype(int) == 1)

    qb_pbp = qb_pbp[mask & qb_pbp["passer_player_name"].notna()].copy()

    if not qb_pbp.empty:
        qb_counts = (
            qb_pbp.groupby(["game_id", "posteam", "passer_player_name"])
            .size()
            .reset_index(name="n_dropbacks")
        )
        idx = qb_counts.groupby(["game_id", "posteam"])["n_dropbacks"].idxmax()
        starting_qb = qb_counts.loc[idx, ["game_id", "posteam", "passer_player_name"]].copy()
        starting_qb = starting_qb.rename(columns={"passer_player_name": "starting_qb"})

#Merge QB into stats if we found it
if starting_qb is not None and not starting_qb.empty:
    stats = stats.merge(starting_qb, on=["game_id", "posteam"], how="left")
else:
    stats["starting_qb"] = np.nan

#Merging offensive stats
df = df.merge(
    stats,
    left_on=["game_id", "team"],
    right_on=["game_id", "posteam"],
    how="left",
)
df = df.drop(columns=["posteam"])

#opponent stats merged
opp_cols = [
    "game_id",
    "team",
    "yards",
    "plays",
    "passing_yards",
    "rushing_yards",
    "turnovers_committed",
    "epa",
    "yards_per_play",
    "epa_per_play",
]

opp = df[opp_cols].copy()
opp = opp.rename(
    columns={
        "team": "opponent",
        "yards": "yards_allowed",
        "plays": "plays_defended",
        "passing_yards": "passing_yards_allowed",
        "rushing_yards": "rushing_yards_allowed",
        "turnovers_committed": "turnovers_forced",
        "epa": "epa_allowed",
        "yards_per_play": "yards_per_play_allowed",
        "epa_per_play": "epa_per_play_allowed",
    }
)

df = df.merge(opp, on=["game_id", "opponent"], how="left")

#Turnover differential
df["turnover_diff"] = df["turnovers_forced"] - df["turnovers_committed"]

#Column cleanup
num_cols = [
    "points_scored", "points_allowed", "margin", "win", "game_total_points",
    "yards", "plays", "passing_yards", "rushing_yards", "turnovers_committed", "epa",
    "yards_per_play", "epa_per_play",
    "yards_allowed", "plays_defended", "passing_yards_allowed", "rushing_yards_allowed",
    "turnovers_forced", "epa_allowed", "yards_per_play_allowed", "epa_per_play_allowed",
    "turnover_diff",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH} ({len(df)} rows)")