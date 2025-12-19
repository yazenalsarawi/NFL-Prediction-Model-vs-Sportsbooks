import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

ROLL_N = 5
INPUT_PATH = "game_level_data.csv"
OUTPUT_PATH = "model_dataset.csv"

#Team coords and divisions
TEAM_COORDS = {
    "ARI": (33.5277, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2779, -76.6227),
    "BUF": (42.7738, -78.7868), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LAC": (33.9535, -118.3392), "LAR": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830), "MIA": (25.9580, -80.2389), "MIN": (44.9738, -93.2581),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645),
}

TEAM_DIVISION = {
    "BUF": "AFC_E", "MIA": "AFC_E", "NE": "AFC_E", "NYJ": "AFC_E",
    "BAL": "AFC_N", "CIN": "AFC_N", "CLE": "AFC_N", "PIT": "AFC_N",
    "HOU": "AFC_S", "IND": "AFC_S", "JAX": "AFC_S", "TEN": "AFC_S",
    "DEN": "AFC_W", "KC": "AFC_W", "LAC": "AFC_W", "LV": "AFC_W",
    "DAL": "NFC_E", "NYG": "NFC_E", "PHI": "NFC_E", "WAS": "NFC_E",
    "CHI": "NFC_N", "DET": "NFC_N", "GB": "NFC_N", "MIN": "NFC_N",
    "ATL": "NFC_S", "CAR": "NFC_S", "NO": "NFC_S", "TB": "NFC_S",
    "ARI": "NFC_W", "LAR": "NFC_W", "SF": "NFC_W", "SEA": "NFC_W",
}

#Helpers
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def travel_miles(home_abbr, away_abbr):
    if home_abbr not in TEAM_COORDS or away_abbr not in TEAM_COORDS:
        return np.nan
    h_lat, h_lon = TEAM_COORDS[home_abbr]
    a_lat, a_lon = TEAM_COORDS[away_abbr]
    return haversine_miles(a_lat, a_lon, h_lat, h_lon)

def compute_win_streak(series_win: pd.Series) -> pd.Series:
    streak = []
    cur = 0
    for v in series_win.fillna(0).astype(int).tolist():
        if v == 1:
            cur += 1
        else:
            cur = 0
        streak.append(cur)
    return pd.Series(streak, index=series_win.index)

#Loading
print("Loading game_level_data.csv...")
df = pd.read_csv(INPUT_PATH)
df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")

required = ["season", "week", "game_id", "team", "opponent", "is_home", "points_scored", "points_allowed"]
missing_req = [c for c in required if c not in df.columns]
if missing_req:
    raise ValueError(f"Missing required columns in {INPUT_PATH}: {missing_req}")

#Double checking
for col in ["turnovers_committed", "turnovers_forced", "turnover_diff", "yards_per_play", "epa_per_play", "starting_qb"]:
    if col not in df.columns:
        df[col] = np.nan

#Sorting for time
df = df.sort_values(["team", "game_date", "season", "week"]).reset_index(drop=True)

#Win
df["win"] = (pd.to_numeric(df["points_scored"], errors="coerce") > pd.to_numeric(df["points_allowed"], errors="coerce")).astype(int)

#Turnover diff, backup because in previous cases there was issues
df["turnover_diff"] = pd.to_numeric(df["turnover_diff"], errors="coerce")
mask_td_all_nan = df["turnover_diff"].isna().all()
if mask_td_all_nan:
    df["turnover_diff"] = pd.to_numeric(df["turnovers_forced"], errors="coerce") - pd.to_numeric(df["turnovers_committed"], errors="coerce")

#Defense stats via opponent (stats based on offenses)
opp_off = df[["game_id", "team", "yards_per_play", "epa_per_play", "points_scored"]].copy()
opp_off = opp_off.rename(columns={
    "team": "opponent",
    "yards_per_play": "def_yards_per_play_allowed",
    "epa_per_play": "def_epa_per_play_allowed",
    "points_scored": "def_points_allowed",
})
df = df.merge(opp_off, on=["game_id", "opponent"], how="left")

#rest days (we include this because not all teams play every Sunday. There are bye weeks, Thursdays, Mondays, etc)
df["prev_game_date"] = df.groupby("team")["game_date"].shift(1)
df["rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days
df["rest_days"] = df["rest_days"].fillna(7).clip(lower=3, upper=21)
df = df.drop(columns=["prev_game_date"])

#QB Changed
df["prev_starting_qb"] = df.groupby("team")["starting_qb"].shift(1)
df["qb_changed"] = (df["starting_qb"] != df["prev_starting_qb"]).astype(int)
df["qb_changed"] = df["qb_changed"].fillna(0)
df = df.drop(columns=["prev_starting_qb"])

#Home and away win pct
df["home_win_pct_prev"] = np.nan
df["away_win_pct_prev"] = np.nan

for is_home_val, out_col in [(1, "home_win_pct_prev"), (0, "away_win_pct_prev")]:
    tmp = df[df["is_home"] == is_home_val].copy()
    tmp = tmp.sort_values(["team", "game_date", "season", "week"])
    tmp[out_col] = tmp.groupby("team")["win"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df.loc[tmp.index, out_col] = tmp[out_col]

#Win streaks
df["win_streak"] = df.groupby("team")["win"].apply(compute_win_streak).reset_index(level=0, drop=True)
df["win_streak_prev"] = df.groupby("team")["win_streak"].shift(1).fillna(0).astype(int)

#Rolling (based on N prior games, useful for team form.)
ROLL_COLS = [
    "points_scored",
    "points_allowed",
    "yards_per_play",
    "epa_per_play",
    "def_yards_per_play_allowed",
    "def_epa_per_play_allowed",
    "turnovers_committed",
    "turnovers_forced",
    "turnover_diff",
    "qb_changed",
    "rest_days",
    "win_streak_prev",
    "home_win_pct_prev",
    "away_win_pct_prev",
]

for c in ROLL_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"Building rolling features (ROLL_N={ROLL_N})...")
for col in ROLL_COLS:
    df[f"roll_{col}_{ROLL_N}"] = df.groupby("team")[col].transform(
        lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean()
    )

#One row per game
print("Building matchup rows...")

home = df[df["is_home"] == 1].copy()
away = df[df["is_home"] == 0].copy()

game_keys = ["season", "week", "game_id", "game_date"]

home_base = home[game_keys + ["team", "opponent", "points_scored", "points_allowed", "starting_qb"]].copy()
away_base = away[game_keys + ["team", "opponent", "points_scored", "points_allowed", "starting_qb"]].copy()

home_base = home_base.rename(columns={
    "team": "home_team",
    "opponent": "away_team",
    "points_scored": "home_score",
    "points_allowed": "away_score",
    "starting_qb": "home_starting_qb",
})
away_base = away_base.rename(columns={
    "team": "away_team_check",
    "opponent": "home_team_check",
    "points_scored": "away_score_check",
    "points_allowed": "home_score_check",
    "starting_qb": "away_starting_qb",
})

games = home_base.merge(away_base, on=game_keys, how="inner")

#Keep only consistent rows
good = (games["away_team"] == games["away_team_check"]) & (games["home_team"] == games["home_team_check"])
games = games[good].copy()
games = games.drop(columns=["away_team_check", "home_team_check", "away_score_check", "home_score_check"])

#Targets
games["margin"] = pd.to_numeric(games["home_score"], errors="coerce") - pd.to_numeric(games["away_score"], errors="coerce")
games["total_points"] = pd.to_numeric(games["home_score"], errors="coerce") + pd.to_numeric(games["away_score"], errors="coerce")

#Divisional + travel
games["is_divisional"] = (games["home_team"].map(TEAM_DIVISION) == games["away_team"].map(TEAM_DIVISION)).astype(int)
games["away_travel_miles"] = games.apply(lambda r: travel_miles(r["home_team"], r["away_team"]), axis=1)
games["away_travel_miles"] = games["away_travel_miles"].fillna(games["away_travel_miles"].median())

#Rest days (diff)
home_rest = home[game_keys + ["rest_days"]].rename(columns={"rest_days": "home_rest_days"})
away_rest = away[game_keys + ["rest_days"]].rename(columns={"rest_days": "away_rest_days"})
games = games.merge(home_rest, on=game_keys, how="left").merge(away_rest, on=game_keys, how="left")
games["diff_rest_days"] = pd.to_numeric(games["home_rest_days"], errors="coerce") - pd.to_numeric(games["away_rest_days"], errors="coerce")

#Helper: add rolling diffs
def add_diff_roll(games_df, home_df, away_df, roll_col_name, out_name):
    h = home_df[game_keys + [roll_col_name]].rename(columns={roll_col_name: f"home_{roll_col_name}"})
    a = away_df[game_keys + [roll_col_name]].rename(columns={roll_col_name: f"away_{roll_col_name}"})
    tmp = games_df.merge(h, on=game_keys, how="left").merge(a, on=game_keys, how="left")
    games_df[out_name] = pd.to_numeric(tmp[f"home_{roll_col_name}"], errors="coerce") - pd.to_numeric(tmp[f"away_{roll_col_name}"], errors="coerce")
    return games_df

#Required features
games = add_diff_roll(games, home, away, f"roll_points_scored_{ROLL_N}", "diff_roll_points_scored")
games = add_diff_roll(games, home, away, f"roll_points_allowed_{ROLL_N}", "diff_roll_points_allowed")
games = add_diff_roll(games, home, away, f"roll_yards_per_play_{ROLL_N}", "diff_roll_yards_per_play")
games = add_diff_roll(games, home, away, f"roll_epa_per_play_{ROLL_N}", "diff_roll_epa_per_play")
games = add_diff_roll(games, home, away, f"roll_def_yards_per_play_allowed_{ROLL_N}", "diff_roll_def_ypp_allowed")
games = add_diff_roll(games, home, away, f"roll_def_epa_per_play_allowed_{ROLL_N}", "diff_roll_def_epa_allowed")
games = add_diff_roll(games, home, away, f"roll_turnovers_committed_{ROLL_N}", "diff_roll_turnovers_committed")
games = add_diff_roll(games, home, away, f"roll_turnovers_forced_{ROLL_N}", "diff_roll_turnovers_forced")
games = add_diff_roll(games, home, away, f"roll_turnover_diff_{ROLL_N}", "diff_roll_turnover_diff")
games = add_diff_roll(games, home, away, f"roll_qb_changed_{ROLL_N}", "diff_roll_qb_changed")

games = add_diff_roll(games, home, away, f"roll_win_streak_prev_{ROLL_N}", "diff_roll_win_streak")
games = add_diff_roll(games, home, away, f"roll_home_win_pct_prev_{ROLL_N}", "diff_roll_home_win_pct")
games = add_diff_roll(games, home, away, f"roll_away_win_pct_prev_{ROLL_N}", "diff_roll_away_win_pct")

games = games.sort_values(["season", "week", "game_date"]).reset_index(drop=True)

games.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH} ({len(games)} rows)")
#print(games.head(5).to_string(index=False))