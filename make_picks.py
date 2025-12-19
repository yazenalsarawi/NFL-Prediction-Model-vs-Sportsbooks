import os
import requests
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2, erf
from sklearn.metrics import accuracy_score

#Basic configuration stuff
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
if not ODDS_API_KEY:
    raise RuntimeError("Missing ODDS_API_KEY env var. Set it before running.")

MODEL_ARTIFACT = "model_artifacts.pkl"
GAME_LEVEL_PATH = "game_level_data.csv"
DATASET_PATH = "model_dataset.csv"

OUTPUT_CSV = "weekly_picks.csv"
OUTPUT_XLSX = "weekly_picks.xlsx"

ROLLING_N = 5
REGION = "us"
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"

PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus", "betrivers", "williamhill_us"]

EDGE_THRESHOLD_SPREAD = 2.0     #points
EDGE_THRESHOLD_ML = 0.05        #5% vs implied
EDGE_THRESHOLD_TOTAL = 2.0      #points

LEAGUE_AVG_TOTAL = 44.0         #used if total missing
ALIGN_PLAYS_WITH_PRED_WINNER = True

#7 day slate for games
SLATE_DAYS = 7
DISPLAY_TZ = "America/New_York"

#Team coords and abbreviastions/divisons
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
    "BUF":"AFC_E","MIA":"AFC_E","NE":"AFC_E","NYJ":"AFC_E",
    "BAL":"AFC_N","CIN":"AFC_N","CLE":"AFC_N","PIT":"AFC_N",
    "HOU":"AFC_S","IND":"AFC_S","JAX":"AFC_S","TEN":"AFC_S",
    "DEN":"AFC_W","KC":"AFC_W","LAC":"AFC_W","LV":"AFC_W",
    "DAL":"NFC_E","NYG":"NFC_E","PHI":"NFC_E","WAS":"NFC_E",
    "CHI":"NFC_N","DET":"NFC_N","GB":"NFC_N","MIN":"NFC_N",
    "ATL":"NFC_S","CAR":"NFC_S","NO":"NFC_S","TB":"NFC_S",
    "ARI":"NFC_W","LAR":"NFC_W","SEA":"NFC_W","SF":"NFC_W",
}

#API team names abbreviated for simplicitys sake
FULLNAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

#Helpers
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat/2)**2) + cos(radians(lat1)) * cos(radians(lat2)) * (sin(dlon/2)**2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def fmt_spread(x):
    if pd.isna(x):
        return ""
    x = float(x)
    return f"+{x:g}" if x > 0 else f"{x:g}"

def fmt_american(x):
    if pd.isna(x):
        return ""
    x = int(float(x))
    return f"+{x}" if x > 0 else f"{x}"

def american_to_implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def to_display_time(dt_utc):
    try:
        return dt_utc.tz_convert(DISPLAY_TZ).tz_localize(None)
    except Exception:
        return dt_utc.tz_localize(None)

#Loads trained models
art = joblib.load(MODEL_ARTIFACT)

pipe_margin = art.get("pipeline_margin", art.get("pipeline", None))
features_margin = art.get("features_margin", art.get("features", None))

pipe_total = art.get("pipeline_total", None)
features_total = art.get("features_total", features_margin)

sigma_margin = art.get("sigma_margin", None)

if pipe_margin is None or features_margin is None:
    raise RuntimeError("model_artifacts.pkl missing pipeline/features. Re-run training.")

has_total_model = pipe_total is not None

#Load history and loads rolling stats
hist = pd.read_csv(GAME_LEVEL_PATH)
hist["game_date"] = pd.to_datetime(hist["game_date"], errors="coerce")
hist = hist.dropna(subset=["team", "game_date"]).sort_values(["team", "game_date"])

#opponent offense -> defensive allowed
opp_off = hist[["game_id", "team", "yards_per_play", "epa_per_play", "points_scored"]].copy()
opp_off = opp_off.rename(columns={
    "team": "opponent",
    "yards_per_play": "def_yards_per_play_allowed",
    "epa_per_play": "def_epa_per_play_allowed",
    "points_scored": "def_points_allowed"
})
hist = hist.merge(opp_off, on=["game_id", "opponent"], how="left")

hist["turnover_diff"] = hist["turnovers_forced"] - hist["turnovers_committed"]

ROLL_COLS = [
    "points_scored", "points_allowed",
    "yards_per_play", "epa_per_play",
    "def_yards_per_play_allowed", "def_epa_per_play_allowed",
    "turnovers_committed", "turnovers_forced", "turnover_diff"
]

hist = hist.sort_values(["team", "game_date", "season", "week"]).reset_index(drop=True)
last_game_date = hist.groupby("team")["game_date"].max()

#Rolling averages
for col in ROLL_COLS:
    hist[f"roll_{col}_{ROLLING_N}"] = (
        hist.groupby("team")[col]
            .shift(1)
            .rolling(ROLLING_N, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

#Qb change rolling
hist["prev_starting_qb"] = hist.groupby("team")["starting_qb"].shift(1)
hist["qb_changed"] = (hist["starting_qb"] != hist["prev_starting_qb"]).astype(int)
hist["qb_changed"] = hist["qb_changed"].fillna(0)
hist = hist.drop(columns=["prev_starting_qb"])

hist[f"roll_qb_changed_{ROLLING_N}"] = (
    hist.groupby("team")["qb_changed"]
        .shift(1)
        .rolling(ROLLING_N, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
)

#Win streak entering next game
hist["streak"] = 0
for t, g in hist.groupby("team", sort=False):
    cur = 0
    streaks = []
    for w in g["win"].fillna(0).astype(int).tolist():
        if w == 1:
            cur += 1
        else:
            cur = 0
        streaks.append(cur)
    hist.loc[g.index, "streak"] = streaks

hist[f"roll_win_streak_{ROLLING_N}"] = (
    hist.groupby("team")["streak"]
        .shift(1)
        .rolling(ROLLING_N, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
)

#home/away rolling win%
home_only = hist[hist["is_home"] == 1].copy()
away_only = hist[hist["is_home"] == 0].copy()

home_only[f"roll_home_win_pct_{ROLLING_N}"] = (
    home_only.groupby("team")["win"]
        .shift(1)
        .rolling(ROLLING_N, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
)
away_only[f"roll_away_win_pct_{ROLLING_N}"] = (
    away_only.groupby("team")["win"]
        .shift(1)
        .rolling(ROLLING_N, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
)

latest = hist.sort_values(["team", "game_date"]).groupby("team").tail(1).set_index("team")
latest_home = home_only.sort_values(["team", "game_date"]).groupby("team").tail(1).set_index("team")
latest_away = away_only.sort_values(["team", "game_date"]).groupby("team").tail(1).set_index("team")

def get_latest(team, col):
    if team not in latest.index:
        return np.nan
    return latest.loc[team].get(col, np.nan)

def get_latest_home(team, col):
    if team not in latest_home.index:
        return np.nan
    return latest_home.loc[team].get(col, np.nan)

def get_latest_away(team, col):
    if team not in latest_away.index:
        return np.nan
    return latest_away.loc[team].get(col, np.nan)

#Fallback sigma_margin
if sigma_margin is None or pd.isna(sigma_margin):
    sigma_margin = float(np.nanstd(hist["points_scored"] - hist["points_allowed"]))
sigma_margin = max(float(sigma_margin), 10.0)

#Fetching odds from API
url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
params = {
    "apiKey": ODDS_API_KEY,
    "regions": REGION,
    "markets": MARKETS,
    "oddsFormat": ODDS_FORMAT,
}

print("Fetching odds from The Odds API...")
resp = requests.get(url, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

rows = []
for ev in data:
    event_id = ev.get("id", None)
    commence = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")

    home_full = ev.get("home_team")
    away_full = ev.get("away_team")
    if pd.isna(commence) or not home_full or not away_full:
        continue

    #Maps full name to abbreviation
    if home_full not in FULLNAME_TO_ABBR or away_full not in FULLNAME_TO_ABBR:
        continue
    home = FULLNAME_TO_ABBR[home_full]
    away = FULLNAME_TO_ABBR[away_full]

    #pick preferred sportsbook
    book = None
    for b in ev.get("bookmakers", []):
        if b.get("key") in PREFERRED_BOOKS:
            book = b
            break
    if book is None:
        bms = ev.get("bookmakers", [])
        if not bms:
            continue
        book = bms[0]

    book_key = book.get("key", "")

    spread_home = np.nan
    ml_home = np.nan
    ml_away = np.nan
    total = np.nan

    for m in book.get("markets", []):
        k = m.get("key")

        if k == "spreads":
            for out in m.get("outcomes", []):
                if out.get("name") == home_full:
                    spread_home = out.get("point", np.nan)

        elif k == "h2h":
            for out in m.get("outcomes", []):
                if out.get("name") == home_full:
                    ml_home = out.get("price", np.nan)
                elif out.get("name") == away_full:
                    ml_away = out.get("price", np.nan)

        elif k == "totals":
            outs = m.get("outcomes", [])
            if outs:
                total = outs[0].get("point", np.nan)

    rows.append({
        "event_id": event_id,
        "game_date_utc": commence,
        "game_date": to_display_time(commence),
        "book_key": book_key,

        #Abbreviations
        "home_team": home,
        "away_team": away,

        "spread_home": spread_home,
        "ml_home": ml_home,
        "ml_away": ml_away,
        "total": total,
    })

odds_df = pd.DataFrame(rows)
if odds_df.empty:
    raise RuntimeError("No odds returned from API (check key/region/markets).")

#Added in case of duplicates, which happened in previous runs
dedupe_cols = ["event_id", "book_key"] if odds_df["event_id"].notna().any() else ["game_date", "home_team", "away_team", "book_key"]
odds_df = odds_df.sort_values(["game_date", "book_key"]).drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

#What games to pick
min_dt = odds_df["game_date"].min()
max_dt = min_dt + pd.Timedelta(days=SLATE_DAYS)
odds_df = odds_df[(odds_df["game_date"] >= min_dt) & (odds_df["game_date"] < max_dt)].copy()
odds_df = odds_df.sort_values("game_date").reset_index(drop=True)

#Travel miles + rest days + rolling diffs
def travel_miles(home_abbr, away_abbr):
    if home_abbr not in TEAM_COORDS or away_abbr not in TEAM_COORDS:
        return np.nan
    h_lat, h_lon = TEAM_COORDS[home_abbr]
    a_lat, a_lon = TEAM_COORDS[away_abbr]
    return haversine_miles(a_lat, a_lon, h_lat, h_lon)

up = odds_df.copy()

up["is_divisional"] = (up["home_team"].map(TEAM_DIVISION) == up["away_team"].map(TEAM_DIVISION)).astype(int)

up["away_travel_miles"] = up.apply(lambda r: travel_miles(r["home_team"], r["away_team"]), axis=1)

median_travel = up["away_travel_miles"].median()
up["away_travel_miles"] = up["away_travel_miles"].fillna(0 if pd.isna(median_travel) else median_travel)

def compute_rest(team, game_dt):
    if team not in last_game_date.index or pd.isna(game_dt):
        return 7.0
    d = (game_dt - last_game_date.loc[team]).days
    return float(np.clip(d, 3, 21))

up["home_rest_days"] = up.apply(lambda r: compute_rest(r["home_team"], r["game_date"]), axis=1)
up["away_rest_days"] = up.apply(lambda r: compute_rest(r["away_team"], r["game_date"]), axis=1)
up["diff_rest_days"] = up["home_rest_days"] - up["away_rest_days"]

#rolling diffs (home-away)
up["diff_roll_points_scored"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_points_scored_{ROLLING_N}")) - \
                                up["away_team"].apply(lambda t: get_latest(t, f"roll_points_scored_{ROLLING_N}"))

up["diff_roll_points_allowed"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_points_allowed_{ROLLING_N}")) - \
                                 up["away_team"].apply(lambda t: get_latest(t, f"roll_points_allowed_{ROLLING_N}"))

up["diff_roll_yards_per_play"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_yards_per_play_{ROLLING_N}")) - \
                                 up["away_team"].apply(lambda t: get_latest(t, f"roll_yards_per_play_{ROLLING_N}"))

up["diff_roll_epa_per_play"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_epa_per_play_{ROLLING_N}")) - \
                               up["away_team"].apply(lambda t: get_latest(t, f"roll_epa_per_play_{ROLLING_N}"))

up["diff_roll_def_ypp_allowed"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_def_yards_per_play_allowed_{ROLLING_N}")) - \
                                  up["away_team"].apply(lambda t: get_latest(t, f"roll_def_yards_per_play_allowed_{ROLLING_N}"))

up["diff_roll_def_epa_allowed"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_def_epa_per_play_allowed_{ROLLING_N}")) - \
                                  up["away_team"].apply(lambda t: get_latest(t, f"roll_def_epa_per_play_allowed_{ROLLING_N}"))

up["diff_roll_turnovers_committed"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_turnovers_committed_{ROLLING_N}")) - \
                                      up["away_team"].apply(lambda t: get_latest(t, f"roll_turnovers_committed_{ROLLING_N}"))

up["diff_roll_turnovers_forced"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_turnovers_forced_{ROLLING_N}")) - \
                                   up["away_team"].apply(lambda t: get_latest(t, f"roll_turnovers_forced_{ROLLING_N}"))

up["diff_roll_turnover_diff"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_turnover_diff_{ROLLING_N}")) - \
                                up["away_team"].apply(lambda t: get_latest(t, f"roll_turnover_diff_{ROLLING_N}"))

up["diff_roll_qb_changed"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_qb_changed_{ROLLING_N}")) - \
                             up["away_team"].apply(lambda t: get_latest(t, f"roll_qb_changed_{ROLLING_N}"))

up["diff_roll_win_streak"] = up["home_team"].apply(lambda t: get_latest(t, f"roll_win_streak_{ROLLING_N}")) - \
                             up["away_team"].apply(lambda t: get_latest(t, f"roll_win_streak_{ROLLING_N}"))

up["diff_roll_home_win_pct"] = up["home_team"].apply(lambda t: get_latest_home(t, f"roll_home_win_pct_{ROLLING_N}")) - \
                               up["away_team"].apply(lambda t: get_latest_home(t, f"roll_home_win_pct_{ROLLING_N}"))

up["diff_roll_away_win_pct"] = up["home_team"].apply(lambda t: get_latest_away(t, f"roll_away_win_pct_{ROLLING_N}")) - \
                               up["away_team"].apply(lambda t: get_latest_away(t, f"roll_away_win_pct_{ROLLING_N}"))

up["diff_travel_miles"] = 0.0 - up["away_travel_miles"]

#Prediction, margin and total
X_margin = up.reindex(columns=features_margin)
up["pred_margin"] = pipe_margin.predict(X_margin)
up["pred_winner"] = np.where(up["pred_margin"] >= 0, up["home_team"], up["away_team"])

if has_total_model:
    X_total = up.reindex(columns=features_total)
    up["pred_total_points"] = pipe_total.predict(X_total)
else:
    up["pred_total_points"] = np.nan

#Expected score
up["expected_total"] = up["pred_total_points"]
up["expected_total"] = up["expected_total"].fillna(up["total"])
up["expected_total"] = up["expected_total"].fillna(LEAGUE_AVG_TOTAL)

up["expected_home_score"] = ((up["expected_total"] + up["pred_margin"]) / 2).round(1)
up["expected_away_score"] = ((up["expected_total"] - up["pred_margin"]) / 2).round(1)

def expected_result(row):
    m = float(row["pred_margin"])
    if m >= 0:
        return f"{row['home_team']} by ≈ {abs(m):.1f}"
    return f"{row['away_team']} by ≈ {abs(m):.1f}"

up["expected_result"] = up.apply(expected_result, axis=1)

#Spread pick
up["edge_vs_spread"] = up["pred_margin"] + up["spread_home"]

def spread_pick(row):
    if pd.isna(row["spread_home"]) or pd.isna(row["edge_vs_spread"]):
        return "NO PLAY", np.nan

    e = float(row["edge_vs_spread"])

    if e >= EDGE_THRESHOLD_SPREAD:
        pick_team = row["home_team"]
        pick_line = float(row["spread_home"])
        pick_edge = abs(e)
    elif e <= -EDGE_THRESHOLD_SPREAD:
        pick_team = row["away_team"]
        pick_line = -float(row["spread_home"])  #away line
        pick_edge = abs(e)
    else:
        return "NO PLAY", np.nan

    if ALIGN_PLAYS_WITH_PRED_WINNER and pick_team != row["pred_winner"]:
        return "NO PLAY", np.nan

    return f"Bet {pick_team} {fmt_spread(pick_line)}", pick_edge

up[["spread_play", "spread_edge_pts"]] = up.apply(lambda r: pd.Series(spread_pick(r)), axis=1)

#ML value
up["p_home_win_model"] = up["pred_margin"].apply(lambda m: norm_cdf(float(m) / float(sigma_margin)))

up["p_home_implied"] = up["ml_home"].astype(float).apply(american_to_implied_prob)
up["p_away_implied"] = up["ml_away"].astype(float).apply(american_to_implied_prob)

up["ml_edge_home"] = up["p_home_win_model"] - up["p_home_implied"]
up["ml_edge_away"] = (1.0 - up["p_home_win_model"]) - up["p_away_implied"]

up["model_home_win_pct"] = (up["p_home_win_model"] * 100).round(1)
up["implied_home_win_pct"] = (up["p_home_implied"] * 100).round(1)
up["implied_away_win_pct"] = (up["p_away_implied"] * 100).round(1)

def ml_pick(row):
    if pd.isna(row["ml_edge_home"]) or pd.isna(row["ml_edge_away"]):
        return "NO PLAY", np.nan

    home_edge = float(row["ml_edge_home"])
    away_edge = float(row["ml_edge_away"])

    #pick larger edge if it clears threshold
    if home_edge >= EDGE_THRESHOLD_ML and home_edge >= away_edge:
        pick_team = row["home_team"]
        pick_edge = home_edge
    elif away_edge >= EDGE_THRESHOLD_ML:
        pick_team = row["away_team"]
        pick_edge = away_edge
    else:
        return "NO PLAY", np.nan

    if ALIGN_PLAYS_WITH_PRED_WINNER and pick_team != row["pred_winner"]:
        return "NO PLAY", np.nan

    return f"Bet {pick_team} ML", pick_edge

up[["ml_play", "ml_edge_pct"]] = up.apply(lambda r: pd.Series(ml_pick(r)), axis=1)

#Total pick
up["edge_vs_total"] = up["pred_total_points"] - up["total"]

def total_pick(row):
    if pd.isna(row["total"]) or pd.isna(row["edge_vs_total"]):
        return "NO PLAY", np.nan

    e = float(row["edge_vs_total"])
    if e >= EDGE_THRESHOLD_TOTAL:
        return f"Bet Over {row['total']:g}", abs(e)
    if e <= -EDGE_THRESHOLD_TOTAL:
        return f"Bet Under {row['total']:g}", abs(e)
    return "NO PLAY", np.nan

up[["total_play", "total_edge_pts"]] = up.apply(lambda r: pd.Series(total_pick(r)), axis=1)

#Edge summary
def edge_summary(row):
    # Prefer the play that exists. simple priority: Spread > ML > Total
    if row["spread_play"] != "NO PLAY" and not pd.isna(row["spread_edge_pts"]):
        return f"Spread edge {row['spread_edge_pts']:.1f} pts"
    if row["ml_play"] != "NO PLAY" and not pd.isna(row["ml_edge_pct"]):
        return f"ML edge {row['ml_edge_pct']*100:.1f}%"
    if row["total_play"] != "NO PLAY" and not pd.isna(row["total_edge_pts"]):
        return f"Total edge {row['total_edge_pts']:.1f} pts"
    return ""

up["edge_ev"] = up.apply(edge_summary, axis=1)

#Output table
out = pd.DataFrame({
    "Game Date": up["game_date"],
    "Book": up["book_key"],
    "Matchup": up["away_team"] + " @ " + up["home_team"],
    "Spread (Home)": up["spread_home"].apply(fmt_spread),
    "Home ML": up["ml_home"].apply(fmt_american),
    "Away ML": up["ml_away"].apply(fmt_american),
    "Total": up["total"],

    "Expected Score": up["home_team"] + " " + up["expected_home_score"].astype(str) +
                      " - " + up["away_team"] + " " + up["expected_away_score"].astype(str),
    "Expected Result": up["expected_result"],

    "Spread Play": up["spread_play"],
    "ML Play": up["ml_play"],
    "Total Play": up["total_play"],

    "Model Home Win %": up["model_home_win_pct"].astype(str) + "%",
    "Implied Home Win %": up["implied_home_win_pct"].astype(str) + "%",
    "Implied Away Win %": up["implied_away_win_pct"].astype(str) + "%",

    "Edge/EV": up["edge_ev"],
}).sort_values("Game Date").reset_index(drop=True)

#Save CSV
out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved weekly picks to: {OUTPUT_CSV}")
print(out.to_string(index=False))

#Save XLSX
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="Picks", index=False)
print(f"Saved weekly picks to: {OUTPUT_XLSX}")

#Model accuracy (winner) on latest season
df_eval = pd.read_csv(DATASET_PATH)

df_eval["season"] = pd.to_numeric(df_eval["season"], errors="coerce")
eval_season = int(df_eval["season"].dropna().max())
df_eval = df_eval[df_eval["season"] == eval_season].copy()

df_eval["margin"] = pd.to_numeric(df_eval["margin"], errors="coerce")

df_eval = df_eval.dropna(subset=["margin"] + list(features_margin))

y_true = (df_eval["margin"].astype(float) > 0).astype(int)

X_eval = df_eval[list(features_margin)].copy()
y_pred = (pipe_margin.predict(X_eval) > 0).astype(int)

acc = accuracy_score(y_true, y_pred)
print(f"Model accuracy (winner) on season {eval_season}: {acc*100:.1f}%")