import time
import os
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail, leaguedashteamstats
from nba_api.stats.static import teams


# ── Helpers ────────────────────────────────────────────────────────────────

def get_team_id(team_name):
    """
    Return the NBA team ID for a given team name.

    Args:
        team_name: Full or partial team name (e.g. 'Lakers', 'Golden State Warriors')

    Returns:
        Integer team ID

    Raises:
        ValueError: If no matching team is found
    """
    all_teams = teams.get_teams()
    matches = [
        t for t in all_teams
        if team_name.lower() in t["full_name"].lower()
        or team_name.lower() in t["nickname"].lower()
    ]
    if not matches:
        raise ValueError(f"No team found matching '{team_name}'")
    return matches[0]["id"]


def normalize_season_string(season):
    """
    Normalize season strings to the NBA Stats expected format 'YYYY-YY'.

    Examples:
        '2010-2011' -> '2010-11'
        '2018-19'   -> '2018-19' (unchanged)
    """
    # handle formats like '2010-2011' or '2010/2011'
    try:
        parts = season.replace('/', '-').split('-')
        if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) in (2, 4):
            start = parts[0]
            end = parts[1]
            if len(end) == 4:
                end = end[2:]
            return f"{start}-{end}"
    except Exception:
        pass
    return season


def get_all_team_ids():
    """
    Return a dictionary mapping team full names to their NBA team IDs.

    Returns:
        Dict of {team_full_name: team_id}
    """
    return {t["full_name"]: t["id"] for t in teams.get_teams()}


# ── Shot Data ───────────────────────────────────────────────────────────────

def get_shot_data(team_name, season):
    """
    Collect shot chart data for a given team and season from the NBA Stats API.

    Args:
        team_name: Team name string (e.g. 'Lakers')
        season: Season string in format '2023-24'

    Returns:
        DataFrame with shot level data including coordinates, zone, and outcome
    """
    team_id = get_team_id(team_name)
    season = normalize_season_string(season)

    data_frames = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=0,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    ).get_data_frames()

    if not data_frames:
        raise RuntimeError(f"No shotchart results for {team_name} ({team_id}) season {season}")

    df = data_frames[0]

    cols = [
        "TEAM_NAME", "SEASON_1", "PERIOD",
        "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
        "SHOT_DISTANCE", "LOC_X", "LOC_Y",
        "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG",
        "ACTION_TYPE", "SHOT_TYPE"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df["season"] = season

    return df


def get_all_teams_shot_data(season, delay = .6):
    """
    Collect shot chart data for ALL 30 NBA teams for a given season.

    Args:
        season: Season string in format '2023-24'
        delay: Seconds to wait between API calls to avoid rate limiting

    Returns:
        Combined DataFrame with shot data for all teams
    """
    all_team_ids = get_all_team_ids()
    frames = []

    season = normalize_season_string(season)
    for team_name, team_id in all_team_ids.items():
        print(f"Fetching {team_name} — {season}...")
        try:
            data_frames = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,
                season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",
            ).get_data_frames()

            if not data_frames:
                print(f"  No shot data returned for {team_name} ({team_id}) season {season} - skipping")
                time.sleep(delay)
                continue

            df = data_frames[0]

            cols = [
                "TEAM_NAME", "PERIOD",
                "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
                "SHOT_DISTANCE", "LOC_X", "LOC_Y",
                "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG",
                "ACTION_TYPE", "SHOT_TYPE"
            ]
            cols = [c for c in cols if c in df.columns]
            df = df[cols].copy()
            df["season"] = season
            frames.append(df)
            time.sleep(delay)

        except Exception as e:
            print(f"  Failed for {team_name}: {e}")
            # backoff a little more when an error occurs
            time.sleep(max(delay * 2, 1.0))

    return pd.concat(frames, ignore_index=True)


def get_multiple_seasons_shot_data(seasons, delay = .6):
    """
    Collect shot data for all teams across multiple seasons.

    Args:
        seasons: List of season strings e.g. ['2013-14', '2014-15', ...]
        delay: Seconds between API calls

    Returns:
        Combined DataFrame across all seasons
    """
    frames = []
    for season in seasons:
        norm = normalize_season_string(season)
        print(f"\n=== Season {season} -> {norm} ===")
        df = get_all_teams_shot_data(norm, delay=delay)
        if df is not None and len(df):
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ── Team Season Metrics ─────────────────────────────────────────────────────

def get_team_season_metrics(season_range):
    """
    Collect team level offensive metrics for a list of seasons
    from the NBA Stats API.

    Args:
        season_range: List of season strings e.g. ['2013-14', '2023-24']

    Returns:
        DataFrame with one row per team per season including
        offensive rating, eFG%, pace, and win percentage
    """
    frames = []

    for season in season_range:
        print(f"Fetching team metrics — {season}...")
        try:
            norm = normalize_season_string(season)
            data_frames = leaguedashteamstats.LeagueDashTeamStats(
                season=norm,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame"
            ).get_data_frames()

            if not data_frames:
                print(f"  No metrics returned for season {season} (normalized {norm}) - skipping")
                time.sleep(0.6)
                continue

            df = data_frames[0]
            df["season"] = norm
            frames.append(df)
            time.sleep(0.8)

        except Exception as e:
            print(f"  Failed for season {season}: {e}")

    return pd.concat(frames, ignore_index=True)


# ── Save / Load ─────────────────────────────────────────────────────────────

def save_dataset(data, path):
    """
    Save a DataFrame to a CSV file.

    Args:
        data: DataFrame to save
        path: File path string (e.g. 'data/shot_data.csv')
    """
    # ensure parent directory exists (when path is just a filename, dirname=="")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    data.to_csv(path, index=False)
    print(f"Saved {len(data)} rows to {path}")


def load_dataset(path):
    """
    Load a CSV file into a DataFrame.

    Args:
        path: File path string

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path)