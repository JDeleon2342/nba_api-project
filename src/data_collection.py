import time
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

    df = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=0,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    ).get_data_frames()[0]

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


def get_all_teams_shot_data(season, delay = 0.6):
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

    for team_name, team_id in all_team_ids.items():
        print(f"Fetching {team_name} — {season}...")
        try:
            df = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,
                season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",
            ).get_data_frames()[0]

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
            time.sleep(delay * 2)

    return pd.concat(frames, ignore_index=True)


def get_multiple_seasons_shot_data(seasons, delay = 0.6):
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
        print(f"\n=== Season {season} ===")
        df = get_all_teams_shot_data(season, delay=delay)
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
            df = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame"
            ).get_data_frames()[0]

            df["season"] = season
            frames.append(df)
            time.sleep(0.6)

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