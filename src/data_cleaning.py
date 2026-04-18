import pandas as pd
import numpy as np


# ── Column Standardization ──────────────────────────────────────────────────

def standardize_columns(df):
    """
    Standardize column names to lowercase with underscores and
    rename key columns for consistency.

    Args:
        df: Raw DataFrame from NBA API

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        "team_name":            "team",
        "shot_made_flag":       "made",
        "shot_attempted_flag":  "attempted",
        "shot_distance":        "distance",
        "loc_x":                "x",
        "loc_y":                "y",
        "shot_zone_basic":      "zone_basic",
        "shot_zone_area":       "zone_area",
        "shot_zone_range":      "zone_range",
        "shot_type":            "shot_type",
        "action_type":          "action_type",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


# ── Missing Value Handling ──────────────────────────────────────────────────

def drop_missing_coordinates(df):
    """
    Remove rows where shot coordinates (x, y) are missing.

    Args:
        df: Shot data DataFrame with 'x' and 'y' columns

    Returns:
        Cleaned DataFrame with no missing coordinates
    """
    before = len(df)
    df = df.dropna(subset=["x", "y"]).copy()
    after = len(df)
    print(f"Dropped {before - after} rows with missing coordinates")
    return df


def drop_backcourt_shots(df):
    """
    Remove backcourt heave shots that are not meaningful
    for shot selection analysis.

    Args:
        df: Shot data DataFrame with 'shot_zone' column

    Returns:
        DataFrame with backcourt shots removed
    """
    before = len(df)
    df = df[df["zone_basic"] != "Backcourt"].copy()
    after = len(df)
    print(f"Dropped {before - after} backcourt shots")
    return df


# ── Feature Engineering ─────────────────────────────────────────────────────

def add_three_point_flag(df):
    """
    Add a binary column indicating whether a shot was a three pointer.

    Args:
        df: Shot data DataFrame with 'shot_zone' column

    Returns:
        DataFrame with new 'is_three' column (1 = three pointer, 0 = two pointer)
    """
    df = df.copy()
    three_zones = {"Above the Break 3", "Left Corner 3", "Right Corner 3"}
    df["is_three"] = df["zone_basic"].isin(three_zones).astype(int)
    return df


def add_points_value(df):
    """
    Add a column for points scored on each shot attempt.

    Args:
        df: Shot data DataFrame with 'made' and 'is_three' columns

    Returns:
        DataFrame with new 'points' column
    """
    df = df.copy()
    df["points"] = df["made"] * (df["is_three"] + 2)
    return df


def add_era(df):
    """
    Add an era label column based on season.

    Eras:
        - Early (2013-14 to 2016-17)
        - Mid   (2017-18 to 2020-21)
        - Recent (2021-22 to 2023-24)

    Args:
        df: Shot data DataFrame with 'season' column

    Returns:
        DataFrame with new 'era' column
    """
    df = df.copy()

    def label_era(season):
        year = int(season.split("-")[0])
        if year <= 2016:
            return "Early (2013-17)"
        elif year <= 2020:
            return "Mid (2017-21)"
        else:
            return "Recent (2021-24)"

    df["era"] = df["season"].apply(label_era)
    return df


# ── Team Metrics Cleaning ───────────────────────────────────────────────────

def clean_team_metrics(df):
    """
    Clean and standardize the team season metrics DataFrame.

    Args:
        df: Raw team metrics DataFrame from collection module

    Returns:
        Cleaned DataFrame with standardized columns and no missing values
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    keep = [
        "team_name", "season", "w", "l", "w_pct",
        "pts", "fg_pct", "fg3_pct", "fg3a",
        "oreb", "dreb", "reb", "ast", "tov", "blk"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    rename_map = {
        "team_name": "team",
        "w":         "wins",
        "l":         "losses",
        "w_pct":     "win_pct",
        "pts":       "pts_per_game",
        "fg_pct":    "fg_pct",
        "fg3_pct":   "three_pt_pct",
        "fg3a":      "three_pt_attempts",
        "oreb":      "off_reb",
        "dreb":      "def_reb",
        "reb":       "total_reb",
        "ast":       "assists",
        "tov":       "turnovers",
        "blk":       "blocks"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.dropna(subset=["team", "season"])

    return df


# ── Master Cleaning Pipeline ────────────────────────────────────────────────

def clean_shot_data(df):
    """
    Run the full cleaning pipeline on raw shot data.

    Steps:
        1. Standardize column names
        2. Drop rows with missing coordinates
        3. Drop backcourt shots
        5. Add three point flag
        5. Add points value
        6. Add era label

    Args:
        df: Raw shot data DataFrame from collection module

    Returns:
        Fully cleaned and enriched shot data DataFrame
    """
    print("Starting shot data cleaning pipeline...")
    df = standardize_columns(df)
    df = drop_missing_coordinates(df)
    df = drop_backcourt_shots(df)
    df = add_three_point_flag(df)
    df = add_points_value(df)
    df = add_era(df)
    print(f"Cleaning complete. Final shape: {df.shape}")
    return df