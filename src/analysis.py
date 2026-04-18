import pandas as pd
import numpy as np


# ── Shot Zone Distribution ──────────────────────────────────────────────────

def zone_distribution(df: pd.DataFrame, group_by: str = "season") -> pd.DataFrame:
    """
    Calculate the percentage of shots from each zone grouped by a given column.

    Args:
        df: Cleaned shot data DataFrame
        group_by: Column to group by — 'season', 'team', or 'era'

    Returns:
        DataFrame with shot zone percentages per group
    """
    grouped = (
        df.groupby([group_by, "zone_basic"])
        .size()
        .reset_index(name="count")
    )

    totals = grouped.groupby(group_by)["count"].transform("sum")
    grouped["pct"] = grouped["count"] / totals * 100

    return grouped


def three_point_rate_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the league wide three point attempt rate per season.

    Args:
        df: Cleaned shot data DataFrame with 'is_three' and 'season' columns

    Returns:
        DataFrame with columns season and three_pt_rate
    """
    result = (
        df.groupby("season")
        .agg(
            total_attempts=("attempted", "sum"),
            three_pt_attempts=("is_three", "sum")
        )
        .reset_index()
    )
    result["three_pt_rate"] = result["three_pt_attempts"] / result["total_attempts"] * 100
    return result.sort_values("season")


def three_point_rate_by_team_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate three point attempt rate per team per season.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        DataFrame with columns team, season, three_pt_rate
    """
    result = (
        df.groupby(["team", "season"])
        .agg(
            total_attempts=("attempted", "sum"),
            three_pt_attempts=("is_three", "sum")
        )
        .reset_index()
    )
    result["three_pt_rate"] = result["three_pt_attempts"] / result["total_attempts"] * 100
    return result.sort_values(["team", "season"])


# ── Efficiency Metrics ──────────────────────────────────────────────────────

def compute_efg(df: pd.DataFrame, group_by: list) -> pd.DataFrame:
    """
    Compute effective field goal percentage (eFG%) grouped by given columns.

    eFG% = (FGM + 0.5 * 3PM) / FGA

    Args:
        df: Cleaned shot data DataFrame with 'made', 'is_three', 'attempted'
        group_by: List of columns to group by e.g. ['team', 'season']

    Returns:
        DataFrame with eFG% per group
    """
    result = (
        df.groupby(group_by)
        .agg(
            fga=("attempted", "sum"),
            fgm=("made", "sum"),
            three_pm=("is_three", lambda x: (x * df.loc[x.index, "made"]).sum())
        )
        .reset_index()
    )
    result["efg_pct"] = (result["fgm"] + 0.5 * result["three_pm"]) / result["fga"] * 100
    return result


def compute_points_per_shot(df: pd.DataFrame, group_by: list) -> pd.DataFrame:
    """
    Compute points per shot attempt grouped by given columns.

    Args:
        df: Cleaned shot data DataFrame with 'points' and 'attempted' columns
        group_by: List of columns to group by e.g. ['zone_basic', 'season']

    Returns:
        DataFrame with points_per_shot per group
    """
    result = (
        df.groupby(group_by)
        .agg(
            total_points=("points", "sum"),
            total_attempts=("attempted", "sum")
        )
        .reset_index()
    )
    result["points_per_shot"] = result["total_points"] / result["total_attempts"]
    return result


# ── Era Comparison ──────────────────────────────────────────────────────────

def compare_eras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare shot zone distribution across early, mid, and recent eras.

    Args:
        df: Cleaned shot data DataFrame with 'era' and 'zone_basic' columns

    Returns:
        DataFrame with shot zone percentages per era
    """
    return zone_distribution(df, group_by="era")


def biggest_zone_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which shot zones changed the most between the earliest
    and most recent season in the dataset.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        DataFrame showing zone pct in first season, last season, and the difference
    """
    seasons = sorted(df["season"].unique())
    first_season = seasons[0]
    last_season  = seasons[-1]

    dist = zone_distribution(df, group_by="season")

    first = dist[dist["season"] == first_season][["zone_basic", "pct"]].rename(
        columns={"pct": "pct_first"}
    )
    last = dist[dist["season"] == last_season][["zone_basic", "pct"]].rename(
        columns={"pct": "pct_last"}
    )

    merged = first.merge(last, on="zone_basic")
    merged["change"] = merged["pct_last"] - merged["pct_first"]
    return merged.sort_values("change", ascending=False)


# ── Team Analysis ───────────────────────────────────────────────────────────

def team_shot_profile(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Return shot zone distribution over all seasons for a specific team.

    Args:
        df: Cleaned shot data DataFrame
        team: Team name string e.g. 'Houston Rockets'

    Returns:
        DataFrame with zone percentages per season for that team
    """
    team_df = df[df["team"] == team]
    if team_df.empty:
        raise ValueError(f"No data found for team '{team}'")
    return zone_distribution(team_df, group_by="season")


def rank_teams_by_three_point_adoption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all teams by how much their three point rate changed
    from their first to last season in the dataset.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        DataFrame ranked by change in three point rate descending
    """
    rates = three_point_rate_by_team_season(df)

    first = (
        rates.sort_values("season")
        .groupby("team")
        .first()
        .reset_index()
        [["team", "three_pt_rate"]]
        .rename(columns={"three_pt_rate": "rate_first"})
    )

    last = (
        rates.sort_values("season")
        .groupby("team")
        .last()
        .reset_index()
        [["team", "three_pt_rate"]]
        .rename(columns={"three_pt_rate": "rate_last"})
    )

    merged = first.merge(last, on="team")
    merged["change"] = merged["rate_last"] - merged["rate_first"]
    return merged.sort_values("change", ascending=False)


def correlate_three_pt_rate_with_wins(
    shot_df: pd.DataFrame,
    metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge shot data with team metrics and compute correlation between
    three point attempt rate and win percentage.

    Args:
        shot_df: Cleaned shot data DataFrame
        metrics_df: Cleaned team metrics DataFrame with 'win_pct' column

    Returns:
        DataFrame with team, season, three_pt_rate, win_pct columns
        ready for correlation analysis
    """
    rates = three_point_rate_by_team_season(shot_df)

    merged = rates.merge(
        metrics_df[["team", "season", "win_pct"]],
        on=["team", "season"],
        how="inner"
    )

    correlation = merged[["three_pt_rate", "win_pct"]].corr().iloc[0, 1]
    print(f"Correlation between three point rate and win pct: {correlation:.3f}")

    return merged


# ── Summary Table ───────────────────────────────────────────────────────────

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a master summary table with one row per team per season
    containing key shot selection and efficiency metrics.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        Summary DataFrame with one row per team-season
    """
    rates = three_point_rate_by_team_season(df)
    efg   = compute_efg(df, group_by=["team", "season"])
    pps   = compute_points_per_shot(df, group_by=["team", "season"])

    summary = rates.merge(efg,  on=["team", "season"], how="inner")
    summary = summary.merge(pps, on=["team", "season"], how="inner")

    return summary.sort_values(["team", "season"])