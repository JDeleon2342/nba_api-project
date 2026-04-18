import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, Rectangle
import seaborn as sns
from analysis import (
    zone_distribution,
    three_point_rate_by_season,
    three_point_rate_by_team_season,
    compare_eras,
    biggest_zone_shifts,
    rank_teams_by_three_point_adoption
)


# ── Court Drawing ───────────────────────────────────────────────────────────

def draw_court(ax: plt.Axes, color: str = "black", lw: float = 1.5) -> plt.Axes:
    """
    Draw an NBA half court on a matplotlib Axes object.

    Args:
        ax: Matplotlib Axes to draw on
        color: Line color for court markings
        lw: Line width for court markings

    Returns:
        Axes with court drawn
    """
    # Hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Backboard
    backboard = patches.Rectangle((-30, -7.5), 60, 0,
                                   linewidth=lw, color=color)

    # Outer paint — free throw lane
    outer_box = patches.Rectangle((-80, -47.5), 160, 190,
                                   linewidth=lw, color=color, fill=False)

    # Inner paint
    inner_box = patches.Rectangle((-60, -47.5), 120, 190,
                                   linewidth=lw, color=color, fill=False)

    # Free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120,
                         theta1=0, theta2=180,
                         linewidth=lw, color=color)

    # Free throw bottom arc (dashed)
    bottom_free_throw = Arc((0, 142.5), 120, 120,
                            theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle="dashed")

    # Restricted area
    restricted = Arc((0, 0), 80, 80,
                     theta1=0, theta2=180,
                     linewidth=lw, color=color)

    # Three point line — corner segments
    corner_three_a = patches.Rectangle((-220, -47.5), 0, 140,
                                        linewidth=lw, color=color)
    corner_three_b = patches.Rectangle((220, -47.5), 0, 140,
                                        linewidth=lw, color=color)

    # Three point arc
    three_arc = Arc((0, 0), 475, 475,
                    theta1=22, theta2=158,
                    linewidth=lw, color=color)

    # Center court
    center_outer_arc = Arc((0, 422.5), 120, 120,
                           theta1=180, theta2=0,
                           linewidth=lw, color=color)

    court_elements = [
        hoop, backboard, outer_box, inner_box,
        top_free_throw, bottom_free_throw, restricted,
        corner_three_a, corner_three_b, three_arc,
        center_outer_arc
    ]

    for element in court_elements:
        ax.add_patch(element)

    return ax


# ── Court Heatmap ───────────────────────────────────────────────────────────

def plot_court_heatmap(
    df: pd.DataFrame,
    team: str = None,
    season: str = None,
    title: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Plot a shot location heatmap on an NBA half court.

    Args:
        df: Cleaned shot data DataFrame with 'x', 'y', 'team', 'season' columns
        team: Optional team name to filter by
        season: Optional season string to filter by
        title: Optional plot title
        ax: Optional existing Axes to draw on

    Returns:
        Matplotlib Figure
    """
    filtered = df.copy()
    if team:
        filtered = filtered[filtered["team"] == team]
    if season:
        filtered = filtered[filtered["season"] == season]

    if filtered.empty:
        raise ValueError("No data found for the given filters")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    else:
        fig = ax.get_figure()

    # Hexbin heatmap
    ax.hexbin(
        filtered["x"], filtered["y"],
        gridsize=40,
        cmap="YlOrRd",
        bins="log",
        extent=[-250, 250, -47.5, 422.5]
    )

    draw_court(ax, color="black", lw=1.5)

    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.set_facecolor("#1a1a2e")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plot_title = title or f"Shot Heatmap — {team or 'All Teams'} {season or 'All Seasons'}"
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=12)

    return fig


def plot_era_heatmap_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Plot side by side court heatmaps comparing shot locations
    across early, mid, and recent eras.

    Args:
        df: Cleaned shot data DataFrame with 'era' column

    Returns:
        Matplotlib Figure with three side by side heatmaps
    """
    eras = ["Early (2013-17)", "Mid (2017-21)", "Recent (2021-24)"]
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, era in zip(axes, eras):
        era_df = df[df["era"] == era]
        plot_court_heatmap(df=era_df, title=era, ax=ax)

    plt.suptitle(
        "NBA Shot Location Evolution by Era",
        fontsize=18, fontweight="bold", color="white", y=1.02
    )
    plt.tight_layout()
    return fig


# ── Trend Charts ────────────────────────────────────────────────────────────

def plot_three_point_trend(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the league wide three point attempt rate trend over all seasons.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        Matplotlib Figure
    """
    trend = three_point_rate_by_season(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        trend["season"], trend["three_pt_rate"],
        marker="o", linewidth=2.5, color="#e63946", markersize=8
    )
    ax.fill_between(
        trend["season"], trend["three_pt_rate"],
        alpha=0.15, color="#e63946"
    )

    ax.set_title("League Wide Three Point Attempt Rate Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Three Point Attempt Rate (%)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_zone_distribution_over_time(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a stacked area chart showing shot zone distribution
    across all seasons.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        Matplotlib Figure
    """
    dist = zone_distribution(df, group_by="season")
    pivot = dist.pivot(index="season", columns="zone_basic", values="pct").fillna(0)

    zone_colors = {
        "Restricted Area":          "#e63946",
        "In The Paint (Non-RA)":    "#f4a261",
        "Mid-Range":                "#2a9d8f",
        "Left Corner 3":            "#457b9d",
        "Right Corner 3":           "#1d3557",
        "Above the Break 3":        "#a8dadc",
    }

    colors = [zone_colors.get(z, "#999999") for z in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot.area(ax=ax, color=colors, alpha=0.85)

    ax.set_title("NBA Shot Zone Distribution Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Percentage of Shots (%)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_zone_shift_bar(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a horizontal bar chart showing which zones changed
    most between the first and last season.

    Args:
        df: Cleaned shot data DataFrame

    Returns:
        Matplotlib Figure
    """
    shifts = biggest_zone_shifts(df)

    colors = ["#e63946" if x > 0 else "#457b9d" for x in shifts["change"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(shifts["zone_basic"], shifts["change"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title("Shot Zone Change: First vs Most Recent Season",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Change in Shot Frequency (%)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ── Team Charts ─────────────────────────────────────────────────────────────

def plot_team_three_point_trajectory(
    df: pd.DataFrame,
    teams: list
) -> plt.Figure:
    """
    Plot three point attempt rate over time for a list of specific teams.

    Args:
        df: Cleaned shot data DataFrame
        teams: List of team name strings to include

    Returns:
        Matplotlib Figure
    """
    rates = three_point_rate_by_team_season(df)
    filtered = rates[rates["team"].isin(teams)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for team, group in filtered.groupby("team"):
        ax.plot(
            group["season"], group["three_pt_rate"],
            marker="o", linewidth=2, label=team
        )

    ax.set_title("Three Point Attempt Rate by Team Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Three Point Attempt Rate (%)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_three_pt_vs_wins(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a scatterplot of three point attempt rate vs win percentage
    across all team seasons.

    Args:
        df: DataFrame with 'three_pt_rate', 'win_pct', 'season', 'team' columns
            (output of correlate_three_pt_rate_with_wins)

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        df["three_pt_rate"], df["win_pct"],
        c=df["season"].astype("category").cat.codes,
        cmap="viridis", alpha=0.6, s=60
    )

    # Trend line
    z = np.polyfit(df["three_pt_rate"], df["win_pct"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["three_pt_rate"].min(), df["three_pt_rate"].max(), 100)
    ax.plot(x_line, p(x_line), color="red", linewidth=1.5,
            linestyle="--", label="Trend")

    plt.colorbar(scatter, ax=ax, label="Season")
    ax.set_title("Three Point Attempt Rate vs Win Percentage",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Three Point Attempt Rate (%)", fontsize=12)
    ax.set_ylabel("Win Percentage", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_top_three_point_adopters(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Plot a bar chart of the teams that increased their three point
    attempt rate the most over the dataset period.

    Args:
        df: Cleaned shot data DataFrame
        top_n: Number of top teams to show

    Returns:
        Matplotlib Figure
    """
    rankings = rank_teams_by_three_point_adoption(df).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        rankings["team"], rankings["change"],
        color="#e63946"
    )
    ax.set_title(f"Top {top_n} Teams by Three Point Rate Increase",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Change in Three Point Rate (%) — First to Last Season",
                  fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig