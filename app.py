import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_collection import load_shot_data, load_dataset
from src.analysis import (
    zone_distribution,
    three_point_rate_by_season,
    three_point_rate_by_team_season,
    compare_eras,
    biggest_zone_shifts,
    rank_teams_by_three_point_adoption,
    correlate_three_pt_rate_with_wins,
    build_summary_table,
    team_shot_profile
)
from src.visualization import (
    plot_court_heatmap,
    plot_era_heatmap_comparison,
    plot_three_point_trend,
    plot_zone_distribution_over_time,
    plot_zone_shift_bar,
    plot_team_three_point_trajectory,
    plot_three_pt_vs_wins,
    plot_top_three_point_adopters
)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NBA Shot Selection Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    h1 { color: #e63946; }
    h2 { color: #a8dadc; }
    h3 { color: #ffffff; }
    .stMetric label { color: #a8dadc; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    """
    Assemble shot data from 4 part files and load team metrics.
    Cached so reassembly only happens once per session.
    """
    with st.spinner("Assembling shot data from part files..."):
        shots = load_shot_data()

    with st.spinner("Loading team metrics..."):
        metrics = load_dataset("data/team_metrics_clean.csv")

    return shots, metrics


shots, metrics = load_data()


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/7/7a/Basketball.png",
    width=80
)
st.sidebar.title("🏀 NBA Shot Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📈 League Trends",
        "🗺️ Court Heatmaps",
        "🏆 Team Analysis",
        "📊 Shot Zones",
        "🔗 Efficiency & Wins"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"Seasons: `{shots['season'].min()}` — `{shots['season'].max()}`")
st.sidebar.markdown(f"Total Shots: `{len(shots):,}`")
st.sidebar.markdown(f"Teams: `{shots['team'].nunique()}`")
st.sidebar.markdown(f"Zones: `{shots['zone_basic'].nunique()}`")


# ── Page: Overview ───────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("🏀 NBA Shot Selection Evolution")
    st.markdown(
        "### How has NBA shot selection evolved over the past decade, "
        "and which teams gained the greatest competitive advantage?"
    )
    
    st.markdown("---")

    # Key metrics row
    seasons    = sorted(shots["season"].unique())
    first_s    = seasons[0]
    last_s     = seasons[-1]
    first_3pt  = shots[shots["season"] == first_s]["is_three"].mean() * 100
    last_3pt   = shots[shots["season"] == last_s]["is_three"].mean() * 100
    delta_3pt  = last_3pt - first_3pt

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shot Attempts",      f"{len(shots):,}")
    col2.metric("Seasons Covered",          f"{len(seasons)}")
    col3.metric(f"3PT Rate {first_s}",      f"{first_3pt:.1f}%")
    col4.metric(f"3PT Rate {last_s}",       f"{last_3pt:.1f}%",
                delta=f"+{delta_3pt:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Three Point Rate Over Time")
        fig = plot_three_point_trend(shots)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Biggest Zone Shifts")
        fig = plot_zone_shift_bar(shots)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Shot Location by Era")
    fig = plot_era_heatmap_comparison(shots)
    st.pyplot(fig)
    plt.close()


# ── Page: League Trends ──────────────────────────────────────────────────────

elif page == "📈 League Trends":
    st.title("📈 League Wide Trends")
    st.markdown("---")

    st.subheader("Three Point Attempt Rate Over Time")
    fig = plot_three_point_trend(shots)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Shot Zone Distribution Over Time")
    fig = plot_zone_distribution_over_time(shots)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Zone Shift: First vs Most Recent Season")

    col1, col2 = st.columns(2)

    with col1:
        fig = plot_zone_shift_bar(shots)
        st.pyplot(fig)
        plt.close()

    with col2:
        shifts = biggest_zone_shifts(shots)
        st.dataframe(
            shifts.rename(columns={
                "zone_basic": "Zone",
                "pct_first":  f"% in {shots['season'].min()}",
                "pct_last":   f"% in {shots['season'].max()}",
                "change":     "Change (%)"
            }).round(2),
            use_container_width=True,
            hide_index=True
        )


# ── Page: Court Heatmaps ─────────────────────────────────────────────────────

elif page == "🗺️ Court Heatmaps":
    st.title("🗺️ Court Heatmaps")
    st.markdown("Explore where shots are taken by team and season.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        selected_team = st.selectbox(
            "Select Team",
            sorted(shots["team"].unique())
        )
    with col2:
        selected_season = st.selectbox(
            "Select Season",
            sorted(shots["season"].unique(), reverse=True)
        )

    fig = plot_court_heatmap(
        shots,
        team=selected_team,
        season=selected_season
    )
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Era Comparison — Shot Locations")
    fig = plot_era_heatmap_comparison(shots)
    st.pyplot(fig)
    plt.close()


# ── Page: Team Analysis ──────────────────────────────────────────────────────

elif page == "🏆 Team Analysis":
    st.title("🏆 Team Analysis")
    st.markdown("---")

    st.subheader("Three Point Rate Trajectory")
    st.markdown("Compare three point adoption across teams over time.")

    selected_teams = st.multiselect(
        "Select Teams",
        sorted(shots["team"].unique()),
        default=["Houston Rockets", "San Antonio Spurs", "Golden State Warriors"]
    )

    if selected_teams:
        fig = plot_team_three_point_trajectory(shots, teams=selected_teams)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Please select at least one team.")

    st.markdown("---")
    st.subheader("Top Three Point Adopters")

    top_n = st.slider(
        "Number of teams to show",
        min_value=5, max_value=30, value=10
    )
    fig = plot_top_three_point_adopters(shots, top_n=top_n)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Team Shot Zone Profile")

    profile_team = st.selectbox(
        "Select Team",
        sorted(shots["team"].unique()),
        key="profile_team"
    )

    profile_df = team_shot_profile(shots, profile_team)
    pivot = profile_df.pivot(
        index="season",
        columns="zone_basic",
        values="pct"
    ).fillna(0).round(2)

    st.dataframe(pivot, use_container_width=True)


# ── Page: Shot Zones ─────────────────────────────────────────────────────────

elif page == "📊 Shot Zones":
    st.title("📊 Shot Zone Breakdown")
    st.markdown("---")

    view = st.radio(
        "View by",
        ["League Wide", "By Team", "By Era"],
        horizontal=True
    )

    if view == "League Wide":
        season_filter = st.selectbox(
            "Select Season",
            ["All Seasons"] + sorted(shots["season"].unique(), reverse=True)
        )

        if season_filter == "All Seasons":
            dist = (
                shots.groupby("zone_basic")
                .size()
                .reset_index(name="count")
            )
            dist["pct"] = dist["count"] / dist["count"].sum() * 100
        else:
            filtered = shots[shots["season"] == season_filter]
            dist = (
                filtered.groupby("zone_basic")
                .size()
                .reset_index(name="count")
            )
            dist["pct"] = dist["count"] / dist["count"].sum() * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(dist["zone_basic"], dist["pct"], color="#e63946")
        ax.set_xlabel("Percentage of Shots (%)")
        ax.set_title(f"Shot Zone Distribution — {season_filter}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif view == "By Team":
        team = st.selectbox(
            "Select Team",
            sorted(shots["team"].unique())
        )
        fig = plot_zone_distribution_over_time(shots[shots["team"] == team])
        st.pyplot(fig)
        plt.close()

    elif view == "By Era":
        era_dist = compare_eras(shots)
        pivot = era_dist.pivot(
            index="era",
            columns="zone_basic",
            values="pct"
        ).fillna(0).round(2)

        st.dataframe(pivot, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.T.plot(kind="bar", ax=ax, colormap="viridis")
        ax.set_title("Shot Zone Distribution by Era")
        ax.set_xlabel("Zone")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Era", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── Page: Efficiency & Wins ──────────────────────────────────────────────────

elif page == "🔗 Efficiency & Wins":
    st.title("🔗 Shot Selection & Winning")
    st.markdown("Does shooting more threes correlate with more wins?")
    st.markdown("---")

    try:
        corr_df     = correlate_three_pt_rate_with_wins(shots, metrics)
        correlation = corr_df[["three_pt_rate", "win_pct"]].corr().iloc[0, 1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Correlation (3PT Rate vs Win%)", f"{correlation:.3f}")
        col2.metric("Team Seasons Analyzed",          f"{len(corr_df):,}")
        col3.metric(
            "Direction",
            "Positive ↑" if correlation > 0 else "Negative ↓"
        )

        st.markdown("---")
        fig = plot_three_pt_vs_wins(corr_df)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Full Summary Table")
        summary = build_summary_table(shots)
        st.dataframe(summary.round(3), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load efficiency data: {e}")
        st.info(
            "Make sure team_metrics_clean.csv is present in the data/ folder."
        )


# ── Footer ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**STAT 386 Final Project**  \n"
    "Jose De Leon & Nick Austin  \n"
    "Data: NBA Stats API"
)