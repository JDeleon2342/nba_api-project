import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="NBA Shot Story", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("shot_data_clean.csv")

df = load_data()

players = sorted(df['player_name'].unique())

# -----------------------------
# Story Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Introduction",
    "2. Shot Selection Patterns",
    "3. Shot Efficiency & Heatmaps",
    "4. Player Comparison"
])

# -----------------------------
# TAB 1 — INTRODUCTION
# -----------------------------
with tab1:
    st.title("NBA Shot Selection Story")
    st.header("1. Introduction & Motivation")

    st.write("""
    This dashboard explores NBA shot selection using play-by-play shot data.
    The goal is to understand **where players shoot**, **what types of shots they take**, 
    and **how efficient they are** from different locations.

    This story walks through:
    - How shot selection varies by player  
    - Where players take the most shots  
    - Which areas of the court are most efficient  
    - How two players compare in style and accuracy  
    """)

    st.write("Use the tabs above to navigate through the story.")

# -----------------------------
# TAB 2 — SHOT SELECTION PATTERNS
# -----------------------------
with tab2:
    st.header("2. Shot Selection Patterns")

    selected_player = st.selectbox("Choose a player", players)

    filtered = df[df['player_name'] == selected_player]

    st.subheader(f"Shot Chart for {selected_player}")

    chart = (
        alt.Chart(filtered)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X("loc_x:Q", scale=alt.Scale(domain=[-250, 250])),
            y=alt.Y("loc_y:Q", scale=alt.Scale(domain=[-50, 900])),
            color="shot_made_flag:N",
            tooltip=["shot_distance", "shot_type", "shot_made_flag"]
        )
        .properties(height=600)
    )

    st.altair_chart(chart, use_container_width=True)

    st.write("""
    **Interpretation:**  
    This chart shows where the player tends to shoot from. 
    Clusters indicate preferred shooting zones, while color shows makes vs misses.
    """)

# -----------------------------
# TAB 3 — SHOT EFFICIENCY & HEATMAPS
# -----------------------------
with tab3:
    st.header("3. Shot Efficiency & Heatmaps")

    selected_player2 = st.selectbox("Choose a player for efficiency analysis", players)

    filtered2 = df[df['player_name'] == selected_player2]

    st.subheader(f"Shot Efficiency for {selected_player2}")

    heatmap = (
        alt.Chart(filtered2)
        .transform_bin(["xbin", "ybin"], ["loc_x", "loc_y"])
        .transform_aggregate(
            made="mean(shot_made_flag)",
            count="count()",
            groupby=["xbin", "ybin"]
        )
        .mark_rect()
        .encode(
            x="xbin:Q",
            y="ybin:Q",
            color=alt.Color("made:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["made", "count"]
        )
        .properties(height=600)
    )

    st.altair_chart(heatmap, use_container_width=True)

    st.write("""
    **Interpretation:**  
    Green areas show high-efficiency zones.  
    Red areas show low-efficiency zones.  
    This helps identify strengths and weaknesses in shot selection.
    """)

# -----------------------------
# TAB 4 — PLAYER COMPARISON
# -----------------------------
with tab4:
    st.header("4. Player Comparison")

    col1, col2 = st.columns(2)

    with col1:
        p1 = st.selectbox("Player 1", players, key="p1")

    with col2:
        p2 = st.selectbox("Player 2", players, key="p2")

    df1 = df[df['player_name'] == p1]
    df2 = df[df['player_name'] == p2]

    st.subheader("Shot Volume Comparison")

    vol_chart = (
        alt.Chart(pd.concat([df1.assign(player=p1), df2.assign(player=p2)]))
        .mark_bar()
        .encode(
            x="player:N",
            y="count():Q",
            color="player:N"
        )
    )

    st.altair_chart(vol_chart, use_container_width=True)

    st.write("""
    **Interpretation:**  
    This comparison highlights differences in shot volume and style between two players.
    """)

