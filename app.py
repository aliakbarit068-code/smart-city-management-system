"""Interactive Streamlit dashboard for the AI-based smart city project."""
import re

import pandas as pd
import streamlit as st

import data
import m1_agents
import m2_search
import m3_adversarial as m3
import m4_informed as m4
import m5_fuzzy as m5
import m6_ml
import m8_rl
import m9_cv_robotics as m9
import m10_datascience as m10
from common import city_graph


st.set_page_config(
    page_title="Smart City Control Center",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17263c;
        --muted: #64748b;
        --accent: #0e8a82;
        --panel: #ffffff;
    }
    .stApp {
        background: #f3f6fa;
        color: var(--ink);
    }
    .main .block-container {
        max-width: 1440px;
        padding: 1.5rem 1.5rem 2.5rem;
    }
    [data-testid="stHeader"] {
        background: rgba(243, 246, 250, 0.88);
    }
    [data-testid="stSidebar"] {
        background: #13233b;
    }
    [data-testid="stSidebar"] * {
        color: #e6edf7;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #1b304d;
        border: 1px solid #2d4564;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff;
    }
    .hero {
        background: linear-gradient(120deg, #10243d 0%, #145b67 58%, #0e8a82 100%);
        border-radius: 22px;
        color: white;
        margin: 0 0 1.5rem;
        padding: 2rem 2.2rem;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 50%;
        right: 7%;
        top: -110px;
        box-shadow: 0 0 0 28px rgba(255,255,255,.035),
                    0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker {
        color: #9de5d9;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: white;
        font-size: clamp(2rem, 3.2vw, 3rem);
        letter-spacing: -.04em;
        line-height: 1.1;
        margin: .55rem 0;
    }
    .hero p {
        color: #d3e2ed;
        font-size: 1rem;
        margin: 0;
        max-width: 650px;
    }
    .section-heading {
        margin: .2rem 0 1rem;
    }
    .section-heading h2 {
        color: #17263c;
        font-size: 1.55rem;
        letter-spacing: -.025em;
        margin: 0;
    }
    .section-heading p {
        color: #64748b;
        margin: .25rem 0 0;
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid #e1e8f0;
        border-radius: 15px;
        box-shadow: 0 5px 18px rgba(23, 38, 60, .045);
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: .84rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 720;
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: .8rem;
        border-bottom: 1px solid #dce5ee;
    }
    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 10px 10px 0 0;
        color: #53647b;
        font-weight: 650;
        padding: .7rem 1.05rem;
        white-space: nowrap;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #e2f3f1;
        color: #08776f;
    }
    [data-testid="stDataFrame"], [data-testid="stImage"] {
        border-radius: 14px;
        overflow: hidden;
    }
    div.stButton > button {
        background: #087f78;
        border: 1px solid #087f78;
        border-radius: 10px;
        color: #ffffff;
        font-weight: 650;
        min-height: 2.7rem;
    }
    div.stButton > button:hover {
        background: #066b65;
        border-color: #066b65;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_city_data():
    return m10.preprocess(data.load())


@st.cache_data(show_spinner="Training traffic models...")
def traffic_model_results(frame):
    svm_result = m6_ml.train_svm(frame)
    _, clusters = m6_ml.kmeans_patterns(frame)
    return svm_result, clusters


def answer_traffic_question(question, frame):
    """Answer common traffic questions using only the currently loaded dataset."""
    query = question.casefold()
    filtered = frame
    filters = []

    zones = sorted(frame["zone"].dropna().unique(), key=len, reverse=True)
    zone = next((name for name in zones if name.casefold() in query), None)
    if zone:
        filtered = filtered[filtered["zone"] == zone]
        filters.append(f"zone {zone}")

    weather_types = sorted(frame["weather"].dropna().unique(), key=len, reverse=True)
    weather = next((name for name in weather_types if name.casefold() in query), None)
    if weather:
        filtered = filtered[filtered["weather"] == weather]
        filters.append(f"{weather} weather")

    traffic_levels = ("High", "Medium", "Low")
    level = next(
        (name for name in traffic_levels if re.search(rf"\b{name.casefold()}\b", query)),
        None,
    )
    if level:
        filtered = filtered[filtered["traffic_level"] == level]
        filters.append(f"{level} traffic")

    hour_match = re.search(r"\b(?:hour|at)\s*(\d{1,2})\b", query)
    if hour_match:
        hour = int(hour_match.group(1))
        if not 0 <= hour <= 23:
            return "Please specify an hour from 0 to 23."
        filtered = filtered[filtered["hour"] == hour]
        filters.append(f"hour {hour}:00")

    zone_ranking_question = "zone" in query and any(
        phrase in query
        for phrase in ("busiest", "most traffic", "highest", "most vehicles")
    )
    if zone_ranking_question and "occupancy" not in query and "speed" not in query:
        zone_means = filtered.groupby("zone")["vehicle_count"].mean().sort_values(ascending=False)
        if zone_means.empty:
            return "No traffic records match those filters."
        busiest = zone_means.index[0]
        return (
            f"{busiest} has the highest average vehicle count "
            f"({zone_means.iloc[0]:.1f} vehicles per record)"
            f"{' for ' + ', '.join(filters) if filters else ''}."
        )

    if "weather" in query and not weather:
        if filtered.empty:
            return "No traffic records match those filters."
        weather_summary = (
            filtered.groupby("weather")
            .agg(records=("weather", "size"), avg_vehicles=("vehicle_count", "mean"))
            .sort_values("avg_vehicles", ascending=False)
        )
        lines = [
            f"{condition}: {int(row['records']):,} records, "
            f"{row['avg_vehicles']:.1f} average vehicles"
            for condition, row in weather_summary.iterrows()
        ]
        return "Weather summary (highest average volume first):\n\n" + "\n".join(
            f"- {line}" for line in lines
        )

    if "peak" in query or "rush hour" in query:
        hourly = (
            filtered.groupby("hour")["vehicle_count"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
        )
        if hourly.empty:
            return "No traffic records match those filters."
        peaks = ", ".join(
            f"{int(hour)}:00 ({volume:.1f} vehicles)" for hour, volume in hourly.items()
        )
        return f"The busiest hours are {peaks}."

    if "traffic condition" in query or "traffic level" in query:
        counts = filtered["traffic_level"].value_counts()
        return (
            f"Traffic conditions in the matching data: "
            f"{int(counts.get('Low', 0)):,} Low, "
            f"{int(counts.get('Medium', 0)):,} Medium, "
            f"{int(counts.get('High', 0)):,} High records."
        )

    if "vehicle" in query or "volume" in query:
        if any(word in query for word in ("average", "avg", "mean")):
            if filtered.empty:
                return "No traffic records match those filters."
            return (
                f"Average vehicle count is {filtered['vehicle_count'].mean():.1f}"
                f"{' for ' + ', '.join(filters) if filters else ' across the dataset'}."
            )
        if any(word in query for word in ("how many", "total", "count")):
            if filtered.empty:
                return "No traffic records match those filters."
            return (
                f"There are {int(filtered['vehicle_count'].sum()):,} total vehicles"
                f"{' for ' + ', '.join(filters) if filters else ' across the dataset'}."
            )

    if filters and any(word in query for word in ("how many", "count", "records", "rows")):
        return f"There are {len(filtered):,} matching records for {', '.join(filters)}."

    if "occupancy" in query or "density" in query:
        if filtered.empty:
            return "No traffic records match those filters."
        return (
            f"Average occupancy is {filtered['occupancy'].mean():.1%}"
            f"{' for ' + ', '.join(filters) if filters else ' across the dataset'}."
        )

    if "speed" in query:
        if filtered.empty:
            return "No traffic records match those filters."
        return (
            f"Average speed is {filtered['avg_speed'].mean():.1f}"
            f"{' for ' + ', '.join(filters) if filters else ' across the dataset'}."
        )

    if filters:
        if filtered.empty:
            return f"No records match {', '.join(filters)}."
        common_level = filtered["traffic_level"].mode().iloc[0]
        return (
            f"Found {len(filtered):,} records for {', '.join(filters)}. "
            f"Average vehicles: {filtered['vehicle_count'].mean():.1f}; "
            f"average occupancy: {filtered['occupancy'].mean():.1%}; "
            f"most common traffic condition: {common_level}."
        )

    if any(word in query for word in ("summary", "dataset", "total", "how many records")):
        return (
            f"The dataset has {len(frame):,} records across {frame['zone'].nunique()} zones. "
            f"Average vehicle count is {frame['vehicle_count'].mean():.1f}; "
            f"average occupancy is {frame['occupancy'].mean():.1%}."
        )

    return (
        "I can answer questions about zones, weather, traffic conditions, vehicle counts, "
        "occupancy, speed, and peak hours. Try: “Which zone is busiest?”, "
        "“How many High traffic records are there?”, or “Compare weather conditions.”"
    )


@st.cache_data(show_spinner=False)
def compare_empty_board():
    return m3.compare(" " * 9)


df = load_city_data()
summary = m10.summary(df)
record_columns = [
    "timestamp",
    "zone",
    "hour",
    "day_of_week",
    "weather",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "traffic_level",
]

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AI-powered operations dashboard</div>
      <h1>Smart City Control Center</h1>
      <p>Monitor traffic patterns, compare routes, tune signals, and explore
      intelligent city systems from one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### CITY SYSTEMS")
    st.caption("Live interactive project dashboard")
    st.metric("Traffic records", f"{len(df):,}")
    st.metric("Data source", "Synthetic")
    st.markdown("---")
    st.caption("Use the sections across the top to explore each AI module.")

tabs = st.tabs(
    [
        "Overview",
        "Traffic data",
        "Route planner",
        "Signal control",
        "Traffic AI",
        "Q-learning",
        "Vision & robotics",
        "Games",
    ]
)

with tabs[0]:
    st.markdown(
        '<div class="section-heading"><h2>City at a glance</h2>'
        '<p>Traffic indicators and model performance across the generated dataset.</p></div>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(2)
    metrics[0].metric("Vehicles counted", f"{summary['total_vehicles']:,}")
    metrics[1].metric("Average density", f"{summary['avg_density']:.1%}")
    details = st.columns(2)
    details[0].metric(
        "Peak traffic hours", ", ".join(f"{hour}:00" for hour in summary["peak_hours"])
    )
    details[1].metric("Busiest zone", summary["busiest_zone"])

    chart_col, model_col = st.columns((1.35, 1))
    with chart_col:
        st.markdown("#### Hourly traffic volume")
        hourly = df.groupby("hour")["vehicle_count"].mean().rename("Average vehicles")
        st.area_chart(hourly, color="#0e8a82", height=310)
    with model_col:
        st.markdown("#### Model performance")
        st.dataframe(m10.model_performance(), hide_index=True, use_container_width=True)
        st.caption("Models are evaluated on a held-out portion of the synthetic traffic data.")

    with st.expander("Preview traffic.csv records", expanded=True):
        st.dataframe(
            df[record_columns].head(12),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Use the Traffic data tab to filter and download the complete dataset.")

with tabs[1]:
    st.markdown(
        '<div class="section-heading"><h2>Traffic data explorer</h2>'
        '<p>Inspect individual CSV records and filter by location, conditions, and time.</p></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "traffic_level is derived from generated traffic load: Low < 0.35, "
        "Medium < 0.70, otherwise High."
    )
    filters = st.columns(2)
    selected_zone = filters[0].selectbox(
        "Zone",
        ["All zones"] + sorted(df["zone"].unique().tolist()),
    )
    selected_weather = filters[1].selectbox(
        "Weather",
        ["All weather"] + sorted(df["weather"].unique().tolist()),
    )
    condition_filters = st.columns(2)
    selected_level = condition_filters[0].selectbox(
        "Traffic condition",
        ["All levels", "Low", "Medium", "High"],
    )
    first_hour, last_hour = condition_filters[1].slider(
        "Hour range",
        min_value=0,
        max_value=23,
        value=(0, 23),
    )

    filtered = df
    if selected_zone != "All zones":
        filtered = filtered[filtered["zone"] == selected_zone]
    if selected_weather != "All weather":
        filtered = filtered[filtered["weather"] == selected_weather]
    if selected_level != "All levels":
        filtered = filtered[filtered["traffic_level"] == selected_level]
    filtered = filtered[filtered["hour"].between(first_hour, last_hour)]

    info = st.columns(2)
    info[0].metric("Matching records", f"{len(filtered):,}")
    info[1].metric(
        "Average vehicles",
        f"{filtered['vehicle_count'].mean():.1f}" if not filtered.empty else "—",
    )
    details = st.columns(2)
    details[0].metric(
        "Average occupancy",
        f"{filtered['occupancy'].mean():.1%}" if not filtered.empty else "—",
    )
    high_share = (filtered["traffic_level"] == "High").mean() if not filtered.empty else 0
    details[1].metric("High-traffic share", f"{high_share:.1%}")
    st.dataframe(
        filtered[record_columns],
        hide_index=True,
        use_container_width=True,
        height=440,
    )
    level_counts = (
        filtered["traffic_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )
    st.markdown("#### Traffic condition counts")
    st.bar_chart(level_counts, color="#0e8a82", height=220)
    st.download_button(
        "Download filtered records as CSV",
        data=filtered[record_columns].to_csv(index=False).encode("utf-8"),
        file_name="traffic_filtered.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )

with tabs[2]:
    st.markdown(
        '<div class="section-heading"><h2>Route planner</h2>'
        '<p>Compare uninformed and heuristic search on the city road network.</p></div>',
        unsafe_allow_html=True,
    )
    graph = city_graph()
    controls = st.columns((1, 1, 1.4))
    source = controls[0].selectbox("Starting point", list(graph), 0)
    destination = controls[1].selectbox("Destination", list(graph), 6)
    stops = controls[2].multiselect(
        "Delivery stops",
        list(graph),
        ["A", "C", "E", "F", "G"],
    )

    routes = pd.DataFrame(m4.compare_algorithms(source, destination))
    st.dataframe(
        routes[["algorithm", "path", "cost", "nodes_expanded", "optimal"]],
        hide_index=True,
        use_container_width=True,
    )
    if len(stops) >= 3:
        distances = m4.dist_all(graph)
        greedy, astar = st.columns(2)
        greedy_path, greedy_cost = m4.tsp_greedy(stops, distances, stops[0])
        astar_path, astar_cost = m4.tsp_astar(stops, distances, stops[0])
        greedy.metric("Greedy delivery loop", f"{greedy_cost} min")
        greedy.caption(" → ".join(greedy_path))
        astar.metric("A* delivery loop", f"{astar_cost} min")
        astar.caption(" → ".join(astar_path))

with tabs[3]:
    st.markdown(
        '<div class="section-heading"><h2>Adaptive signal control</h2>'
        '<p>Explore fuzzy signal timing and rule-based response to incidents.</p></div>',
        unsafe_allow_html=True,
    )
    controls, result = st.columns((1, 1.2))
    with controls:
        density = st.slider("Traffic density", 0.0, 1.0, 0.5, 0.05)
        waiting = st.slider("Waiting time (seconds)", 0, 120, 40, 5)
        emergency = st.checkbox("Emergency vehicle present")
    green_time = m5.fuzzy_green(density, waiting, float(emergency))
    with result:
        st.metric("Recommended green signal", f"{green_time:.1f} sec")
        perception = {"density": density, "emergency": emergency}
        reflex = m1_agents.SimpleReflexAgent().act(perception)
        utility = m1_agents.UtilityBasedAgent().act(perception)
        st.write(f"**Reflex agent:** {reflex}")
        st.write(f"**Utility agent:** {utility} sec")

    facts = st.multiselect(
        "Expert-system observations",
        ["accident", "rain", "high_density", "low_density", "long_wait", "emergency_present"],
        ["accident", "high_density"],
    )
    known_facts, fired_rules = m5.forward_chain(set(facts))
    with st.expander("View facts and fired rules", expanded=True):
        st.write("**Inferred actions:**", ", ".join(sorted(known_facts - set(facts))) or "None")
        st.write(fired_rules or "No rules fired.")

with tabs[4]:
    st.markdown(
        '<div class="section-heading"><h2>Traffic intelligence</h2>'
        '<p>Classify congestion and discover recurring traffic patterns.</p></div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Training traffic models..."):
        svm_result, clusters = traffic_model_results(df)
    model_metric, cluster_table = st.columns((1, 1.5))
    model_metric.metric("SVM test accuracy", f"{svm_result['accuracy']:.1%}")
    model_metric.markdown("#### Classification report")
    model_metric.code(svm_result["report"], language="text")
    cluster_table.markdown("#### Discovered traffic patterns")
    cluster_table.dataframe(clusters, use_container_width=True)

    st.markdown("#### Ask the Traffic AI assistant")
    st.caption(
        "This local chatbot answers from traffic.csv only; it does not send your data to an external AI service."
    )
    st.caption(
        "Try: “Which zone is busiest?”, “How many High traffic records are there?”, "
        "“Compare weather conditions.”, or “What are the peak hours?”"
    )
    if "traffic_ai_messages" not in st.session_state:
        st.session_state["traffic_ai_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Ask me about traffic zones, weather, congestion levels, "
                    "vehicle counts, occupancy, speed, or peak hours."
                ),
            }
        ]
    for message in st.session_state["traffic_ai_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about the traffic data...",
        key="traffic_ai_chat_input",
    )
    if question:
        st.session_state["traffic_ai_messages"].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)
        answer = answer_traffic_question(question, df)
        st.session_state["traffic_ai_messages"].append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)

with tabs[5]:
    st.markdown(
        '<div class="section-heading"><h2>Reinforcement learning</h2>'
        '<p>Train a Q-learning signal policy and compare it with fixed-time control.</p></div>',
        unsafe_allow_html=True,
    )
    st.info("Training runs when you press the button; it may take a few seconds.")
    if st.button("Train Q-learning agent", type="primary"):
        q_values, rewards = m8_rl.train()
        performance = m8_rl.compare(q_values)
        result_cols = st.columns(2)
        result_cols[0].metric("Learned reward", f"{float(performance['learned_reward']):.1f}")
        result_cols[1].metric("Fixed-time reward", f"{float(performance['fixed_reward']):.1f}")
        st.markdown("#### Training reward")
        st.line_chart(rewards, color="#0e8a82", height=300)
        st.json({key: float(value) for key, value in performance.items()})

with tabs[6]:
    st.markdown(
        '<div class="section-heading"><h2>Vision & emergency robotics</h2>'
        '<p>Detect vehicles in a generated road scene and test robot replanning.</p></div>',
        unsafe_allow_html=True,
    )
    vehicle_count = st.slider("Vehicles in scene", 4, 16, 12)
    image, truth = m9.make_traffic_image(n=vehicle_count)
    detected_image, detections = m9.detect_vehicles(image)
    image_col, stats_col = st.columns((1.4, 1))
    image_col.image(
        m9.to_rgb(detected_image),
        caption=f"{len(detections)} vehicles detected",
        use_container_width=True,
    )
    from collections import Counter

    stats_col.markdown("#### Detection summary")
    stats_col.markdown(f"**Generated:** `{dict(Counter(truth))}`")
    stats_col.markdown(
        f"**Detected:** `{dict(Counter(kind for kind, _ in detections))}`"
    )
    robot = m9.robot_run(m9.make_grid(), (0, 0), (14, 14))
    stats_col.markdown("#### Emergency robot")
    stats_col.metric("Goal reached", "Yes" if robot["reached"] else "No")
    stats_col.metric("Replans", robot["replans"])
    st.pyplot(m9.plot_robot_figure(robot), use_container_width=True)

with tabs[7]:
    st.markdown(
        '<div class="section-heading"><h2>Adversarial search games</h2>'
        '<p>See how alpha-beta pruning reduces the work needed for a best move.</p></div>',
        unsafe_allow_html=True,
    )
    game_result = compare_empty_board()
    minimax, alphabeta = st.columns(2)
    minimax.metric("Minimax nodes", f"{game_result['minimax'][2]:,}")
    alphabeta.metric("Alpha-beta nodes", f"{game_result['alphabeta'][2]:,}")
    traffic_game, vacuum = st.columns(2)
    traffic_game.markdown("#### Traffic signal game")
    traffic_game.json(m3.traffic_game_tree())
    vacuum.markdown("#### Vacuum world plan")
    vacuum.write(m2_search.vacuum_world()[1])
