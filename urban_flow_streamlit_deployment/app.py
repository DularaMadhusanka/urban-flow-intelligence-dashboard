
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Urban Flow Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #07111f;
        color: #e5edf8;
    }

    [data-testid="stSidebar"] {
        background-color: #0c1929;
        border-right: 1px solid #1f344d;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #10243a,
            #0b1929
        );
        border: 1px solid #23415f;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #9eb4ca;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #f2f7fd;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .business-card {
        background: linear-gradient(
            135deg,
            #10283e,
            #0c1c2e
        );
        border-left: 4px solid #22c7a9;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }

    .intro-card {
        background: linear-gradient(135deg, #12304a, #0c1c2e);
        border: 1px solid #2a526f;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 8px 0 18px 0;
        font-size: 1rem;
        line-height: 1.65;
    }

    .section-question {
        color: #9eb4ca;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    .small-note {
        color: #9eb4ca;
        font-size: 0.90rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dashboard_data():
    hourly = pd.read_parquet(
        BASE_DIR / "hourly_zone_demand.parquet"
    )

    zone_summary = pd.read_csv(
        BASE_DIR / "zone_summary.csv"
    )

    od_flows = pd.read_csv(
        BASE_DIR / "od_flows.csv"
    )

    forecast = pd.read_csv(
        BASE_DIR / "demand_forecast.csv"
    )

    model_metrics = pd.read_csv(
        BASE_DIR / "model_metrics.csv"
    )

    seed_metrics = pd.read_csv(
        BASE_DIR / "seed_stability.csv"
    )

    hourly["timestamp"] = pd.to_datetime(
        hourly["timestamp"]
    )

    hourly["pickup_date"] = pd.to_datetime(
        hourly["pickup_date"]
    )

    forecast["forecast_timestamp"] = pd.to_datetime(
        forecast["forecast_timestamp"]
    )

    return (
        hourly,
        zone_summary,
        od_flows,
        forecast,
        model_metrics,
        seed_metrics
    )


(
    hourly,
    zone_summary,
    od_flows,
    forecast,
    model_metrics,
    seed_metrics
) = load_dashboard_data()


# ============================================================
# SHARED CHART STYLE
# ============================================================

def style_figure(figure, height=420):
    figure.update_layout(
        height=height,
        paper_bgcolor="#0b1929",
        plot_bgcolor="#0b1929",
        font_color="#dce8f5",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text=""
    )

    figure.update_xaxes(
        gridcolor="#1c3148",
        zerolinecolor="#1c3148"
    )

    figure.update_yaxes(
        gridcolor="#1c3148",
        zerolinecolor="#1c3148"
    )

    return figure


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Choose what to explore")

st.sidebar.caption(
    "These filters update the historical demand and passenger-route views."
)

available_boroughs = sorted(
    hourly["borough"].dropna().unique().tolist()
)

selected_boroughs = st.sidebar.multiselect(
    "Pickup area",
    options=available_boroughs,
    default=available_boroughs
)

minimum_date = hourly["pickup_date"].min().date()
maximum_date = hourly["pickup_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Dates to review",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = minimum_date
    end_date = maximum_date

historical_filtered = hourly[
    hourly["borough"].isin(selected_boroughs)
    & (
        hourly["pickup_date"].dt.date >= start_date
    )
    & (
        hourly["pickup_date"].dt.date <= end_date
    )
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Historical period: May 2025\n\n"
    "Forecast horizon: 24 hours\n\n"
    "Selected model: GRU-GAT, seed 7"
)

with st.sidebar.expander("How to use this dashboard"):
    st.markdown(
        """
        1. Choose one or more pickup areas.
        2. Select the historical dates you want to examine.
        3. Review demand peaks and common passenger routes.
        4. Open **Fleet recommendations** to see where vehicles should be prioritised.

        The forecast page uses a fixed historical test period, so the date filter
        does not change its results.
        """
    )


# ============================================================
# TITLE
# ============================================================

st.title("Urban Flow Intelligence")

st.markdown(
    """
    <div class="intro-card">
    <b>Purpose:</b> Help taxi operations managers understand where and when
    passengers request rides, where they travel, and which zones should receive
    fleet attention during the next 24 forecast hours.<br><br>
    <b>Start here:</b> Use the filters on the left, review the overview, then open
    <b>Fleet recommendations</b> for clear positioning priorities.
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(
    [
        "Overview",
        "Demand patterns",
        "Passenger routes",
        "Fleet recommendations",
        "Forecast reliability"
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader("What is happening across the selected area?")
    st.markdown(
        '<p class="section-question">A management summary of passenger demand during the selected dates.</p>',
        unsafe_allow_html=True
    )

    if historical_filtered.empty:
        st.warning(
            "No historical records match the selected filters."
        )
    else:
        total_demand = int(
            historical_filtered["demand"].sum()
        )

        zones_analyzed = historical_filtered[
            "loc_id"
        ].nunique()

        city_hourly = (
            historical_filtered
            .groupby("timestamp", as_index=False)["demand"]
            .sum()
        )

        peak_row = city_hourly.loc[
            city_hourly["demand"].idxmax()
        ]

        filtered_zone_totals = (
            historical_filtered
            .groupby(
                [
                    "loc_id",
                    "borough",
                    "zone_name"
                ],
                as_index=False
            )["demand"]
            .sum()
            .sort_values(
                "demand",
                ascending=False
            )
        )

        top_zone = filtered_zone_totals.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total passenger pickups",
            f"{total_demand:,}",
            help=(
                "The number of valid taxi trips that began in the selected "
                "areas and dates."
            )
        )

        col2.metric(
            "Pickup zones covered",
            f"{zones_analyzed:,}",
            help="The number of taxi zones represented by the current selection."
        )

        col3.metric(
            "Busiest one-hour period",
            f"{int(peak_row['demand']):,} pickups",
            help="The highest combined pickup count recorded during one hour."
        )

        col4.metric(
            "Busiest pickup zone",
            top_zone["zone_name"],
            help="The zone with the most passenger pickups in the selected period."
        )

        left, right = st.columns([1.65, 1])

        with left:
            city_figure = px.line(
                city_hourly,
                x="timestamp",
                y="demand",
                title="Citywide Pickup Demand",
                labels={
                    "timestamp": "Time",
                    "demand": "Pickups"
                },
                color_discrete_sequence=["#27d7b2"]
            )

            city_figure.update_traces(
                line=dict(width=2)
            )

            st.plotly_chart(
                style_figure(city_figure),
                use_container_width=True
            )

        with right:
            top_zones = filtered_zone_totals.head(10)

            zone_figure = px.bar(
                top_zones.sort_values("demand"),
                x="demand",
                y="zone_name",
                orientation="h",
                title="Top 10 Demand Zones",
                labels={
                    "demand": "Pickups",
                    "zone_name": ""
                },
                color="demand",
                color_continuous_scale=[
                    "#1f5574",
                    "#22c7a9"
                ]
            )

            zone_figure.update_layout(
                coloraxis_showscale=False
            )

            st.plotly_chart(
                style_figure(zone_figure),
                use_container_width=True
            )

        st.caption(
            "How to read these charts: repeated peaks reveal recurring periods "
            "when vehicles should be available before demand rises. The zone "
            "ranking shows where that capacity is most likely to be needed."
        )

        st.subheader("What this means for operations")

        st.markdown(
            f"""
            <div class="business-card">
            Demand is concentrated around
            <b>{top_zone["zone_name"]}</b> in
            <b>{top_zone["borough"]}</b>, which generated
            <b>{int(top_zone["demand"]):,}</b> pickups within
            the selected period.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="business-card">
            The highest citywide activity occurred at
            <b>{peak_row["timestamp"]:%A, %d %B at %H:%M}</b>,
            with <b>{int(peak_row["demand"]):,}</b> pickups.
            Fleet availability should be reviewed before
            comparable recurring peak periods.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 2 — DEMAND PATTERNS
# ============================================================

with tabs[1]:

    st.subheader("Where and when is passenger demand highest?")
    st.markdown(
        '<p class="section-question">Select a zone to identify its recurring busy hours and quieter periods.</p>',
        unsafe_allow_html=True
    )

    zone_options = (
        historical_filtered[
            ["loc_id", "borough", "zone_name"]
        ]
        .drop_duplicates()
        .sort_values(["borough", "zone_name"])
    )

    if zone_options.empty:
        st.warning("No zones match the current filters.")
    else:
        selected_zone_name = st.selectbox(
            "Pickup zone",
            options=zone_options["zone_name"].tolist(),
            help="Choose one pickup zone to examine its hourly and weekly pattern."
        )

        selected_zone_id = zone_options.loc[
            zone_options["zone_name"] ==
            selected_zone_name,
            "loc_id"
        ].iloc[0]

        zone_data = historical_filtered[
            historical_filtered["loc_id"] ==
            selected_zone_id
        ].copy()

        zone_peak = zone_data.loc[zone_data["demand"].idxmax()]

        zone_line = px.line(
            zone_data,
            x="timestamp",
            y="demand",
            title=f"Hourly Demand — {selected_zone_name}",
            labels={
                "timestamp": "Time",
                "demand": "Pickups"
            },
            color_discrete_sequence=["#ffb547"]
        )

        st.plotly_chart(
            style_figure(zone_line),
            use_container_width=True
        )

        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        heatmap_data = (
            zone_data
            .groupby(
                ["day_name", "hour_of_day"],
                as_index=False
            )["demand"]
            .mean()
        )

        heatmap_pivot = (
            heatmap_data
            .pivot(
                index="day_name",
                columns="hour_of_day",
                values="demand"
            )
            .reindex(weekday_order)
        )

        heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale=[
                    [0.0, "#081421"],
                    [0.5, "#176b80"],
                    [1.0, "#31e6bd"]
                ],
                colorbar=dict(
                    title="Average<br>pickups"
                ),
                hovertemplate=(
                    "Day: %{y}<br>"
                    "Hour: %{x}:00<br>"
                    "Average pickups: %{z:.2f}"
                    "<extra></extra>"
                )
            )
        )

        heatmap.update_layout(
            title=f"Weekly Demand Profile — {selected_zone_name}",
            xaxis_title="Hour of day",
            yaxis_title=""
        )

        st.plotly_chart(
            style_figure(heatmap, height=460),
            use_container_width=True
        )

        st.markdown(
            f"""
            <div class="business-card">
            <b>Operational reading:</b> {selected_zone_name} reached its largest
            observed hourly demand on <b>{zone_peak['timestamp']:%A, %d %B at %H:%M}</b>,
            when <b>{int(zone_peak['demand']):,} pickups</b> were recorded. Darker or
            brighter heatmap cells identify recurring day-and-hour combinations
            where advance vehicle coverage may be useful.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 3 — OD INTELLIGENCE
# ============================================================

with tabs[2]:

    st.subheader("Where do passengers travel after pickup?")
    st.markdown(
        '<p class="section-question">Choose a starting zone to see its ten most common destinations and typical trip characteristics.</p>',
        unsafe_allow_html=True
    )

    filtered_od = od_flows[
        od_flows["origin_borough"].isin(
            selected_boroughs
        )
    ].copy()

    od_zone_options = sorted(
        filtered_od["origin_zone_name"]
        .dropna()
        .unique()
        .tolist()
    )

    if not od_zone_options:
        st.warning(
            "No OD routes match the selected boroughs."
        )
    else:
        selected_origin = st.selectbox(
            "Starting pickup zone",
            options=od_zone_options,
            help="The chart will show the most common destinations from this zone."
        )

        selected_routes = (
            filtered_od[
                filtered_od["origin_zone_name"] ==
                selected_origin
            ]
            .sort_values("trip_count", ascending=False)
            .head(10)
        )

        route_bar = px.bar(
            selected_routes.sort_values("trip_count"),
            x="trip_count",
            y="destination_zone_name",
            orientation="h",
            color="trip_count",
            color_continuous_scale=[
                "#2c3f65",
                "#ff9e45"
            ],
            title=f"Top Destinations from {selected_origin}",
            labels={
                "trip_count": "Trips",
                "destination_zone_name": ""
            },
            hover_data=[
                "destination_borough",
                "average_distance_miles",
                "average_duration_minutes",
                "average_total_charge"
            ]
        )

        route_bar.update_layout(
            coloraxis_showscale=False
        )

        st.plotly_chart(
            style_figure(route_bar, height=500),
            use_container_width=True
        )

        st.dataframe(
            selected_routes[
                [
                    "origin_flow_rank",
                    "destination_zone_name",
                    "destination_borough",
                    "trip_count",
                    "percentage_of_origin_trips",
                    "average_distance_miles",
                    "average_duration_minutes",
                    "average_total_charge"
                ]
            ].rename(
                columns={
                    "origin_flow_rank": "Route rank",
                    "destination_zone_name": "Destination",
                    "destination_borough": "Destination area",
                    "trip_count": "Trips",
                    "percentage_of_origin_trips": "Share of origin trips (%)",
                    "average_distance_miles": "Average distance (miles)",
                    "average_duration_minutes": "Average duration (minutes)",
                    "average_total_charge": "Average total charge"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        leading_route = selected_routes.iloc[0]
        st.markdown(
            f"""
            <div class="business-card">
            <b>Operational reading:</b> The most common destination from
            <b>{selected_origin}</b> is <b>{leading_route['destination_zone_name']}</b>
            with <b>{int(leading_route['trip_count']):,} trips</b>. This helps managers
            anticipate where vehicles tend to finish journeys and whether those
            destination areas are likely to need continued coverage or repositioning.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 4 — FORECAST AND DISPATCH
# ============================================================

with tabs[3]:

    st.subheader("Where should available vehicles be positioned next?")
    st.markdown(
        '<p class="section-question">Review one forecast hour at a time and prioritise zones with both high expected demand and positive growth.</p>',
        unsafe_allow_html=True
    )

    st.info(
        "Forecast demonstration: the predictions were produced using only "
        "information available before each forecast hour. Actual demand is shown "
        "afterward to evaluate performance. This is a historical backtest, not a "
        "live real-time forecast."
    )

    forecast_borough = forecast[
        forecast["borough"].isin(selected_boroughs)
    ].copy()

    forecast_times = sorted(
        forecast_borough[
            "forecast_timestamp"
        ].unique()
    )

    if not forecast_times:
        st.warning(
            "No forecast zones match the selected boroughs."
        )
    else:
        selected_forecast_time = st.select_slider(
            "Forecast hour",
            options=forecast_times,
            value=forecast_times[-1],
            format_func=lambda value: pd.Timestamp(
                value
            ).strftime("%A %d %b, %H:%M")
        )

        hour_forecast = forecast_borough[
            forecast_borough["forecast_timestamp"] ==
            selected_forecast_time
        ].copy()

        predicted_total = hour_forecast[
            "predicted_demand"
        ].sum()

        actual_total = hour_forecast[
            "actual_demand"
        ].sum()

        hour_wape = (
            100.0 *
            hour_forecast["absolute_error"].sum() /
            max(hour_forecast["actual_demand"].sum(), 1)
        )

        # Recalculate priority rank within the currently selected boroughs
        hour_forecast["selected_area_priority_rank"] = (
            hour_forecast["dispatch_priority_score"]
            .rank(
                method="first",
                ascending=False
            )
            .astype(int)
        )
        
        highest_priority = (
            hour_forecast
            .sort_values("selected_area_priority_rank")
            .iloc[0]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Expected passenger pickups",
            f"{predicted_total:,.0f}",
            help="The GRU-GAT estimate for all currently selected zones."
        )

        col2.metric(
            "Pickups actually observed",
            f"{actual_total:,.0f}",
            help="The known value is shown because this page is a historical backtest."
        )

        col3.metric(
            "Total error for this hour",
            f"{hour_wape:.1f}%",
            help="Total absolute forecast error as a percentage of observed demand. Lower is better."
        )

        col4.metric(
            "First zone to review",
            highest_priority["zone_name"],
            help="The highest relative positioning priority among selected zones."
        )

        top_forecasts = (
            hour_forecast
            .sort_values(
                "predicted_demand",
                ascending=False
            )
            .head(15)
            .sort_values("predicted_demand")
        )

        forecast_chart = go.Figure()

        forecast_chart.add_trace(
            go.Bar(
                name="Seasonal baseline",
                y=top_forecasts["zone_name"],
                x=top_forecasts[
                    "seasonal_naive_demand"
                ],
                orientation="h",
                marker_color="#536d8c"
            )
        )

        forecast_chart.add_trace(
            go.Bar(
                name="GRU-GAT forecast",
                y=top_forecasts["zone_name"],
                x=top_forecasts["predicted_demand"],
                orientation="h",
                marker_color="#22c7a9"
            )
        )

        forecast_chart.update_layout(
            title="Highest Forecast Demand Zones",
            barmode="group",
            xaxis_title="Expected pickups",
            yaxis_title=""
        )

        st.plotly_chart(
            style_figure(forecast_chart, height=580),
            use_container_width=True
        )

        st.subheader("Recommended fleet priorities")

        priority_table = (
            hour_forecast
            .sort_values("dispatch_priority_rank")
            .head(10)
        )

        def assign_priority_status(rank):
            if rank <= 10:
                return "High positioning priority"
            if rank <= 30:
                return "Monitor and prepare"
            return "Maintain normal coverage"

        priority_table = priority_table.copy()
        priority_table["recommended_action"] = priority_table[
            "dispatch_priority_rank"
        ].apply(assign_priority_status)

        st.dataframe(
            priority_table[
                [
                    "dispatch_priority_rank",
                    "borough",
                    "zone_name",
                    "predicted_demand",
                    "seasonal_naive_demand",
                    "predicted_growth",
                    "actual_demand",
                    "recommended_action"
                ]
            ].rename(
                columns={
                    "dispatch_priority_rank": "Priority rank",
                    "borough": "Area",
                    "zone_name": "Pickup zone",
                    "predicted_demand": "Expected pickups",
                    "seasonal_naive_demand": "Same hour previous day",
                    "predicted_growth": "Expected change",
                    "actual_demand": "Observed pickups",
                    "recommended_action": "Recommended action"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            <p class="small-note">
            The priority score combines 70% normalized
            forecast demand and 30% normalized positive
            growth relative to the corresponding previous-day
            hour. It is a decision-support ranking, not a
            guaranteed financial or fuel-saving estimate.
            </p>
            """,
            unsafe_allow_html=True
        )

        if rising_zones > 0:
            growth_message = (
                f"{rising_zones} selected zones are forecast to exceed "
                "their corresponding previous-day demand."
            )
        else:
            growth_message = (
                "No selected zones are forecast to exceed their "
                "corresponding previous-day demand. The recommendation "
                "therefore reflects relative demand within the selected area."
            )
        
        st.markdown(
            f"""
            <div class="business-card">
            <b>Decision summary for the selected backtest hour:</b>
            Review <b>{highest_priority['zone_name']}</b> first because it has
            the highest positioning-priority score among the currently selected
            areas. {growth_message}
            <br><br>
            Exact vehicle quantities require fleet availability, driver location
            and operating-cost data, which are not included here.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 5 — MODEL EVALUATION
# ============================================================

with tabs[4]:

    st.subheader("How reliable is the 24-hour demand forecast?")
    st.markdown(
        '<p class="section-question">Lower error values are better; a higher explained-demand percentage is better.</p>',
        unsafe_allow_html=True
    )

    selected_model = model_metrics[
        model_metrics["selected_model"] == True
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average error per zone-hour",
        f"{selected_model['mae']:.2f} trips",
        help="Mean Absolute Error: the typical absolute difference between predicted and observed pickups."
    )

    col2.metric(
        "Large-error-sensitive score",
        f"{selected_model['rmse']:.2f} trips",
        help="Root Mean Squared Error gives larger mistakes more influence. Lower is better."
    )

    col3.metric(
        "Total forecast error",
        f"{selected_model['wape_percent']:.2f}%",
        help="Total absolute error divided by total observed demand. Lower is better."
    )

    col4.metric(
        "Demand variation explained",
        f"{selected_model['r_squared'] * 100:.1f}%",
        help="The proportion of variation in demand explained by the model. Higher is better."
    )

    left, right = st.columns(2)

    with left:
        error_long = model_metrics.melt(
            id_vars="model",
            value_vars=["mae", "rmse"],
            var_name="metric",
            value_name="error"
        )

        error_chart = px.bar(
            error_long,
            x="model",
            y="error",
            color="metric",
            barmode="group",
            title="Average and Large-Error-Sensitive Scores",
            labels={
                "model": "",
                "error": "Trips",
                "metric": ""
            },
            color_discrete_map={
                "mae": "#22c7a9",
                "rmse": "#ff9e45"
            }
        )

        st.plotly_chart(
            style_figure(error_chart),
            use_container_width=True
        )

    with right:
        wape_chart = px.bar(
            model_metrics.sort_values(
                "wape_percent",
                ascending=False
            ),
            x="model",
            y="wape_percent",
            color="model",
            title="Total Forecast Error by Method",
            labels={
                "model": "",
                "wape_percent": "WAPE (%)"
            }
        )

        wape_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            style_figure(wape_chart),
            use_container_width=True
        )

    seed_chart = px.scatter(
        seed_metrics,
        x="seed",
        y="test_mae",
        size="test_rmse",
        color="test_wape_percent",
        color_continuous_scale=[
            "#22c7a9",
            "#ff9e45"
        ],
        title="Consistency Across Five Training Runs",
        labels={
            "seed": "Random seed",
            "test_mae": "Test MAE",
            "test_rmse": "Test RMSE",
            "test_wape_percent": "WAPE (%)"
        }
    )

    seed_chart.update_traces(
        marker=dict(
            line=dict(
                width=1,
                color="#ffffff"
            )
        )
    )

    st.plotly_chart(
        style_figure(seed_chart),
        use_container_width=True
    )

    st.info(
        "The selected GRU-GAT model averages about 6.42 trips of error per "
        "zone-hour and explains 87.3% of observed demand variation. Five "
        "independent training runs produced very similar results, supporting "
        "the stability of the conclusion."
    )

    with st.expander("Technical details and terminology"):
        st.markdown(
            """
            - **Seasonal baseline:** assumes demand will match the corresponding
              hour of the previous day.
            - **GRU:** learns how demand changes across the preceding 24 hours.
            - **Graph Attention Network (GAT):** learns how connected taxi zones
              influence one another through passenger flows.
            - **Seasonal-residual forecast:** starts from the previous-day value
              and learns a correction rather than predicting demand from scratch.
            - **MAE:** average absolute error per zone and forecast hour.
            - **RMSE:** an error measure that penalises large mistakes more strongly.
            - **WAPE:** total absolute error as a percentage of total demand.
            - **R²:** proportion of observed demand variation explained by the model.
            - **Five training runs:** tests whether performance is stable across
              different random initialisations.

            Seed 7 was selected using validation performance rather than test
            performance, preventing the test data from influencing model selection.
            """
        )
