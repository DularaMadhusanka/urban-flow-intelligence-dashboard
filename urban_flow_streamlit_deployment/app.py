
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

st.sidebar.title("Dashboard Filters")

available_boroughs = sorted(
    hourly["borough"].dropna().unique().tolist()
)

selected_boroughs = st.sidebar.multiselect(
    "Origin borough",
    options=available_boroughs,
    default=available_boroughs
)

minimum_date = hourly["pickup_date"].min().date()
maximum_date = hourly["pickup_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Historical date range",
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


# ============================================================
# TITLE
# ============================================================

st.title("Urban Flow Intelligence")

st.caption(
    "Spatial-temporal taxi demand, origin–destination movement "
    "and data-driven fleet positioning"
)

tabs = st.tabs(
    [
        "Executive Overview",
        "Demand Patterns",
        "OD Intelligence",
        "Forecast & Dispatch",
        "Model Evaluation"
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tabs[0]:

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
            "Total pickups",
            f"{total_demand:,}"
        )

        col2.metric(
            "Zones analyzed",
            f"{zones_analyzed:,}"
        )

        col3.metric(
            "Peak hourly demand",
            f"{int(peak_row['demand']):,}"
        )

        col4.metric(
            "Highest-demand zone",
            top_zone["zone_name"]
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

        st.subheader("Management interpretation")

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
            "Select a zone",
            options=zone_options["zone_name"].tolist()
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


# ============================================================
# TAB 3 — OD INTELLIGENCE
# ============================================================

with tabs[2]:

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
            "Select an origin zone",
            options=od_zone_options
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
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 4 — FORECAST AND DISPATCH
# ============================================================

with tabs[3]:

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
            format_func=lambda value: pd.Timestamp(
                value
            ).strftime("%A %d May, %H:%M")
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

        highest_priority = (
            hour_forecast
            .sort_values("dispatch_priority_rank")
            .iloc[0]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Predicted pickups",
            f"{predicted_total:,.0f}"
        )

        col2.metric(
            "Observed pickups",
            f"{actual_total:,.0f}"
        )

        col3.metric(
            "Hourly WAPE",
            f"{hour_wape:.1f}%"
        )

        col4.metric(
            "Top dispatch priority",
            highest_priority["zone_name"]
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

        st.dataframe(
            priority_table[
                [
                    "dispatch_priority_rank",
                    "borough",
                    "zone_name",
                    "predicted_demand",
                    "seasonal_naive_demand",
                    "predicted_growth",
                    "actual_demand"
                ]
            ],
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


# ============================================================
# TAB 5 — MODEL EVALUATION
# ============================================================

with tabs[4]:

    selected_model = model_metrics[
        model_metrics["selected_model"] == True
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "GRU-GAT MAE",
        f"{selected_model['mae']:.4f}"
    )

    col2.metric(
        "GRU-GAT RMSE",
        f"{selected_model['rmse']:.4f}"
    )

    col3.metric(
        "GRU-GAT WAPE",
        f"{selected_model['wape_percent']:.2f}%"
    )

    col4.metric(
        "GRU-GAT R²",
        f"{selected_model['r_squared']:.4f}"
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
            title="Model Error Comparison",
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
            title="WAPE Comparison",
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
        title="Five-Seed Stability",
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
        "Seed 7 was selected using validation performance, "
        "not test performance. This prevents test-set leakage. "
        "All five seeds produced similar test results, indicating "
        "that the GRU-GAT improvement is stable rather than an "
        "isolated random initialization."
    )
