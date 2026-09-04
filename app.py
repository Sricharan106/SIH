import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from preprocessing.features import prepare_features
from preprocessing.load_data import load_network_data
from preprocessing.state_builder import build_network_states

from inference.predict import NetOraclePredictor

st.set_page_config(page_title="NetOracle", page_icon="🛡️", layout="wide")


st.title("🛡️ NetOracle")

st.subheader("Adaptive AI World Model for Predictive Cyber Defence")


uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])


if uploaded_file:
    # Save uploaded file temporarily

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        temp.write(uploaded_file.getvalue())

        temp_path = temp.name

    # Load

    df = load_network_data(temp_path)

    features = prepare_features(df)

    states = build_network_states(features, window_size=50)

    st.success(f"Created {len(states)} network states")

    # Load AI

    predictor = NetOraclePredictor()

    with st.spinner("AI is analysing network behaviour..."):
        results = predictor.analyze(states)

    # ---------------------------------
    # Display modes
    # ---------------------------------

    st.header("⚡ Adaptive Intelligence Status")

    mode_data = pd.DataFrame(results)

    st.dataframe(mode_data[["window", "anomaly_score", "mode"]])

    # ---------------------------------
    # Anomaly graph
    # ---------------------------------

    st.header("📊 Network Suspicion Timeline")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mode_data["window"],
            y=mode_data["anomaly_score"],
            mode="lines+markers",
            name="Suspicion Score",
        )
    )

    fig.update_layout(xaxis_title="Network Window", yaxis_title="Suspicion Score")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # World Model Results
    # ---------------------------------

    deep_results = [result for result in results if "future_predictions" in result]

    if deep_results:
        st.header("🧠 World Model Attack Forecast")

        latest = deep_results[-1]

        predictions = latest["future_predictions"]

        forecast_df = pd.DataFrame(predictions)

        st.dataframe(forecast_df[["step", "attack_probability", "stage"]])

        # Forecast graph

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(x=forecast_df["step"], y=forecast_df["attack_probability"])
        )

        fig2.update_layout(
            title="Future Infiltration Forecast",
            xaxis_title="Future Step",
            yaxis_title="Attack Probability",
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Attack stages

        st.header("🎯 Predicted Attack Progression")

        for prediction in predictions:
            probability = prediction["attack_probability"]

            stage = prediction["stage"]

            st.write(f"**T+{prediction['step']}** → {stage} ({probability:.1%} risk)")

    else:
        st.info(
            "🟢 No sustained suspicious "
            "behaviour detected. "
            "Heavy World Model was not activated."
        )
