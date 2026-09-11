import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference.predict import NetOraclePredictor
from preprocressing.features import prepare_features
from preprocressing.load_data import load_network_data
from preprocressing.state_builder import build_network_states

st.set_page_config(page_title="NetOracle", page_icon="🛡️", layout="wide")
st.title("NetOracle")
st.subheader("Temporal network traffic risk analysis")

uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])

if uploaded_file:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temp_path = temporary_file.name

        frame = load_network_data(temp_path)
        features = prepare_features(frame)
        states = build_network_states(features, window_size=50)
        st.caption(f"Loaded {len(frame):,} flows and created {len(states):,} complete windows.")

        predictor = NetOraclePredictor()
        report = predictor.analyze(states)

        if report["status"] == "Suspicious":
            st.error(f"Suspicious activity detected: {report['risk']:.1%} maximum risk")
        else:
            st.success(f"No suspicious activity detected: {report['risk']:.1%} maximum risk")

        first, second, third, fourth = st.columns(4)
        first.metric("Decision", report["status"])
        second.metric("Maximum risk", f"{report['risk']:.1%}")
        third.metric("Windows analyzed", report["windows_analyzed"])
        fourth.metric("Suspicious windows", f"{report['suspicious_percentage']:.1f}%")

        results = pd.DataFrame(report["windows"])
        st.subheader("Suspicion timeline")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=results["window"],
                y=results["risk"],
                mode="lines+markers",
                name="Risk",
            )
        )
        figure.add_hline(y=report["threshold"], line_dash="dash", annotation_text="Decision threshold")
        figure.update_layout(xaxis_title="Network window", yaxis_title="Suspicion risk", yaxis_range=[0, 1])
        st.plotly_chart(figure, use_container_width=True)

        st.subheader("Window decisions")
        st.dataframe(results, use_container_width=True, hide_index=True)
    except FileNotFoundError as error:
        st.error(str(error))
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Could not analyze this file: {error}")
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
else:
    st.info("Upload a supported network-flow CSV to begin analysis.")
