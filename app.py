import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Motion Explorer", layout="wide")

st.title("Motion Explorer")
st.write(
    "An interactive visualization connecting calculus derivatives "
    "to physical motion."
)

# Time values
t = np.linspace(0, 8, 200)

# Position function
s = -(t**2) + 8*t

# Slider
time = st.slider(
    "Choose time (seconds)",
    min_value=0.0,
    max_value=8.0,
    value=2.0,
    step=0.1
)

# Current position
position = -(time**2) + 8*time

# Derivative = velocity
velocity = -2*time + 8

# Tangent line
tangent = velocity * (t - time) + position

# Create graph
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=s,
        mode="lines",
        name="Position s(t)"
    )
)

fig.add_trace(
    go.Scatter(
        x=t,
        y=tangent,
        mode="lines",
        name="Tangent Line"
    )
)

fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        marker=dict(size=12),
        name="Current Position"
    )
)

fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Position (meters)"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("What does this mean?")

st.write(
    f"""
At **t = {time:.1f} seconds**:

- Position: **{position:.2f} meters**
- Velocity: **{velocity:.2f} m/s**

The slope of the tangent line represents the instantaneous velocity.
"""
)
