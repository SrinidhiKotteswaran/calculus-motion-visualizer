import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Motion Explorer",
    page_icon="📐",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📐 Motion Explorer")

st.markdown(
    """
    **An interactive visualization connecting calculus derivatives
    to physical motion.**
    
    Explore how a position function creates velocity through its
    derivative — and see the mathematics represented visually.
    """
)

st.divider()

# ---------------------------------------------------------
# POSITION FUNCTION
# ---------------------------------------------------------

st.subheader("1. Position Function")

st.latex(
    r"s(t) = -t^2 + 8t"
)

st.write(
    """
    The position function describes where the object is at each
    moment in time.
    """
)

# ---------------------------------------------------------
# TIME DOMAIN
# ---------------------------------------------------------

t = np.linspace(0, 8, 400)

# Position
s = -(t**2) + 8*t

# Velocity
v = -2*t + 8

# Acceleration
a = -2 * np.ones_like(t)

# ---------------------------------------------------------
# TIME SLIDER
# ---------------------------------------------------------

st.subheader("2. Choose a Moment in Time")

time = st.slider(
    "Time (seconds)",
    min_value=0.0,
    max_value=8.0,
    value=2.0,
    step=0.1
)

# ---------------------------------------------------------
# CURRENT VALUES
# ---------------------------------------------------------

position = -(time**2) + 8*time
velocity = -2*time + 8
acceleration = -2

# Tangent line
tangent = velocity * (t - time) + position

# ---------------------------------------------------------
# MOTION STATUS
# ---------------------------------------------------------

if velocity > 0.01:
    motion_status = "Moving forward →"
elif velocity < -0.01:
    motion_status = "Moving backward ←"
else:
    motion_status = "Momentarily stopped"

# ---------------------------------------------------------
# METRIC CARDS
# ---------------------------------------------------------

st.subheader("3. Instantaneous Motion")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Position",
        f"{position:.2f} m"
    )

with col2:
    st.metric(
        "Velocity",
        f"{velocity:.2f} m/s"
    )

with col3:
    st.metric(
        "Acceleration",
        f"{acceleration:.2f} m/s²"
    )

with col4:
    st.metric(
        "Motion",
        motion_status
    )

# ---------------------------------------------------------
# MAIN GRAPH
# ---------------------------------------------------------

st.subheader("4. Position and Tangent Line")

fig = go.Figure()

# Position curve
fig.add_trace(
    go.Scatter(
        x=t,
        y=s,
        mode="lines",
        name="Position s(t)",
        line=dict(
            width=4
        )
    )
)

# Tangent line
fig.add_trace(
    go.Scatter(
        x=t,
        y=tangent,
        mode="lines",
        name="Tangent Line",
        line=dict(
            dash="dash",
            width=3
        )
    )
)

# Current position
fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Current Position",
        marker=dict(
            size=14
        )
    )
)

# Horizontal reference line
fig.add_hline(
    y=0,
    line_width=1
)

# Vertical reference line
fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot"
)

fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Position (meters)",
    hovermode="x unified",
    height=600,
    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# TANGENT LINE EXPLANATION
# ---------------------------------------------------------

st.subheader("5. What Does the Tangent Line Mean?")

st.markdown(
    f"""
    At **t = {time:.1f} seconds**, the object is at:

    **s({time:.1f}) = {position:.2f} meters**

    The slope of the tangent line is:

    **{velocity:.2f} m/s**

    This slope represents the object's **instantaneous velocity**.
    """
)

st.latex(
    r"v(t) = s'(t)"
)

st.latex(
    r"s'(t) = -2t + 8"
)

# ---------------------------------------------------------
# VELOCITY GRAPH
# ---------------------------------------------------------

st.subheader("6. Velocity as the Derivative")

velocity_fig = go.Figure()

velocity_fig.add_trace(
    go.Scatter(
        x=t,
        y=v,
        mode="lines",
        name="Velocity v(t)",
        line=dict(
            width=4
        )
    )
)

velocity_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[velocity],
        mode="markers",
        name="Current Velocity",
        marker=dict(
            size=13
        )
    )
)

velocity_fig.add_hline(
    y=0,
    line_width=1
)

velocity_fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot"
)

velocity_fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Velocity (m/s)",
    height=450,
    hovermode="x unified"
)

st.plotly_chart(
    velocity_fig,
    use_container_width=True
)

# ---------------------------------------------------------
# ACCELERATION
# ---------------------------------------------------------

st.subheader("7. Acceleration")

st.latex(
    r"a(t) = v'(t) = s''(t)"
)

st.latex(
    r"a(t) = -2"
)

st.write(
    """
    Because acceleration is negative and constant, the object's
    velocity decreases by 2 m/s every second.
    """
)

acceleration_fig = go.Figure()

acceleration_fig.add_trace(
    go.Scatter(
        x=t,
        y=a,
        mode="lines",
        name="Acceleration a(t)",
        line=dict(
            width=4
        )
    )
)

acceleration_fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Acceleration (m/s²)",
    height=350,
    hovermode="x unified"
)

st.plotly_chart(
    acceleration_fig,
    use_container_width=True
)

# ---------------------------------------------------------
# CALCULUS CONNECTION
# ---------------------------------------------------------

st.subheader("8. The Calculus Connection")

st.markdown(
    """
    Motion Explorer demonstrates a central idea in calculus:

    **Position → Velocity → Acceleration**
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📍 Position")

    st.latex(
        r"s(t) = -t^2 + 8t"
    )

    st.write(
        "Describes where the object is."
    )

with col2:
    st.markdown("### ➡️ Velocity")

    st.latex(
        r"v(t) = s'(t)"
    )

    st.write(
        "Describes how quickly position is changing."
    )

with col3:
    st.markdown("### ⚡ Acceleration")

    st.latex(
        r"a(t) = v'(t) = s''(t)"
    )

    st.write(
        "Describes how quickly velocity is changing."
    )

# ---------------------------------------------------------
# IMPORTANT MOMENT
# ---------------------------------------------------------

st.subheader("9. Find the Turning Point")

turning_time = 4.0
turning_position = -(turning_time**2) + 8*turning_time

st.info(
    f"""
    The object reaches its maximum position at **t = 4 seconds**.

    At this moment:

    **Position:** {turning_position:.2f} m

    **Velocity:** 0 m/s

    When velocity equals zero, the object's position stops increasing
    and begins decreasing.
    """
)

# ---------------------------------------------------------
# MOTION INTERPRETATION
# ---------------------------------------------------------

st.subheader("10. Interpret the Motion")

if time < 4:
    interpretation = (
        "The object is moving forward because its velocity is positive. "
        "Its position is increasing."
    )
elif time == 4:
    interpretation = (
        "The object is momentarily stopped. Its velocity is zero and "
        "it is at its maximum position."
    )
else:
    interpretation = (
        "The object is moving backward because its velocity is negative. "
        "Its position is decreasing."
    )

st.write(interpretation)

# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

st.divider()

st.subheader("🧠 How It Works")

st.markdown(
    """
    ### Step 1 — Position

    The program begins with a position function:

    """
)

st.latex(
    r"s(t) = -t^2 + 8t"
)

st.markdown(
    """
    ### Step 2 — Differentiate

    The derivative gives velocity:
    """
)

st.latex(
    r"v(t) = s'(t) = -2t + 8"
)

st.markdown(
    """
    ### Step 3 — Evaluate at the selected time

    When you move the slider, the program calculates the position
    and velocity at that exact moment.

    ### Step 4 — Build the tangent line

    The tangent line uses the instantaneous velocity as its slope:

    """
)

st.latex(
    r"y = v(t_0)(t-t_0)+s(t_0)"
)

st.markdown(
    """
    This makes the geometric meaning of the derivative visible:
    **the derivative is the slope of the tangent line.**
    """
)

# ---------------------------------------------------------
# PROJECT PURPOSE
# ---------------------------------------------------------

st.divider()

st.subheader("🎯 Why I Built This")

st.write(
    """
    Many students learn derivative rules procedurally without
    developing an intuitive understanding of what derivatives
    represent.

    Motion Explorer connects the algebraic derivative to a physical
    interpretation by showing position, tangent slope, velocity,
    and acceleration simultaneously.
    """
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Motion Explorer • Built with Python, Streamlit, NumPy, and Plotly"
)
