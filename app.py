import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Motion Explorer",
    layout="wide"
)

# ---------------------------------------------------------
# PAGE STYLE
# ---------------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.25rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 2rem !important;
        margin-bottom: 0.7rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }

    .explanation {
        border-left: 3px solid #888;
        padding-left: 1rem;
        margin: 1rem 0 1.5rem 0;
    }

    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #ddd;
        text-align: center;
        color: #777;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("Motion Explorer")

st.markdown(
    '<div class="subtitle">'
    "An interactive visualization connecting calculus derivatives to physical motion."
    "</div>",
    unsafe_allow_html=True
)

st.write(
    "Explore how the derivative of a position function represents "
    "instantaneous velocity and see that relationship directly on a graph."
)


# ---------------------------------------------------------
# POSITION FUNCTION
# ---------------------------------------------------------

st.header("1. Position Function")

st.latex(r"s(t) = -t^2 + 8t")

st.write(
    "The position function describes where the object is located at "
    "each moment in time."
)


# ---------------------------------------------------------
# TIME SLIDER
# ---------------------------------------------------------

st.header("2. Choose a Moment in Time")

time = st.slider(
    "Time (seconds)",
    min_value=0.0,
    max_value=8.0,
    value=2.0,
    step=0.1
)


# ---------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------

position = -(time ** 2) + 8 * time

velocity = -2 * time + 8

acceleration = -2


if velocity > 0.05:
    motion = "Moving forward"
elif velocity < -0.05:
    motion = "Moving backward"
else:
    motion = "Stopped"


# ---------------------------------------------------------
# CURRENT MOTION
# ---------------------------------------------------------

st.header("3. Instantaneous Motion")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Position", f"{position:.2f} m")

with col2:
    st.metric("Velocity", f"{velocity:.2f} m/s")

with col3:
    st.metric("Acceleration", f"{acceleration:.2f} m/s²")

with col4:
    st.metric("Motion", motion)


# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

st.header("4. Position and Tangent Line")

t = np.linspace(0, 8, 400)

position_curve = -(t ** 2) + 8 * t

tangent_line = (
    velocity * (t - time) + position
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=position_curve,
        mode="lines",
        name="Position s(t)",
        line=dict(width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=t,
        y=tangent_line,
        mode="lines",
        name="Tangent line",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Current position",
        marker=dict(size=11)
    )
)

fig.add_vline(
    x=time,
    line_dash="dot",
    line_width=1
)

fig.add_hline(
    y=0,
    line_width=1
)

fig.update_layout(
    height=520,
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20
    ),
    xaxis_title="Time (seconds)",
    yaxis_title="Position (meters)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# TANGENT LINE
# ---------------------------------------------------------

st.header("5. What Does the Tangent Line Mean?")

st.write(
    f"At **t = {time:.1f} seconds**, the object is at "
    f"**s({time:.1f}) = {position:.2f} meters**."
)

st.write(
    f"The slope of the tangent line is "
    f"**{velocity:.2f} m/s**."
)

st.markdown(
    '<div class="explanation">'
    "The slope of the tangent line represents the object's "
    "<strong>instantaneous velocity</strong>. "
    "In calculus, this slope is the derivative of the position function."
    "</div>",
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# VELOCITY
# ---------------------------------------------------------

st.header("6. Velocity as the Derivative")

st.latex(r"v(t) = s'(t) = -2t + 8")

st.write(
    "Velocity tells us how quickly the object's position is changing."
)

st.write("At the selected time:")

st.latex(
    rf"v({time:.1f}) = {velocity:.2f}\ \mathrm{{m/s}}"
)


# ---------------------------------------------------------
# ACCELERATION
# ---------------------------------------------------------

st.header("7. Acceleration")

st.latex(r"a(t) = v'(t) = s''(t) = -2")

st.write(
    "Acceleration describes how quickly velocity changes."
)

st.write(
    "Because the acceleration is constant and negative, "
    "the object's velocity decreases by 2 m/s every second."
)


# ---------------------------------------------------------
# CALCULUS CONNECTION
# ---------------------------------------------------------

st.header("8. The Calculus Connection")

st.write(
    "Motion Explorer demonstrates the relationship between "
    "position, velocity, and acceleration."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Position")
    st.latex(r"s(t) = -t^2 + 8t")
    st.write("Describes where the object is.")

with col2:
    st.subheader("Velocity")
    st.latex(r"v(t) = s'(t)")
    st.write("Describes how quickly position is changing.")

with col3:
    st.subheader("Acceleration")
    st.latex(r"a(t) = v'(t) = s''(t)")
    st.write("Describes how quickly velocity is changing.")


# ---------------------------------------------------------
# TURNING POINT
# ---------------------------------------------------------

st.header("9. Find the Turning Point")

st.write(
    "The object reaches its maximum position when its velocity becomes zero."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Time", "4.00 s")

with col2:
    st.metric("Position", "16.00 m")

with col3:
    st.metric("Velocity", "0.00 m/s")

st.write(
    "At this point, the object stops moving forward and begins moving backward."
)


# ---------------------------------------------------------
# INTERPRET THE MOTION
# ---------------------------------------------------------

st.header("10. Interpret the Motion")

if velocity > 0:
    st.write(
        f"At **t = {time:.1f} s**, velocity is positive, so the "
        "object's position is increasing. It is moving forward."
    )

elif velocity < 0:
    st.write(
        f"At **t = {time:.1f} s**, velocity is negative, so the "
        "object's position is decreasing. It is moving backward."
    )

else:
    st.write(
        f"At **t = {time:.1f} s**, velocity is zero. "
        "The object is momentarily stopped."
    )


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

st.header("How It Works")

with st.expander("Step 1 — Define the position"):
    st.write(
        "The program begins with a position function describing "
        "the object's location over time."
    )
    st.latex(r"s(t) = -t^2 + 8t")

with st.expander("Step 2 — Differentiate"):
    st.write(
        "Taking the derivative gives the instantaneous velocity."
    )
    st.latex(r"v(t) = s'(t) = -2t + 8")

with st.expander("Step 3 — Choose a time"):
    st.write(
        "The slider selects a specific moment. The application "
        "evaluates the position and velocity at that time."
    )

with st.expander("Step 4 — Build the tangent line"):
    st.write(
        "The tangent line is constructed using the instantaneous "
        "velocity as its slope."
    )
    st.latex(
        r"y = v(t_0)(t-t_0)+s(t_0)"
    )
    st.write(
        "This makes the geometric meaning of the derivative visible: "
        "the derivative is the slope of the tangent line."
    )


# ---------------------------------------------------------
# WHY I BUILT THIS
# ---------------------------------------------------------

st.header("Why I Built This")

st.write(
    "Many students can calculate derivatives without developing "
    "an intuition for what a derivative actually represents."
)

st.write(
    "I built Motion Explorer to connect the algebraic derivative "
    "to physical motion by showing position, tangent slope, velocity, "
    "and acceleration together in an interactive visualization."
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    '<div class="footer">'
    "Motion Explorer · Built with Python, Streamlit, NumPy, and Plotly"
    "</div>",
    unsafe_allow_html=True
)
