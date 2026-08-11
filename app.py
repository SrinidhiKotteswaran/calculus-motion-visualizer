import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Motion Explorer",
    page_icon="📐",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1.8rem;
        margin-bottom: 0.5rem;
    }

    .equation-box {
        background: #f7f7f8;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.7rem 0 1rem 0;
        font-size: 1.15rem;
    }

    .metric-card {
        background: #f7f7f8;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 650;
    }

    .explanation {
        background: #fafafa;
        border-left: 4px solid #555;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📐 Motion Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An interactive visualization connecting calculus derivatives to physical motion.'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Explore how the derivative of a position function represents "
    "instantaneous velocity — and see that relationship directly on a graph."
)


# ---------------------------------------------------------
# POSITION FUNCTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">1. Position Function</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="equation-box">$$s(t) = -t^2 + 8t$$</div>',
    unsafe_allow_html=True
)

st.write(
    "The position function describes where the object is located at each "
    "moment in time."
)


# ---------------------------------------------------------
# TIME SLIDER
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">2. Choose a Moment in Time</div>',
    unsafe_allow_html=True
)

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

# Position:
# s(t) = -t² + 8t

position = -(time ** 2) + 8 * time

# Velocity:
# v(t) = s'(t) = -2t + 8

velocity = -2 * time + 8

# Acceleration:
# a(t) = v'(t) = -2

acceleration = -2


# Determine motion state

if velocity > 0.05:
    motion = "Moving forward →"
elif velocity < -0.05:
    motion = "Moving backward ←"
else:
    motion = "Stopped"


# ---------------------------------------------------------
# MOTION SUMMARY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">3. Instantaneous Motion</div>',
    unsafe_allow_html=True
)

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
        motion
    )


# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">4. Position and Tangent Line</div>',
    unsafe_allow_html=True
)

# Time values for graph

t = np.linspace(0, 8, 400)

# Position function

s = -(t ** 2) + 8 * t

# Tangent line:
#
# y = v(t0)(t - t0) + s(t0)

tangent = velocity * (t - time) + position


fig = go.Figure()


# Position curve

fig.add_trace(
    go.Scatter(
        x=t,
        y=s,
        mode="lines",
        name="Position s(t)",
        line=dict(width=3)
    )
)


# Tangent line

fig.add_trace(
    go.Scatter(
        x=t,
        y=tangent,
        mode="lines",
        name="Tangent line",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


# Current position

fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Current position",
        marker=dict(
            size=12
        )
    )
)


# Zero line

fig.add_hline(
    y=0,
    line_width=1
)


# Vertical line at selected time

fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot"
)


fig.update_layout(
    height=520,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),
    xaxis_title="Time (seconds)",
    yaxis_title="Position (meters)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# TANGENT LINE EXPLANATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">5. What Does the Tangent Line Mean?</div>',
    unsafe_allow_html=True
)

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
    'The slope of the tangent line represents the object\'s '
    '<strong>instantaneous velocity</strong>. '
    'In calculus, this slope is the derivative of the position function.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# VELOCITY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">6. Velocity as the Derivative</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="equation-box">'
    '$$v(t) = s\'(t) = -2t + 8$$'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Velocity tells us how quickly the object's position is changing."
)

st.write(
    f"At the selected time:"
)

st.latex(
    f"v({time:.1f}) = {-2 * time + 8:.2f}\\;\\text{{m/s}}"
)


# ---------------------------------------------------------
# ACCELERATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">7. Acceleration</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="equation-box">'
    '$$a(t) = v\'(t) = s\'\'(t) = -2$$'
    '</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    '<div class="section-title">8. The Calculus Connection</div>',
    unsafe_allow_html=True
)

st.write(
    "Motion Explorer demonstrates the relationship between three "
    "fundamental quantities:"
)

connection_col1, connection_col2, connection_col3 = st.columns(3)

with connection_col1:
    st.markdown("### 📍 Position")
    st.latex(r"s(t) = -t^2 + 8t")
    st.write("Describes where the object is.")

with connection_col2:
    st.markdown("### ➡️ Velocity")
    st.latex(r"v(t) = s'(t)")
    st.write("Describes how quickly position is changing.")

with connection_col3:
    st.markdown("### ⚡ Acceleration")
    st.latex(r"a(t) = v'(t) = s''(t)")
    st.write("Describes how quickly velocity is changing.")


# ---------------------------------------------------------
# TURNING POINT
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">9. Find the Turning Point</div>',
    unsafe_allow_html=True
)

turning_time = 4.0
turning_position = 16.0
turning_velocity = 0.0

st.write(
    "The object reaches its maximum position when its velocity becomes zero."
)

turn_col1, turn_col2, turn_col3 = st.columns(3)

with turn_col1:
    st.metric(
        "Time",
        "4.00 s"
    )

with turn_col2:
    st.metric(
        "Position",
        "16.00 m"
    )

with turn_col3:
    st.metric(
        "Velocity",
        "0.00 m/s"
    )

st.write(
    "At this point, the object stops moving forward and begins moving backward."
)


# ---------------------------------------------------------
# INTERPRET THE CURRENT MOTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">10. Interpret the Motion</div>',
    unsafe_allow_html=True
)

if velocity > 0:
    interpretation = (
        f"At **t = {time:.1f} s**, velocity is positive, so the object's "
        "position is increasing. It is moving forward."
    )

elif velocity < 0:
    interpretation = (
        f"At **t = {time:.1f} s**, velocity is negative, so the object's "
        "position is decreasing. It is moving backward."
    )

else:
    interpretation = (
        f"At **t = {time:.1f} s**, velocity is zero. "
        "The object is momentarily stopped."
    )

st.info(interpretation)


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🧠 How It Works</div>',
    unsafe_allow_html=True
)

with st.expander("Step 1 — Define the position"):
    st.write(
        "The program begins with a position function describing the "
        "object's location over time."
    )
    st.latex(r"s(t) = -t^2 + 8t")

with st.expander("Step 2 — Differentiate"):
    st.write(
        "Taking the derivative gives the instantaneous velocity."
    )
    st.latex(r"v(t) = s'(t) = -2t + 8")

with st.expander("Step 3 — Choose a time"):
    st.write(
        "The slider selects a specific moment. The application evaluates "
        "the position and velocity at that exact time."
    )

with st.expander("Step 4 — Build the tangent line"):
    st.write(
        "The tangent line is constructed using the instantaneous velocity "
        "as its slope."
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

st.markdown(
    '<div class="section-title">🎯 Why I Built This</div>',
    unsafe_allow_html=True
)

st.write(
    "Many students can calculate derivatives without developing an "
    "intuition for what a derivative actually represents."
)

st.write(
    "I built Motion Explorer to connect the algebraic derivative to "
    "physical motion by showing position, tangent slope, velocity, "
    "and acceleration together in an interactive visualization."
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    '<div class="footer">'
    'Motion Explorer · Built with Python, Streamlit, NumPy, and Plotly'
    '</div>',
    unsafe_allow_html=True
)
