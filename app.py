import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Motion Explorer",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .block-container {
        max-width: 1180px;
        padding-top: 4.5rem;
        padding-bottom: 6rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .stApp {
        background: #ffffff;
    }

    p {
        line-height: 1.75;
    }

    /* More breathing room between Streamlit elements */
    div[data-testid="stVerticalBlock"] {
        gap: 0.9rem;
    }

    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        margin-bottom: 4.5rem;
    }

    .hero-kicker {
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.4rem;
        line-height: 1.05;
        letter-spacing: -0.055em;
        font-weight: 720;
        color: #0f172a;
        margin: 0;
    }

    .hero-description {
        max-width: 720px;
        font-size: 1.08rem;
        line-height: 1.75;
        color: #64748b;
        margin-top: 1.15rem;
    }

    /* --------------------------------------------------------
       SECTION HEADINGS
       -------------------------------------------------------- */

    .section {
        margin-top: 5rem;
        margin-bottom: 1.8rem;
    }

    .section-number {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        color: #94a3b8;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 1.65rem;
        line-height: 1.2;
        font-weight: 680;
        letter-spacing: -0.025em;
        color: #0f172a;
        margin: 0;
    }

    .section-description {
        color: #64748b;
        margin-top: 0.65rem;
        max-width: 760px;
        line-height: 1.7;
    }

    /* --------------------------------------------------------
       FORM CONTROLS
       -------------------------------------------------------- */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 8px;
    }

    .helper-text {
        color: #94a3b8;
        font-size: 0.88rem;
    }

    /* --------------------------------------------------------
       METRICS
       -------------------------------------------------------- */

    .metrics-row {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.15rem 0;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 650;
    }

    /* --------------------------------------------------------
       CALLOUT
       -------------------------------------------------------- */

    .concept {
        margin-top: 1.7rem;
        margin-bottom: 1.7rem;
        padding: 1.25rem 1.4rem;
        border-left: 3px solid #2563eb;
        background: #f8fafc;
        color: #334155;
        border-radius: 0 8px 8px 0;
        line-height: 1.75;
    }

    /* --------------------------------------------------------
       CALCULUS CONNECTION
       -------------------------------------------------------- */

    .concept-column {
        padding-top: 0.4rem;
        padding-right: 1.5rem;
    }

    .concept-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .concept-text {
        color: #64748b;
        line-height: 1.7;
        margin-top: 0.7rem;
    }

    /* --------------------------------------------------------
       INTERPRETATION
       -------------------------------------------------------- */

    .interpretation {
        margin-top: 1.5rem;
        padding: 1.3rem 1.5rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        color: #334155;
        line-height: 1.75;
    }

    /* --------------------------------------------------------
       HOW IT WORKS
       -------------------------------------------------------- */

    .step {
        margin-bottom: 2.1rem;
    }

    .step-title {
        font-weight: 650;
        color: #0f172a;
        margin-bottom: 0.45rem;
    }

    .step-description {
        color: #64748b;
        line-height: 1.7;
    }

    /* --------------------------------------------------------
       WHY I BUILT THIS
       -------------------------------------------------------- */

    .why-built {
        max-width: 800px;
        color: #475569;
        font-size: 1.03rem;
        line-height: 1.85;
    }

    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .footer {
        margin-top: 6rem;
        padding-top: 1.8rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.82rem;
    }

    /* --------------------------------------------------------
       DARK MODE
       -------------------------------------------------------- */

    @media (prefers-color-scheme: dark) {

        .stApp {
            background: #0f172a;
        }

        .hero-title,
        .section-title,
        .step-title,
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        .hero-description,
        .section-description,
        .concept-text,
        .step-description,
        .why-built {
            color: #94a3b8;
        }

        .concept {
            background: #111c2f;
            color: #cbd5e1;
            border-left-color: #60a5fa;
        }

        .interpretation {
            background: #111c2f;
            border-color: #1e293b;
            color: #cbd5e1;
        }

        .footer {
            border-color: #1e293b;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYMBOLIC CALCULUS
# ============================================================

t = sp.Symbol("t", real=True)

TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

ALLOWED_LOCALS = {
    "t": t,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,

    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,

    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,

    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,

    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,

    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


# ============================================================
# EXAMPLES
# ============================================================

PRESETS = {
    "Quadratic": "-t^2 + 8t",
    "Cubic": "t^3 - 3t",
    "Sine": "sin(t)",
    "Cosine": "cos(t)",
    "Exponential": "e^t",
    "Exponential Decay": "e^(-t)",
    "Logarithmic": "ln(t)",
    "Power": "t^2.5",
    "Rational": "1/t",
    "Damped Oscillation": "e^(-0.2t)*sin(t)",
}


# ============================================================
# PARSING
# ============================================================

def parse_function(expression):

    expression = expression.strip()

    if not expression:
        raise ValueError(
            "Please enter a position function."
        )

    if len(expression) > 150:
        raise ValueError(
            "Please keep the function under 150 characters."
        )

    if not re.fullmatch(
        r"[0-9a-zA-Z_+\-*/^().,\s]+",
        expression,
    ):
        raise ValueError(
            "The function contains an unsupported character."
        )

    names = re.findall(
        r"[A-Za-z_][A-Za-z_0-9]*",
        expression,
    )

    unknown_names = [
        name
        for name in names
        if name not in ALLOWED_LOCALS
    ]

    if unknown_names:
        raise ValueError(
            "Unsupported name(s): "
            + ", ".join(sorted(set(unknown_names)))
        )

    try:

        parsed = parse_expr(
            expression,
            local_dict=ALLOWED_LOCALS,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

    except Exception as exc:

        raise ValueError(
            "I couldn't interpret that function. "
            "Try something like sin(t), e^t, "
            "ln(t), or t^3 - 3t."
        ) from exc

    if t not in parsed.free_symbols:
        raise ValueError(
            "The position function must depend on t."
        )

    return parsed


# ============================================================
# NUMERICAL EVALUATION
# ============================================================

def numerical_function(expr):

    return sp.lambdify(
        t,
        expr,
        modules=["numpy"],
    )


def evaluate_array(expr, values):

    function = numerical_function(expr)

    values = np.asarray(
        values,
        dtype=float,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):

        result = function(values)

    result = np.asarray(result)

    if np.iscomplexobj(result):

        result = np.real_if_close(result)

        if np.iscomplexobj(result):

            result = np.full(
                values.shape,
                np.nan,
            )

    try:

        result = result.astype(float)

    except (TypeError, ValueError):

        result = np.full(
            values.shape,
            np.nan,
        )

    result[~np.isfinite(result)] = np.nan

    return result


def evaluate_at(expr, value):

    try:

        result = sp.N(
            expr.subs(
                t,
                value,
            )
        )

        if result.is_real is False:
            return None

        result = float(result)

        if not np.isfinite(result):
            return None

        return result

    except Exception:

        return None


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            Interactive Calculus Visualization
        </div>

        <div class="hero-title">
            Motion Explorer
        </div>

        <div class="hero-description">
            Explore how derivatives describe physical motion.
            Connect position, velocity, acceleration, and the
            geometry of tangent lines through an interactive
            mathematical model.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 01 — POSITION FUNCTION
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">01</div>

        <div class="section-title">
            Define the motion
        </div>

        <div class="section-description">
            Choose an example or enter a position function
            of time. The application symbolically computes
            its first and second derivatives.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

function_col1, function_col2 = st.columns(
    [1, 2],
    gap="large",
)

with function_col1:

    preset = st.selectbox(
        "Example",
        list(PRESETS.keys()),
    )

with function_col2:

    custom_function = st.text_input(
        "Position function",
        value=PRESETS[preset],
        placeholder="e.g. -t^2 + 8t",
        help=(
            "Use t as the independent variable. "
            "Examples: sin(t), e^t, ln(t), t^3 - 3t."
        ),
    )


try:

    position_function = parse_function(
        custom_function
    )

    velocity_function = sp.simplify(
        sp.diff(
            position_function,
            t,
        )
    )

    acceleration_function = sp.simplify(
        sp.diff(
            velocity_function,
            t,
        )
    )

except ValueError as error:

    st.error(str(error))
    st.stop()


st.markdown(
    '<div style="margin-top: 1.8rem;"></div>',
    unsafe_allow_html=True,
)

st.latex(
    rf"s(t) = {sp.latex(position_function)}"
)

st.markdown(
    '<div class="helper-text">'
    "Position describes where the object is at each moment."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# 02 — TIME
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">02</div>

        <div class="section-title">
            Choose a moment
        </div>

        <div class="section-description">
            Select the time interval you want to explore,
            then move through the motion one instant at a time.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

time_col1, time_col2 = st.columns(
    2,
    gap="large",
)

with time_col1:

    start_time = st.number_input(
        "Start time",
        value=0.0,
        step=0.5,
    )

with time_col2:

    end_time = st.number_input(
        "End time",
        value=8.0,
        step=0.5,
    )

if end_time <= start_time:

    st.error(
        "End time must be greater than start time."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

default_time = (
    start_time
    + 0.25 * (end_time - start_time)
)

if "selected_time" not in st.session_state:

    st.session_state.selected_time = default_time

else:

    st.session_state.selected_time = min(
        max(
            float(st.session_state.selected_time),
            float(start_time),
        ),
        float(end_time),
    )


time = st.slider(
    "Time",
    min_value=float(start_time),
    max_value=float(end_time),
    value=float(
        st.session_state.selected_time
    ),
    step=0.1,
    key="time_slider",
    format="%.2f s",
)

st.session_state.selected_time = time


# ============================================================
# CURRENT VALUES
# ============================================================

position = evaluate_at(
    position_function,
    time,
)

velocity = evaluate_at(
    velocity_function,
    time,
)

acceleration = evaluate_at(
    acceleration_function,
    time,
)

if (
    position is None
    or velocity is None
    or acceleration is None
):

    st.warning(
        "The selected time is outside the real-valued "
        "domain of this function or one of its derivatives."
    )

    st.stop()


TOLERANCE = 0.05

if velocity > TOLERANCE:

    motion = "Moving forward"

elif velocity < -TOLERANCE:

    motion = "Moving backward"

else:

    motion = "Momentarily stopped"


# ============================================================
# 03 — INSTANTANEOUS MOTION
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">03</div>

        <div class="section-title">
            Instantaneous motion
        </div>

        <div class="section-description">
            At the selected time, the derivatives give the
            object's instantaneous velocity and acceleration.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(
    3,
    gap="large",
)

with metric1:

    st.metric(
        "Position",
        f"{position:.2f} m",
    )

with metric2:

    st.metric(
        "Velocity",
        f"{velocity:.2f} m/s",
    )

with metric3:

    st.metric(
        "Acceleration",
        f"{acceleration:.2f} m/s²",
    )


st.markdown(
    f"""
    <div style="
        margin-top: 1.4rem;
        color: #64748b;
        font-size: 0.95rem;
    ">
        Motion state:
        <strong style="color: #0f172a;">
            {motion}
        </strong>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GRAPH DATA
# ============================================================

graph_t = np.linspace(
    start_time,
    end_time,
    800,
)

position_values = evaluate_array(
    position_function,
    graph_t,
)

velocity_values = evaluate_array(
    velocity_function,
    graph_t,
)

acceleration_values = evaluate_array(
    acceleration_function,
    graph_t,
)


# ============================================================
# TANGENT
# ============================================================

tangent_values = (
    velocity * (graph_t - time)
    + position
)


# ============================================================
# GRAPH THEME
# ============================================================

GRAPH_BLUE = "#2563EB"
GRAPH_RED = "#DC2626"
GRAPH_GREEN = "#059669"
GRAPH_PURPLE = "#7C3AED"
GRAPH_ORANGE = "#F59E0B"
GRAPH_GRID = "rgba(148, 163, 184, 0.22)"
GRAPH_ZERO = "rgba(100, 116, 139, 0.65)"


# ============================================================
# 04 — POSITION + TANGENT
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">04</div>

        <div class="section-title">
            Position and tangent line
        </div>

        <div class="section-description">
            The tangent line captures the instantaneous rate
            of change of position at the selected moment.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Move the slider, or click a point on the position curve."
)


position_fig = go.Figure()

position_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=position_values,
        mode="lines",
        name="Position",
        line=dict(
            color=GRAPH_BLUE,
            width=3.5,
        ),
        hovertemplate=(
            "t = %{x:.2f} s"
            "<br>"
            "s(t) = %{y:.2f} m"
            "<extra></extra>"
        ),
    )
)

position_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=tangent_values,
        mode="lines",
        name="Tangent",
        line=dict(
            color=GRAPH_RED,
            width=2.5,
            dash="dash",
        ),
        hovertemplate=(
            "Tangent"
            "<br>"
            "slope = "
            f"{velocity:.2f} m/s"
            "<extra></extra>"
        ),
    )
)

position_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Selected time",
        marker=dict(
            size=13,
            color=GRAPH_ORANGE,
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            f"t = {time:.2f} s"
            "<br>"
            f"s(t) = {position:.2f} m"
            "<extra></extra>"
        ),
    )
)

position_fig.add_hline(
    y=0,
    line_width=1,
    line_color=GRAPH_ZERO,
)

position_fig.add_vline(
    x=time,
    line_width=1.4,
    line_dash="dot",
    line_color=GRAPH_ORANGE,
)

position_fig.update_layout(
    height=560,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=30,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title="Time (s)",
        showgrid=True,
        gridcolor=GRAPH_GRID,
        zeroline=False,
    ),
    yaxis=dict(
        title="Position (m)",
        showgrid=True,
        gridcolor=GRAPH_GRID,
        zeroline=False,
    ),
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

position_event = st.plotly_chart(
    position_fig,
    width="stretch",
    key="position_graph",
    on_select="rerun",
    selection_mode="points",
    config={
        "displaylogo": False,
        "scrollZoom": False,
        "responsive": True,
    },
)


# ============================================================
# GRAPH SELECTION
# ============================================================

try:

    selected_points = (
        position_event.selection.points
    )

    if selected_points:

        selected_x = float(
            selected_points[-1]["x"]
        )

        if (
            start_time
            <= selected_x
            <= end_time
        ):

            st.session_state.selected_time = (
                selected_x
            )

            st.rerun()

except Exception:
    pass


# ============================================================
# 05 — TANGENT EXPLANATION
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">05</div>

        <div class="section-title">
            What does the tangent line mean?
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    f"At **t = {time:.2f} seconds**, the object is at "
    f"**s({time:.2f}) = {position:.2f} meters**."
)

st.write(
    f"The slope of the tangent line is "
    f"**{velocity:.2f} m/s**."
)

st.markdown(
    """
    <div class="concept">
        The slope of the tangent line represents
        <strong>instantaneous velocity</strong>.
        In calculus, this slope is the derivative of
        the position function.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 06 — VELOCITY
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">06</div>

        <div class="section-title">
            Velocity as the derivative
        </div>

        <div class="section-description">
            Differentiating position gives the instantaneous
            rate at which position changes.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    rf"v(t) = s'(t) = {sp.latex(velocity_function)}"
)

st.write(
    "Velocity tells us how quickly the object's position "
    "is changing."
)

st.latex(
    rf"v({time:.2f}) = "
    rf"{velocity:.2f}\ \mathrm{{m/s}}"
)


# ============================================================
# 07 — VELOCITY GRAPH
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">07</div>

        <div class="section-title">
            Velocity
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

velocity_fig = go.Figure()

velocity_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=velocity_values,
        mode="lines",
        name="Velocity",
        line=dict(
            color=GRAPH_GREEN,
            width=3.5,
        ),
        hovertemplate=(
            "t = %{x:.2f} s"
            "<br>"
            "v(t) = %{y:.2f} m/s"
            "<extra></extra>"
        ),
    )
)

velocity_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[velocity],
        mode="markers",
        name="Selected time",
        marker=dict(
            size=12,
            color=GRAPH_ORANGE,
            line=dict(
                color="white",
                width=2,
            ),
        ),
    )
)

velocity_fig.add_hline(
    y=0,
    line_width=1,
    line_color=GRAPH_ZERO,
)

velocity_fig.add_vline(
    x=time,
    line_width=1.4,
    line_dash="dot",
    line_color=GRAPH_ORANGE,
)

velocity_fig.update_layout(
    height=420,
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=30,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title="Time (s)",
        showgrid=True,
        gridcolor=GRAPH_GRID,
        zeroline=False,
    ),
    yaxis=dict(
        title="Velocity (m/s)",
        showgrid=True,
        gridcolor=GRAPH_GRID,
        zeroline=False,
    ),
    hovermode="closest",
    showlegend=False,
)

st.plotly_chart(
    velocity_fig,
    width="stretch",
    key="velocity_graph",
    config={
        "displaylogo": False,
        "responsive": True,
    },
)


# ============================================================
# 08 — ACCELERATION
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">08</div>

        <div class="section-title">
            Acceleration
        </div>

        <div class="section-description">
            Differentiating velocity gives the rate at which
            velocity itself is changing.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    rf"a(t) = v'(t) = s''(t) = "
    rf"{sp.latex(acceleration_function)}"
)

st.write(
    "Acceleration describes how quickly velocity changes."
)

st.latex(
    rf"a({time:.2f}) = "
    rf"{acceleration:.2f}\ \mathrm{{m/s^2}}"
)


# ============================================================
# 09 — ACCELERATION GRAPH
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">09</div>

        <div class="section-title">
            Acceleration
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

acceleration_fig = go.Figure()

acceleration_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=acceleration_values,
        mode="lines",
        name="Acceleration",
        line=dict(
            color=GRAPH_PURPLE,
            width=3.5,
        ),
        hovertemplate=(
            "t = %{x:.2f} s"
            "<br>"
            "a(t) = %{y:.2f} m/s²"
            "<extra></extra>"
        ),
    )
)

acceleration_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[acceleration],
        mode="markers",
        name="Selected time",
        marker=dict(
            size=12,
            color=GRAPH_ORANGE,
            line=dict(
                color="white",
                width=2,
            ),
        ),
    )
)

acceleration_fig.add_hline(
    y=0,
    line_width=1,
    line_color=GRAPH_ZERO,
)

acceleration_fig.add_vline(
    x=time,
    line_width=1.4,
    line_dash="dot",
    line_color=GRAPH_ORANGE,
)

acceleration_fig.update_layout(
    height=420,
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=30,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title="Time (s)",
        showgrid=True,
        gridcolor=GRAPH_GRID,
        zeroline=False,
    ),
    yaxis=dict(
        title="Acceleration (m/s²)",
        showgrid=True,
        gridcolor=GRAPH_GRID,
        zeroline=False,
    ),
    hovermode="closest",
    showlegend=False,
)

st.plotly_chart(
    acceleration_fig,
    width="stretch",
    key="acceleration_graph",
    config={
        "displaylogo": False,
        "responsive": True,
    },
)


# ============================================================
# 10 — CALCULUS CONNECTION
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">10</div>

        <div class="section-title">
            The calculus connection
        </div>

        <div class="section-description">
            Three quantities describe the same motion from
            different mathematical perspectives.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

connection1, connection2, connection3 = st.columns(
    3,
    gap="large",
)

with connection1:

    st.markdown(
        """
        <div class="concept-column">

            <div class="concept-label">
                Position
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        rf"s(t) = {sp.latex(position_function)}"
    )

    st.markdown(
        '<div class="concept-text">'
        "Describes where the object is."
        "</div>",
        unsafe_allow_html=True,
    )


with connection2:

    st.markdown(
        """
        <div class="concept-column">

            <div class="concept-label">
                Velocity
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        rf"v(t) = s'(t)"
    )

    st.markdown(
        '<div class="concept-text">'
        "Describes how quickly position is changing."
        "</div>",
        unsafe_allow_html=True,
    )


with connection3:

    st.markdown(
        """
        <div class="concept-column">

            <div class="concept-label">
                Acceleration
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        rf"a(t) = v'(t) = s''(t)"
    )

    st.markdown(
        '<div class="concept-text">'
        "Describes how quickly velocity is changing."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 11 — TURNING POINTS
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">11</div>

        <div class="section-title">
            Explore turning points
        </div>

        <div class="section-description">
            A turning point can occur when instantaneous
            velocity is zero.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    r"s'(t) = 0"
)


try:

    critical_points = sp.solve(
        sp.Eq(
            velocity_function,
            0,
        ),
        t,
    )

    valid_points = []

    for point in critical_points:

        if point.is_real is False:
            continue

        try:

            point_time = float(point)

            if not (
                start_time
                <= point_time
                <= end_time
            ):
                continue

            point_position = evaluate_at(
                position_function,
                point_time,
            )

            if point_position is not None:

                valid_points.append(
                    (
                        point_time,
                        point_position,
                    )
                )

        except Exception:
            continue

    valid_points.sort(
        key=lambda item: item[0]
    )

    if valid_points:

        for (
            point_time,
            point_position,
        ) in valid_points:

            st.write(
                f"Critical point at "
                f"**t = {point_time:.2f} s**"
            )

            c1, c2, c3 = st.columns(
                3,
                gap="large",
            )

            with c1:

                st.metric(
                    "Time",
                    f"{point_time:.2f} s",
                )

            with c2:

                st.metric(
                    "Position",
                    f"{point_position:.2f} m",
                )

            with c3:

                st.metric(
                    "Velocity",
                    "0.00 m/s",
                )

            st.markdown(
                '<div style="height: 1.2rem;"></div>',
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "No critical points where v(t) = 0 "
            "were found in this interval."
        )

except Exception:

    st.info(
        "Critical points could not be determined "
        "symbolically for this function."
    )


# ============================================================
# 12 — INTERPRETATION
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">12</div>

        <div class="section-title">
            Interpret the motion
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

if velocity > TOLERANCE:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is positive, "
        "so position is increasing. The object is moving "
        "forward."
    )

elif velocity < -TOLERANCE:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is negative, "
        "so position is decreasing. The object is moving "
        "backward."
    )

else:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is approximately "
        "zero. The object is momentarily stopped."
    )

st.markdown(
    f'<div class="interpretation">{interpretation}</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">Behind the visualization</div>

        <div class="section-title">
            How it works
        </div>

        <div class="section-description">
            The application turns a symbolic position function
            into an interactive mathematical model.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

steps = [
    (
        "Step 1 — Define the position",
        "The user provides a position function describing "
        "location as a function of time.",
    ),
    (
        "Step 2 — Differentiate",
        "SymPy symbolically differentiates the position "
        "function to obtain velocity.",
    ),
    (
        "Step 3 — Differentiate again",
        "A second derivative gives acceleration.",
    ),
    (
        "Step 4 — Choose a time",
        "The selected time determines the instantaneous "
        "position, velocity, and acceleration.",
    ),
    (
        "Step 5 — Construct the tangent",
        "The tangent line is generated using the instantaneous "
        "velocity as its slope.",
    ),
]

for title, description in steps:

    st.markdown(
        f"""
        <div class="step">

            <div class="step-title">
                {title}
            </div>

            <div class="step-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# WHY I BUILT THIS
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-number">Project motivation</div>

        <div class="section-title">
            Why I built this
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="why-built">

        Many students learn to calculate derivatives without
        developing an intuition for what a derivative actually
        represents.

        <br><br>

        I built <strong>Motion Explorer</strong> to make that
        relationship visible: the derivative becomes the slope
        of a tangent line, the tangent slope becomes velocity,
        and the derivative of velocity becomes acceleration.

        <br><br>

        Rather than treating differentiation as an algebraic
        procedure, the visualization lets students explore
        the same mathematical idea from multiple perspectives.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Motion Explorer &nbsp;·&nbsp;
        Python &nbsp;·&nbsp;
        Streamlit &nbsp;·&nbsp;
        NumPy &nbsp;·&nbsp;
        Plotly &nbsp;·&nbsp;
        SymPy
    </div>
    """,
    unsafe_allow_html=True,
)
