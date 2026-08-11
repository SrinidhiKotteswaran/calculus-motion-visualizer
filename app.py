import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Motion Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CLEAN, MINIMAL STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1100px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Remove excessive vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    /* Main title */
    .motion-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .motion-subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.6rem;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 2.1rem;
        margin-bottom: 0.55rem;
    }

    /* Simple equation styling.
       IMPORTANT: no background, border, or box. */
    .equation {
        margin: 0.4rem 0 0.8rem 0;
    }

    /* Explanation text */
    .explanation {
        margin: 0.8rem 0 1rem 0;
        padding-left: 0.9rem;
        border-left: 3px solid #9ca3af;
    }

    /* Small helper text */
    .helper {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        margin-top: 3.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #e5e7eb;
        text-align: center;
        color: #9ca3af;
        font-size: 0.82rem;
    }

    /* Remove unnecessary borders/backgrounds from metric areas */
    div[data-testid="metric-container"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    /* Make metric labels subtle */
    div[data-testid="stMetricLabel"] {
        color: #6b7280;
    }

    /* Keep widgets visually clean */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SYMBOLIC CALCULUS SETUP
# =========================================================

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


# =========================================================
# EXAMPLES
# =========================================================

PRESETS = {
    "Quadratic": "-t^2 + 8t",
    "Cubic": "t^3 - 3t",
    "Sine": "2sin(t)",
    "Cosine": "2cos(t)",
    "Exponential": "e^t",
    "Exponential Decay": "e^(-t)",
    "Logarithmic": "ln(t)",
    "Power": "t^2.5",
    "Rational": "1/t",
    "Damped Oscillation": "e^(-0.2t)*sin(t)",
}


# =========================================================
# PARSE FUNCTION
# =========================================================

def parse_function(expression):

    expression = expression.strip()

    if not expression:
        raise ValueError("Please enter a position function.")

    if len(expression) > 120:
        raise ValueError(
            "Please keep the function under 120 characters."
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

    allowed_names = set(ALLOWED_LOCALS.keys())

    unknown_names = [
        name
        for name in names
        if name not in allowed_names
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
            "Try something like sin(t), e^t, ln(t), "
            "or t^3 - 3t."
        ) from exc

    if t not in parsed.free_symbols:
        raise ValueError(
            "The position function must depend on t."
        )

    return sp.simplify(parsed)


# =========================================================
# NUMERICAL EVALUATION
# =========================================================

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

        result = np.real_if_close(
            result
        )

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


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="motion-title">Motion Explorer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="motion-subtitle">'
    "An interactive visualization connecting calculus derivatives "
    "to physical motion."
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Explore how the derivative of a position function represents "
    "instantaneous velocity — and see that relationship directly "
    "on a graph."
)


# =========================================================
# 1. POSITION FUNCTION
# =========================================================

st.markdown(
    '<div class="section-heading">1. Position Function</div>',
    unsafe_allow_html=True,
)

function_col1, function_col2 = st.columns(
    [1, 2]
)

with function_col1:

    preset = st.selectbox(
        "Choose an example",
        list(PRESETS.keys()),
    )

with function_col2:

    custom_function = st.text_input(
        "Or enter your own position function",
        value=PRESETS[preset],
        help=(
            "Examples: sin(t), cos(t), e^t, ln(t), "
            "t^3 - 3t, or e^(-t)*sin(t)"
        ),
    )


# =========================================================
# PARSE
# =========================================================

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


# =========================================================
# DISPLAY POSITION FUNCTION
# =========================================================

st.latex(
    rf"s(t) = {sp.latex(position_function)}"
)

st.write(
    "The position function describes where the object "
    "is located at each moment in time."
)


# =========================================================
# 2. TIME RANGE
# =========================================================

st.markdown(
    '<div class="section-heading">2. Choose a Moment in Time</div>',
    unsafe_allow_html=True,
)

range_col1, range_col2 = st.columns(2)

with range_col1:

    start_time = st.number_input(
        "Start time",
        value=0.0,
        step=0.5,
    )

with range_col2:

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


# =========================================================
# SESSION STATE
# =========================================================

if "selected_time" not in st.session_state:

    st.session_state.selected_time = (
        start_time
        + 0.25 * (end_time - start_time)
    )


# Keep selected time inside current interval.

st.session_state.selected_time = min(
    max(
        float(st.session_state.selected_time),
        float(start_time),
    ),
    float(end_time),
)


# =========================================================
# TIME SLIDER
# =========================================================

time = st.slider(
    "Time (seconds)",
    min_value=float(start_time),
    max_value=float(end_time),
    value=float(
        st.session_state.selected_time
    ),
    step=0.1,
    key="time_slider",
)

st.session_state.selected_time = time


# =========================================================
# CALCULATE CURRENT VALUES
# =========================================================

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
        f"The selected time t = {time:.2f} "
        "is outside the real-valued domain of "
        "this function or one of its derivatives."
    )

    st.stop()


# =========================================================
# MOTION STATE
# =========================================================

tolerance = 0.05

if velocity > tolerance:

    motion = "Moving forward"

elif velocity < -tolerance:

    motion = "Moving backward"

else:

    motion = "Momentarily stopped"


# =========================================================
# 3. INSTANTANEOUS MOTION
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "3. Instantaneous Motion"
    "</div>",
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

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

with metric4:

    st.metric(
        "Motion",
        motion,
    )


# =========================================================
# GRAPH DATA
# =========================================================

graph_t = np.linspace(
    start_time,
    end_time,
    600,
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


# =========================================================
# TANGENT LINE
# =========================================================

tangent_values = (
    velocity * (graph_t - time)
    + position
)


# =========================================================
# 4. POSITION GRAPH
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "4. Position and Tangent Line"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Drag the slider, or click a point on the position curve "
    "to examine that moment."
)


position_fig = go.Figure()


# Main curve

position_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=position_values,
        mode="lines+markers",
        name="Position",
        line=dict(
            width=3,
        ),
        marker=dict(
            size=4,
            opacity=0.0,
        ),
        customdata=graph_t,
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "s(t) = %{y:.2f}<extra></extra>"
        ),
    )
)


# Tangent

position_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=tangent_values,
        mode="lines",
        name="Tangent line",
        line=dict(
            width=2,
            dash="dash",
        ),
        hoverinfo="skip",
    )
)


# Current point

position_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Selected time",
        marker=dict(
            size=12,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "s(t) = %{y:.2f}<extra></extra>"
        ),
    )
)


position_fig.add_hline(
    y=0,
    line_width=1,
)

position_fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot",
)


position_fig.update_layout(
    height=520,
    margin=dict(
        l=15,
        r=15,
        t=30,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="Position",
    hovermode="closest",
    showlegend=True,
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
    },
)


# =========================================================
# PROCESS GRAPH SELECTION
# =========================================================

try:

    selected_points = (
        position_event.selection.points
    )

    if selected_points:

        selected_x = selected_points[-1]["x"]

        selected_x = float(selected_x)

        if (
            start_time
            <= selected_x
            <= end_time
        ):

            if abs(
                selected_x - st.session_state.selected_time
            ) > 0.001:

                st.session_state.selected_time = (
                    selected_x
                )

                st.rerun()

except Exception:
    pass


# =========================================================
# 5. TANGENT EXPLANATION
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "5. What Does the Tangent Line Mean?"
    "</div>",
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
    '<div class="explanation">'
    "The slope of the tangent line represents the "
    "<strong>instantaneous velocity</strong>. "
    "In calculus, this slope is the derivative of the "
    "position function."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# 6. VELOCITY AS DERIVATIVE
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "6. Velocity as the Derivative"
    "</div>",
    unsafe_allow_html=True,
)

st.latex(
    rf"v(t) = s'(t) = {sp.latex(velocity_function)}"
)

st.write(
    "Velocity tells us how quickly the object's "
    "position is changing."
)

st.latex(
    rf"v({time:.2f}) = {velocity:.2f}\ \mathrm{{m/s}}"
)


# =========================================================
# 7. VELOCITY GRAPH
# =========================================================

st.markdown(
    '<div class="section-heading">7. Velocity</div>',
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
            width=3,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "v(t) = %{y:.2f}<extra></extra>"
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
            size=11,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "v(t) = %{y:.2f}<extra></extra>"
        ),
    )
)

velocity_fig.add_hline(
    y=0,
    line_width=1,
)

velocity_fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot",
)

velocity_fig.update_layout(
    height=380,
    margin=dict(
        l=15,
        r=15,
        t=25,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="Velocity",
    hovermode="closest",
)

st.plotly_chart(
    velocity_fig,
    width="stretch",
    key="velocity_graph",
    config={
        "displaylogo": False,
        "scrollZoom": False,
    },
)


# =========================================================
# 8. ACCELERATION
# =========================================================

st.markdown(
    '<div class="section-heading">8. Acceleration</div>',
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


# =========================================================
# 9. ACCELERATION GRAPH
# =========================================================

st.markdown(
    '<div class="section-heading">9. Acceleration</div>',
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
            width=3,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "a(t) = %{y:.2f}<extra></extra>"
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
            size=11,
        ),
    )
)

acceleration_fig.add_hline(
    y=0,
    line_width=1,
)

acceleration_fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot",
)

acceleration_fig.update_layout(
    height=380,
    margin=dict(
        l=15,
        r=15,
        t=25,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="Acceleration",
    hovermode="closest",
)

st.plotly_chart(
    acceleration_fig,
    width="stretch",
    key="acceleration_graph",
    config={
        "displaylogo": False,
        "scrollZoom": False,
    },
)


# =========================================================
# 10. CALCULUS CONNECTION
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "10. The Calculus Connection"
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Motion Explorer demonstrates the relationship between "
    "position, velocity, and acceleration."
)

connection1, connection2, connection3 = st.columns(3)

with connection1:

    st.markdown("**Position**")

    st.latex(
        rf"s(t) = {sp.latex(position_function)}"
    )

    st.write(
        "Describes where the object is."
    )

with connection2:

    st.markdown("**Velocity**")

    st.latex(
        rf"v(t) = s'(t)"
    )

    st.write(
        "Describes how quickly position is changing."
    )

with connection3:

    st.markdown("**Acceleration**")

    st.latex(
        rf"a(t) = v'(t) = s''(t)"
    )

    st.write(
        "Describes how quickly velocity is changing."
    )


# =========================================================
# 11. TURNING POINTS
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "11. Explore Turning Points"
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "A turning point can occur when instantaneous velocity "
    "is zero. For a differentiable position function, "
    "this means"
)

st.latex(
    r"s'(t)=0"
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

    if valid_points:

        for (
            point_time,
            point_position,
        ) in valid_points:

            st.write(
                f"A critical point occurs at "
                f"**t = {point_time:.2f} seconds**."
            )

            c1, c2, c3 = st.columns(3)

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


# =========================================================
# 12. INTERPRET THE MOTION
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "12. Interpret the Motion"
    "</div>",
    unsafe_allow_html=True,
)

if velocity > tolerance:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is positive, "
        "so the object's position is increasing. "
        "It is moving forward."
    )

elif velocity < -tolerance:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is negative, "
        "so the object's position is decreasing. "
        "It is moving backward."
    )

else:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is approximately "
        "zero. The object is momentarily stopped."
    )

st.write(
    interpretation
)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "How It Works"
    "</div>",
    unsafe_allow_html=True,
)

with st.expander(
    "Step 1 — Define the position"
):

    st.write(
        "The program begins with a position function "
        "describing the object's location over time."
    )

    st.latex(
        rf"s(t) = {sp.latex(position_function)}"
    )


with st.expander(
    "Step 2 — Differentiate"
):

    st.write(
        "The application uses symbolic differentiation "
        "to obtain instantaneous velocity."
    )

    st.latex(
        rf"v(t) = s'(t) = "
        rf"{sp.latex(velocity_function)}"
    )


with st.expander(
    "Step 3 — Differentiate again"
):

    st.write(
        "Differentiating velocity gives acceleration."
    )

    st.latex(
        rf"a(t) = v'(t) = s''(t) = "
        rf"{sp.latex(acceleration_function)}"
    )


with st.expander(
    "Step 4 — Choose a time"
):

    st.write(
        "The slider selects a specific moment. "
        "You can also select a point directly on the "
        "position graph."
    )


with st.expander(
    "Step 5 — Build the tangent line"
):

    st.write(
        "The tangent line uses instantaneous velocity "
        "as its slope."
    )

    st.latex(
        r"y=v(t_0)(t-t_0)+s(t_0)"
    )

    st.write(
        "This makes the geometric meaning of the derivative "
        "visible: the derivative is the slope of the "
        "tangent line."
    )


# =========================================================
# WHY I BUILT THIS
# =========================================================

st.markdown(
    '<div class="section-heading">'
    "Why I Built This"
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Many students can calculate derivatives without "
    "developing an intuition for what a derivative "
    "actually represents."
)

st.write(
    "I built Motion Explorer to connect the algebraic "
    "derivative to physical motion by showing position, "
    "tangent slope, velocity, and acceleration together "
    "in an interactive visualization."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    "Motion Explorer · Built with Python, Streamlit, "
    "NumPy, Plotly, and SymPy"
    "</div>",
    unsafe_allow_html=True,
)
