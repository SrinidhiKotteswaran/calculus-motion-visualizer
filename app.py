import re
import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Motion Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }

    html, body, [class*="css"] {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    /* Reduce Streamlit's default vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .app-kicker {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8b8f98;
        margin: 0 0 0.35rem 0;
    }

    .app-title {
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.045em;
        margin: 0;
        color: #17181b;
    }

    .app-description {
        max-width: 700px;
        margin-top: 0.55rem;
        margin-bottom: 1.65rem;
        color: #6b7078;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    /* ========================================================
       SECTION HIERARCHY
       ======================================================== */

    .section-number {
        color: #a1a5ad;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-top: 1.65rem;
        margin-bottom: 0.12rem;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 650;
        letter-spacing: -0.018em;
        line-height: 1.25;
        margin: 0;
        color: #202226;
    }

    .section-description {
        color: #747982;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 0.22rem;
        margin-bottom: 0.65rem;
    }

    /* First section sits closer to header */
    .app-description + .section-number {
        margin-top: 0.4rem;
    }

    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-testid="stTextInput"] {
        margin-top: 0;
    }

    div[data-testid="stSelectbox"] {
        margin-top: 0;
    }

    div[data-testid="stNumberInput"] {
        margin-top: 0;
    }

    div[data-testid="stSlider"] {
        padding-top: 0.15rem;
        padding-bottom: 0.2rem;
    }

    label[data-testid="stWidgetLabel"] p {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: #686d75 !important;
    }

    /* Cleaner input boxes */
    div[data-baseweb="input"] {
        border-radius: 7px;
    }

    div[data-baseweb="select"] {
        border-radius: 7px;
    }

    /* ========================================================
       FUNCTION
       ======================================================== */

    .function-display {
        font-size: 1.05rem;
        padding: 0.45rem 0;
        margin-bottom: 0.15rem;
    }

    /* ========================================================
       MOTION SUMMARY
       ======================================================== */

    .summary-label {
        font-size: 0.65rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #858a92;
        margin-bottom: 0.18rem;
    }

    .summary-value {
        font-size: 1.16rem;
        font-weight: 650;
        line-height: 1.2;
        letter-spacing: -0.015em;
        color: #202226;
    }

    .summary-unit {
        font-size: 0.76rem;
        font-weight: 450;
        color: #858a92;
    }

    /* ========================================================
       TEXT
       ======================================================== */

    .muted {
        color: #7a7f87;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .concept-note {
        border-left: 2px solid #c7cbd1;
        padding: 0.15rem 0 0.15rem 0.8rem;
        margin: 0.55rem 0 0.8rem 0;
        color: #555a62;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        padding: 0.55rem 0.7rem;
        border: 1px solid #e7e8eb;
        border-radius: 7px;
        background: rgba(250, 250, 251, 0.65);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        color: #7b8088 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        font-weight: 650 !important;
        color: #24262a !important;
    }

    /* ========================================================
       PLOTLY
       ======================================================== */

    div[data-testid="stPlotlyChart"] {
        margin-top: 0.1rem;
        margin-bottom: 0.35rem;
    }

    /* ========================================================
       COLUMNS
       ======================================================== */

    div[data-testid="column"] {
        padding-left: 0.35rem;
        padding-right: 0.35rem;
    }

    /* ========================================================
       EXPANDERS
       ======================================================== */

    div[data-testid="stExpander"] {
        border: 1px solid #e6e7e9;
        border-radius: 8px;
        margin-top: 1.1rem;
    }

    div[data-testid="stExpander"] summary {
        font-size: 0.84rem;
        font-weight: 600;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 7px;
        font-size: 0.84rem;
    }

    /* ========================================================
       LATEX
       ======================================================== */

    .stLatex {
        margin-top: 0.15rem;
        margin-bottom: 0.15rem;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    button[kind="secondary"] {
        border-radius: 6px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        margin-top: 3rem;
        padding-top: 0.8rem;
        border-top: 1px solid #e7e8eb;
        color: #9a9ea5;
        font-size: 0.68rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-title {
            font-size: 2rem;
        }

        .app-description {
            font-size: 0.88rem;
            margin-bottom: 1.2rem;
        }

        .section-number {
            margin-top: 1.35rem;
        }

        .section-title {
            font-size: 1.05rem;
        }

        .summary-value {
            font-size: 1rem;
        }

        .footer {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.25rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SYMBOLIC SETUP
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
    "Exponential decay": "e^(-t)",
    "Logarithmic": "ln(t)",
    "Power": "t^2.5",
    "Rational": "1/t",
    "Damped oscillation": "e^(-0.2t)*sin(t)",
}


# ============================================================
# PARSING
# ============================================================

def parse_function(expression: str):

    expression = expression.strip()

    if not expression:
        raise ValueError("Enter a position function.")

    if len(expression) > 150:
        raise ValueError("Keep the function under 150 characters.")

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

    unknown = [
        name
        for name in names
        if name not in ALLOWED_LOCALS
    ]

    if unknown:
        raise ValueError(
            "Unsupported function name: "
            + ", ".join(sorted(set(unknown)))
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
            "Try sin(t), e^t, ln(t), or t^3 - 3t."
        ) from exc

    if t not in parsed.free_symbols:
        raise ValueError(
            "The position function must depend on t."
        )

    return sp.sympify(parsed)


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def lambdify_expression(expr):
    return sp.lambdify(
        t,
        expr,
        modules=["numpy"],
    )


def evaluate_array(expr, values):

    fn = lambdify_expression(expr)

    values = np.asarray(values, dtype=float)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):
        result = fn(values)

    result = np.asarray(result)

    if np.iscomplexobj(result):
        result = np.real_if_close(result)

        if np.iscomplexobj(result):
            return np.full(
                values.shape,
                np.nan,
            )

    try:
        result = result.astype(float)
    except (TypeError, ValueError):
        return np.full(
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
                float(value),
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
# GRAPH HELPERS
# ============================================================

GRID = "rgba(128,128,128,0.20)"

POSITION_COLOR = "#2563eb"
TANGENT_COLOR = "#dc2626"
VELOCITY_COLOR = "#059669"
ACCELERATION_COLOR = "#7c3aed"
POINT_COLOR = "#f59e0b"
ZERO_COLOR = "#6b7280"


def base_layout(
    height,
    x_title,
    y_title,
):

    return dict(
        height=height,
        margin=dict(
            l=45,
            r=25,
            t=20,
            b=45,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
        ),

        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
        ),

        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-kicker">Calculus laboratory</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="app-title">Motion Explorer</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-description">
        Explore the derivative through physical motion.
        Change a position function, move through time, and
        watch position, velocity, acceleration, and tangent
        slope change together.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1 — FUNCTION
# ============================================================

st.markdown(
    '<div class="section-number">01</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Position function</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Choose an example or define your own."
    "</div>",
    unsafe_allow_html=True,
)

function_col1, function_col2 = st.columns(
    [1, 2]
)

with function_col1:

    preset = st.selectbox(
        "Example",
        list(PRESETS.keys()),
        label_visibility="collapsed",
    )

with function_col2:

    custom_function = st.text_input(
        "Position function",
        value=PRESETS[preset],
        label_visibility="collapsed",
        placeholder="e.g. -t^2 + 8t",
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


st.latex(
    rf"s(t) = {sp.latex(position_function)}"
)

st.markdown(
    '<div class="muted">'
    "Position describes where the object is at time t."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# 2 — TIME
# ============================================================

st.markdown(
    '<div class="section-number">02</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Choose a moment</div>',
    unsafe_allow_html=True,
)

time_col1, time_col2 = st.columns(2)

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


if (
    "selected_time" not in st.session_state
    or not (
        start_time
        <= st.session_state.selected_time
        <= end_time
    )
):

    st.session_state.selected_time = (
        start_time + 0.25 * (
            end_time - start_time
        )
    )


time = st.slider(
    "Time",
    min_value=float(start_time),
    max_value=float(end_time),
    value=float(
        st.session_state.selected_time
    ),
    step=0.05,
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
        f"The selected time t = {time:.2f} "
        "is outside the real-valued domain of "
        "this function or one of its derivatives."
    )

    st.stop()


# ============================================================
# MOTION CLASSIFICATION
# ============================================================

TOLERANCE = 1e-7

if velocity > TOLERANCE:
    motion = "Moving forward"
elif velocity < -TOLERANCE:
    motion = "Moving backward"
else:
    motion = "Momentarily stopped"


# ============================================================
# 3 — CURRENT MOTION
# ============================================================

st.markdown(
    '<div class="section-number">03</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Instantaneous motion</div>',
    unsafe_allow_html=True,
)

summary_cols = st.columns(4)

with summary_cols[0]:

    st.markdown(
        '<div class="summary-label">Position</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="summary-value">'
        f'{position:.2f} '
        f'<span class="summary-unit">m</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with summary_cols[1]:

    st.markdown(
        '<div class="summary-label">Velocity</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="summary-value">'
        f'{velocity:.2f} '
        f'<span class="summary-unit">m/s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with summary_cols[2]:

    st.markdown(
        '<div class="summary-label">Acceleration</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="summary-value">'
        f'{acceleration:.2f} '
        f'<span class="summary-unit">m/s²</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with summary_cols[3]:

    st.markdown(
        '<div class="summary-label">Motion</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="summary-value">'
        f'{motion}'
        f'</div>',
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
# 4 — POSITION / TANGENT
# ============================================================

st.markdown(
    '<div class="section-number">04</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Position and tangent"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "The tangent line shows the instantaneous rate of change "
    "of position at the selected time."
    "</div>",
    unsafe_allow_html=True,
)


tangent_values = (
    position
    + velocity * (graph_t - time)
)


position_fig = go.Figure()


position_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=position_values,
        mode="lines",
        name="Position",
        line=dict(
            color=POSITION_COLOR,
            width=3,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "s(t) = %{y:.3f}"
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
            color=TANGENT_COLOR,
            width=2,
            dash="dash",
        ),
        hoverinfo="skip",
    )
)


position_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Selected time",
        marker=dict(
            color=POINT_COLOR,
            size=12,
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "s(t) = %{y:.3f}"
            "<extra></extra>"
        ),
    )
)


position_fig.add_vline(
    x=time,
    line_width=1.3,
    line_dash="dot",
    line_color=POINT_COLOR,
)


position_fig.add_hline(
    y=0,
    line_width=1,
    line_color=ZERO_COLOR,
)


position_fig.update_layout(
    **base_layout(
        510,
        "Time",
        "Position",
    )
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


# ============================================================
# GRAPH CLICK
# ============================================================

try:

    points = position_event.selection.points

    if points:

        selected_x = float(
            points[-1]["x"]
        )

        if (
            start_time
            <= selected_x
            <= end_time
        ):

            if abs(
                selected_x
                - st.session_state.selected_time
            ) > 0.01:

                st.session_state.selected_time = (
                    selected_x
                )

                st.rerun()

except Exception:
    pass


# ============================================================
# 5 — TANGENT CONCEPT
# ============================================================

st.markdown(
    '<div class="section-number">05</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "What does the tangent line mean?"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="concept-note">
        At <strong>t = {time:.2f} s</strong>, the object is at
        <strong>s(t) = {position:.2f} m</strong>.
        The tangent line has slope
        <strong>{velocity:.2f} m/s</strong>.
        That slope is the instantaneous velocity.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6 — VELOCITY
# ============================================================

st.markdown(
    '<div class="section-number">06</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Velocity is the derivative"
    "</div>",
    unsafe_allow_html=True,
)

st.latex(
    rf"v(t)=s'(t)={sp.latex(velocity_function)}"
)

st.write(
    "Velocity measures how quickly position changes."
)

st.latex(
    rf"v({time:.2f})={velocity:.3f}\ \mathrm{{m/s}}"
)


velocity_fig = go.Figure()


velocity_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=velocity_values,
        mode="lines",
        name="Velocity",
        line=dict(
            color=VELOCITY_COLOR,
            width=3,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "v(t) = %{y:.3f}"
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
            color=POINT_COLOR,
            size=11,
            line=dict(
                color="white",
                width=2,
            ),
        ),
    )
)


velocity_fig.add_vline(
    x=time,
    line_width=1.3,
    line_dash="dot",
    line_color=POINT_COLOR,
)

velocity_fig.add_hline(
    y=0,
    line_width=1,
    line_color=ZERO_COLOR,
)


velocity_fig.update_layout(
    **base_layout(
        390,
        "Time",
        "Velocity",
    )
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


# ============================================================
# 7 — ACCELERATION
# ============================================================

st.markdown(
    '<div class="section-number">07</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Acceleration is the second derivative"
    "</div>",
    unsafe_allow_html=True,
)

st.latex(
    rf"a(t)=v'(t)=s''(t)={sp.latex(acceleration_function)}"
)

st.write(
    "Acceleration measures how quickly velocity changes."
)

st.latex(
    rf"a({time:.2f})={acceleration:.3f}\ "
    rf"\mathrm{{m/s^2}}"
)


acceleration_fig = go.Figure()


acceleration_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=acceleration_values,
        mode="lines",
        name="Acceleration",
        line=dict(
            color=ACCELERATION_COLOR,
            width=3,
        ),
        hovertemplate=(
            "t = %{x:.2f}<br>"
            "a(t) = %{y:.3f}"
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
            color=POINT_COLOR,
            size=11,
            line=dict(
                color="white",
                width=2,
            ),
        ),
    )
)


acceleration_fig.add_vline(
    x=time,
    line_width=1.3,
    line_dash="dot",
    line_color=POINT_COLOR,
)

acceleration_fig.add_hline(
    y=0,
    line_width=1,
    line_color=ZERO_COLOR,
)


acceleration_fig.update_layout(
    **base_layout(
        390,
        "Time",
        "Acceleration",
    )
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


# ============================================================
# 8 — SECANT TO TANGENT LAB
# ============================================================

st.markdown(
    '<div class="section-number">08</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Secant → tangent"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "See the derivative emerge as the secant interval approaches zero."
    "</div>",
    unsafe_allow_html=True,
)


secant_col1, secant_col2 = st.columns(
    [2, 1]
)

with secant_col1:

    h = st.slider(
        "Interval h",
        min_value=0.001,
        max_value=2.0,
        value=0.5,
        step=0.001,
    )

with secant_col2:

    st.markdown(
        '<div class="muted" style="margin-top:1.8rem">'
        "Smaller h → tangent"
        "</div>",
        unsafe_allow_html=True,
    )


second_time = time + h

if second_time > end_time:

    second_time = end_time

    actual_h = second_time - time

else:

    actual_h = h


second_position = evaluate_at(
    position_function,
    second_time,
)


if (
    second_position is not None
    and actual_h > 0
):

    secant_slope = (
        second_position - position
    ) / actual_h

else:

    secant_slope = None


secant_fig = go.Figure()


secant_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=position_values,
        mode="lines",
        name="Position",
        line=dict(
            color=POSITION_COLOR,
            width=3,
        ),
        hoverinfo="skip",
    )
)


if secant_slope is not None:

    secant_line_x = np.array(
        [time, second_time]
    )

    secant_line_y = (
        position
        + secant_slope
        * (secant_line_x - time)
    )

    secant_fig.add_trace(
        go.Scatter(
            x=secant_line_x,
            y=secant_line_y,
            mode="lines",
            name="Secant",
            line=dict(
                color="#0891b2",
                width=2.5,
            ),
        )
    )

    secant_fig.add_trace(
        go.Scatter(
            x=[
                time,
                second_time,
            ],
            y=[
                position,
                second_position,
            ],
            mode="markers",
            name="Secant points",
            marker=dict(
                color="#0891b2",
                size=9,
            ),
        )
    )


secant_tangent_y = (
    position
    + velocity
    * (graph_t - time)
)


secant_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=secant_tangent_y,
        mode="lines",
        name="Tangent",
        line=dict(
            color=TANGENT_COLOR,
            width=2,
            dash="dash",
        ),
    )
)


secant_fig.update_layout(
    **base_layout(
        470,
        "Time",
        "Position",
    )
)


st.plotly_chart(
    secant_fig,
    width="stretch",
    key="secant_graph",
    config={
        "displaylogo": False,
        "scrollZoom": False,
    },
)


if secant_slope is not None:

    comparison_cols = st.columns(3)

    with comparison_cols[0]:

        st.metric(
            "Secant slope",
            f"{secant_slope:.5f}",
        )

    with comparison_cols[1]:

        st.metric(
            "Tangent slope",
            f"{velocity:.5f}",
        )

    with comparison_cols[2]:

        difference = abs(
            secant_slope - velocity
        )

        st.metric(
            "Difference",
            f"{difference:.5f}",
        )


st.latex(
    rf"\frac{{s(t+h)-s(t)}}{{h}}"
    rf"\;\longrightarrow\;"
    rf"s'(t)"
    rf"\quad\text{{as }}h\to0"
)


# ============================================================
# 9 — MOTION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-number">09</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Motion analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Use the sign of velocity to determine when position increases or decreases."
    "</div>",
    unsafe_allow_html=True,
)


# Numerical sign classification

valid_mask = np.isfinite(
    velocity_values
)

positive_mask = (
    valid_mask
    & (velocity_values > 1e-7)
)

negative_mask = (
    valid_mask
    & (velocity_values < -1e-7)
)


analysis_cols = st.columns(2)

with analysis_cols[0]:

    st.markdown("**Position increasing**")

    if np.any(positive_mask):

        positive_times = graph_t[
            positive_mask
        ]

        st.write(
            f"Approximately "
            f"{positive_times.min():.2f} "
            f"to "
            f"{positive_times.max():.2f} s"
        )

    else:

        st.write(
            "No interval detected."
        )

with analysis_cols[1]:

    st.markdown("**Position decreasing**")

    if np.any(negative_mask):

        negative_times = graph_t[
            negative_mask
        ]

        st.write(
            f"Approximately "
            f"{negative_times.min():.2f} "
            f"to "
            f"{negative_times.max():.2f} s"
        )

    else:

        st.write(
            "No interval detected."
        )


# ============================================================
# 10 — CRITICAL POINTS
# ============================================================

st.markdown(
    '<div class="section-number">10</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Critical points"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Critical points occur where the derivative is zero or undefined, "
    "provided the point lies in the domain of the original function."
    "</div>",
    unsafe_allow_html=True,
)


def find_critical_points(
    position_expr,
    derivative_expr,
    lower,
    upper,
):

    candidates = []

    # --------------------------------------------------------
    # Symbolic solutions
    # --------------------------------------------------------

    try:

        symbolic_solutions = sp.solveset(
            sp.Eq(
                derivative_expr,
                0,
            ),
            t,
            domain=sp.S.Reals,
        )

        if symbolic_solutions is not sp.S.EmptySet:

            if isinstance(
                symbolic_solutions,
                sp.FiniteSet,
            ):

                for solution in symbolic_solutions:

                    if solution.is_real is False:
                        continue

                    try:

                        value = float(solution)

                        if (
                            lower
                            <= value
                            <= upper
                        ):
                            candidates.append(
                                value
                            )

                    except Exception:
                        pass

    except Exception:
        pass

    # --------------------------------------------------------
    # Numerical fallback
    # --------------------------------------------------------

    sample_t = np.linspace(
        lower,
        upper,
        2500,
    )

    derivative_values = evaluate_array(
        derivative_expr,
        sample_t,
    )

    for i in range(
        len(sample_t) - 1
    ):

        y1 = derivative_values[i]
        y2 = derivative_values[i + 1]

        if (
            not np.isfinite(y1)
            or not np.isfinite(y2)
        ):
            continue

        if y1 == 0:

            candidates.append(
                sample_t[i]
            )

        elif y1 * y2 < 0:

            left = sample_t[i]
            right = sample_t[i + 1]

            # Bisection
            for _ in range(40):

                middle = (
                    left + right
                ) / 2

                middle_value = evaluate_at(
                    derivative_expr,
                    middle,
                )

                left_value = evaluate_at(
                    derivative_expr,
                    left,
                )

                if (
                    middle_value is None
                    or left_value is None
                ):
                    break

                if (
                    abs(middle_value)
                    < 1e-10
                ):
                    left = middle
                    right = middle
                    break

                if (
                    left_value
                    * middle_value
                    <= 0
                ):
                    right = middle
                else:
                    left = middle

            candidates.append(
                (left + right) / 2
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    candidates = sorted(
        candidates
    )

    unique = []

    for value in candidates:

        if not unique or abs(
            value - unique[-1]
        ) > 1e-3:

            original_value = evaluate_at(
                position_expr,
                value,
            )

            derivative_value = evaluate_at(
                derivative_expr,
                value,
            )

            if (
                original_value is not None
                and derivative_value is not None
            ):

                unique.append(
                    value
                )

    return unique


critical_points = find_critical_points(
    position_function,
    velocity_function,
    start_time,
    end_time,
)


if critical_points:

    for index, point_time in enumerate(
        critical_points,
        start=1,
    ):

        point_position = evaluate_at(
            position_function,
            point_time,
        )

        st.write(
            f"**Critical point {index}**"
        )

        cp_cols = st.columns(3)

        with cp_cols[0]:

            st.metric(
                "Time",
                f"{point_time:.3f} s",
            )

        with cp_cols[1]:

            st.metric(
                "Position",
                f"{point_position:.3f} m",
            )

        with cp_cols[2]:

            st.metric(
                "Velocity",
                "≈ 0 m/s",
            )

else:

    st.write(
        "No critical points were detected in the selected interval."
    )


# ============================================================
# 11 — CALCULUS CONNECTION
# ============================================================

st.markdown(
    '<div class="section-number">11</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "The calculus connection"
    "</div>",
    unsafe_allow_html=True,
)

connection_cols = st.columns(3)

with connection_cols[0]:

    st.markdown("**Position**")

    st.latex(
        rf"s(t)={sp.latex(position_function)}"
    )

    st.write(
        "Where the object is."
    )

with connection_cols[1]:

    st.markdown("**Velocity**")

    st.latex(
        rf"v(t)=s'(t)"
    )

    st.write(
        "How quickly position changes."
    )

with connection_cols[2]:

    st.markdown("**Acceleration**")

    st.latex(
        rf"a(t)=s''(t)"
    )

    st.write(
        "How quickly velocity changes."
    )


# ============================================================
# 12 — INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-number">12</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Interpret the motion"
    "</div>",
    unsafe_allow_html=True,
)


if velocity > TOLERANCE:

    interpretation = (
        f"At t = {time:.2f} s, velocity is positive, "
        "so position is increasing. The object is moving forward."
    )

elif velocity < -TOLERANCE:

    interpretation = (
        f"At t = {time:.2f} s, velocity is negative, "
        "so position is decreasing. The object is moving backward."
    )

else:

    interpretation = (
        f"At t = {time:.2f} s, velocity is approximately zero. "
        "The object is momentarily stopped."
    )


st.markdown(
    f'<div class="concept-note">'
    f'{interpretation}'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander("How the visualization works"):

    st.markdown(
        """
        **1. Define position**

        The user supplies a position function \(s(t)\).

        **2. Differentiate**

        SymPy calculates the velocity function

        \[
        v(t)=s'(t)
        \]

        **3. Differentiate again**

        The acceleration is

        \[
        a(t)=s''(t)
        \]

        **4. Select a time**

        The selected time determines the point on the
        position graph.

        **5. Construct the tangent**

        The tangent line uses instantaneous velocity as
        its slope.

        **6. Compare secant and tangent slopes**

        The secant slope

        \[
        \frac{s(t+h)-s(t)}{h}
        \]

        approaches the tangent slope as \(h\) approaches zero.
        """
    )


# ============================================================
# ABOUT
# ============================================================

with st.expander("About Motion Explorer"):

    st.write(
        "Motion Explorer was built to make the conceptual "
        "meaning of derivatives visible. Instead of treating "
        "differentiation as an isolated symbolic procedure, "
        "the application connects equations, slopes, graphs, "
        "and physical motion."
    )

    st.write(
        "Built with Python, Streamlit, SymPy, NumPy, and Plotly."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <span>Motion Explorer</span>
        <span>Python · SymPy · NumPy · Plotly · Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)
