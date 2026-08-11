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
    page_icon="📐",
    layout="wide",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 3.5rem;
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            font-size: 1.08rem;
            color: #6b7280;
            margin-bottom: 0.7rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 2rem;
            margin-bottom: 0.55rem;
        }

        .equation-box {
            background: #f7f7f8;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.85rem 1.15rem;
            margin: 0.6rem 0 1rem 0;
        }

        .explanation {
            background: #fafafa;
            border-left: 4px solid #555;
            padding: 0.9rem 1.15rem;
            border-radius: 4px;
            margin: 0.8rem 0;
        }

        .footer {
            text-align: center;
            color: #888;
            font-size: 0.82rem;
            margin-top: 3rem;
            padding-top: 1.3rem;
            border-top: 1px solid #eeeeee;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
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


# =========================================================
# FUNCTION PARSER
# =========================================================

def parse_function(expression):
    """
    Parse a user-entered mathematical expression using a
    restricted set of allowed mathematical characters/functions.
    """

    expression = expression.strip()

    if not expression:
        raise ValueError("Please enter a function.")

    # Basic input-length protection.
    if len(expression) > 120:
        raise ValueError("Please keep the function under 120 characters.")

    # Only allow mathematical characters, names, punctuation,
    # operators, decimal points, and spaces.
    if not re.fullmatch(
        r"[0-9a-zA-Z_+\-*/^().,\s]+",
        expression,
    ):
        raise ValueError(
            "This function contains an unsupported character."
        )

    # Only allow known function/name tokens.
    names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression)

    allowed_names = set(ALLOWED_LOCALS.keys())

    unknown_names = [
        name for name in names
        if name not in allowed_names
    ]

    if unknown_names:
        raise ValueError(
            "Unsupported name(s): "
            + ", ".join(sorted(set(unknown_names)))
            + "."
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
            "I couldn't interpret that expression. "
            "Try something like sin(t), e^t, ln(t), or t^3 - 2t."
        ) from exc

    # Make sure the expression actually depends on t.
    if t not in parsed.free_symbols:
        raise ValueError(
            "The position function must depend on t."
        )

    # Make sure it is real-valued where possible.
    return sp.simplify(parsed)


# =========================================================
# NUMERICAL HELPERS
# =========================================================

def safe_numeric_function(expr):
    """
    Convert a SymPy expression into a NumPy function.
    """

    return sp.lambdify(
        t,
        expr,
        modules=["numpy"],
    )


def evaluate_expression(expr, values):
    """
    Evaluate an expression and convert invalid/non-real values
    into NaN so Plotly can safely skip them.
    """

    func = safe_numeric_function(expr)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):
        result = func(values)

    result = np.asarray(result)

    if np.iscomplexobj(result):
        result = np.real_if_close(result)

    try:
        result = result.astype(float)
    except (TypeError, ValueError):
        return np.full_like(
            np.asarray(values, dtype=float),
            np.nan,
        )

    result[~np.isfinite(result)] = np.nan

    return result


def evaluate_at(expr, value):
    """
    Safely evaluate one expression at one time.
    """

    try:
        result = sp.N(expr.subs(t, value))

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
    '<div class="main-title">Motion Explorer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "An interactive visualization connecting calculus derivatives "
    "to physical motion."
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Explore how the derivative of a position function represents "
    "instantaneous velocity — and see that relationship directly on a graph."
)


# =========================================================
# POSITION FUNCTION
# =========================================================

st.markdown(
    '<div class="section-title">1. Position Function</div>',
    unsafe_allow_html=True,
)

function_col1, function_col2 = st.columns([1, 2])

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
# PARSE FUNCTION
# =========================================================

try:
    position_function = parse_function(custom_function)

    velocity_function = sp.simplify(
        sp.diff(position_function, t)
    )

    acceleration_function = sp.simplify(
        sp.diff(velocity_function, t)
    )

    function_error = None

except ValueError as error:
    function_error = str(error)


if function_error:

    st.error(function_error)
    st.stop()


# =========================================================
# DISPLAY SYMBOLIC EQUATIONS
# =========================================================

st.markdown(
    '<div class="equation-box">',
    unsafe_allow_html=True,
)

st.latex(
    rf"s(t) = {sp.latex(position_function)}"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "The position function describes where the object is located "
    "at each moment in time."
)


# =========================================================
# TIME RANGE
# =========================================================

st.markdown(
    '<div class="section-title">2. Choose a Moment in Time</div>',
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
    st.error("End time must be greater than start time.")
    st.stop()


# =========================================================
# TIME SLIDER
# =========================================================

default_time = start_time + (end_time - start_time) * 0.25

time = st.slider(
    "Time (seconds)",
    min_value=float(start_time),
    max_value=float(end_time),
    value=float(default_time),
    step=0.1,
)


# =========================================================
# CALCULATE VALUES
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


# =========================================================
# HANDLE UNDEFINED POINT
# =========================================================

if position is None or velocity is None or acceleration is None:

    st.warning(
        f"The selected time t = {time:.2f} is outside the real-valued "
        "domain of this function or one of its derivatives. "
        "Move the slider to a valid time."
    )

    st.stop()


# =========================================================
# MOTION INTERPRETATION
# =========================================================

tolerance = 0.05

if velocity > tolerance:
    motion = "Moving forward"
elif velocity < -tolerance:
    motion = "Moving backward"
else:
    motion = "Momentarily stopped"


# =========================================================
# INSTANTANEOUS MOTION
# =========================================================

st.markdown(
    '<div class="section-title">3. Instantaneous Motion</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Position",
        f"{position:.2f} m",
    )

with col2:
    st.metric(
        "Velocity",
        f"{velocity:.2f} m/s",
    )

with col3:
    st.metric(
        "Acceleration",
        f"{acceleration:.2f} m/s²",
    )

with col4:
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
    800,
)

position_values = evaluate_expression(
    position_function,
    graph_t,
)

velocity_values = evaluate_expression(
    velocity_function,
    graph_t,
)

acceleration_values = evaluate_expression(
    acceleration_function,
    graph_t,
)


# =========================================================
# TANGENT LINE
# =========================================================

tangent_values = (
    velocity * (graph_t - time) + position
)


# =========================================================
# MAIN POSITION GRAPH
# =========================================================

st.markdown(
    '<div class="section-title">4. Position and Tangent Line</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=position_values,
        mode="lines",
        name="Position s(t)",
        line=dict(width=3),
    )
)

fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=tangent_values,
        mode="lines",
        name="Tangent line",
        line=dict(
            width=2,
            dash="dash",
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Current position",
        marker=dict(size=12),
    )
)

fig.add_hline(
    y=0,
    line_width=1,
)

fig.add_vline(
    x=time,
    line_width=1,
    line_dash="dot",
)

fig.update_layout(
    height=520,
    margin=dict(
        l=20,
        r=20,
        t=35,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="Position",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# =========================================================
# TANGENT EXPLANATION
# =========================================================

st.markdown(
    '<div class="section-title">5. What Does the Tangent Line Mean?</div>',
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
    "The slope of the tangent line represents the object's "
    "<strong>instantaneous velocity</strong>. "
    "In calculus, this slope is the derivative of the position function."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# VELOCITY
# =========================================================

st.markdown(
    '<div class="section-title">6. Velocity as the Derivative</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="equation-box">',
    unsafe_allow_html=True,
)

st.latex(
    rf"v(t) = s'(t) = {sp.latex(velocity_function)}"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Velocity tells us how quickly the object's position is changing."
)

st.latex(
    rf"v({time:.2f}) = {velocity:.2f}\ \mathrm{{m/s}}"
)


# =========================================================
# VELOCITY GRAPH
# =========================================================

st.markdown(
    '<div class="section-title">7. Velocity</div>',
    unsafe_allow_html=True,
)

velocity_fig = go.Figure()

velocity_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=velocity_values,
        mode="lines",
        name="Velocity v(t)",
        line=dict(width=3),
    )
)

velocity_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[velocity],
        mode="markers",
        name="Selected time",
        marker=dict(size=11),
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
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="Velocity",
    hovermode="x unified",
)

st.plotly_chart(
    velocity_fig,
    use_container_width=True,
)


# =========================================================
# ACCELERATION
# =========================================================

st.markdown(
    '<div class="section-title">8. Acceleration</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="equation-box">',
    unsafe_allow_html=True,
)

st.latex(
    rf"a(t) = v'(t) = s''(t) = {sp.latex(acceleration_function)}"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Acceleration describes how quickly velocity changes."
)

st.latex(
    rf"a({time:.2f}) = {acceleration:.2f}\ \mathrm{{m/s^2}}"
)


# =========================================================
# ACCELERATION GRAPH
# =========================================================

st.markdown(
    '<div class="section-title">9. Acceleration</div>',
    unsafe_allow_html=True,
)

acceleration_fig = go.Figure()

acceleration_fig.add_trace(
    go.Scatter(
        x=graph_t,
        y=acceleration_values,
        mode="lines",
        name="Acceleration a(t)",
        line=dict(width=3),
    )
)

acceleration_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[acceleration],
        mode="markers",
        name="Selected time",
        marker=dict(size=11),
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
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="Acceleration",
    hovermode="x unified",
)

st.plotly_chart(
    acceleration_fig,
    use_container_width=True,
)


# =========================================================
# CALCULUS CONNECTION
# =========================================================

st.markdown(
    '<div class="section-title">10. The Calculus Connection</div>',
    unsafe_allow_html=True,
)

st.write(
    "Motion Explorer demonstrates the relationship between "
    "position, velocity, and acceleration."
)

connection_col1, connection_col2, connection_col3 = st.columns(3)

with connection_col1:
    st.markdown("### Position")
    st.latex(
        rf"s(t) = {sp.latex(position_function)}"
    )
    st.write(
        "Describes where the object is."
    )

with connection_col2:
    st.markdown("### Velocity")
    st.latex(
        rf"v(t) = s'(t) = {sp.latex(velocity_function)}"
    )
    st.write(
        "Describes how quickly position is changing."
    )

with connection_col3:
    st.markdown("### Acceleration")
    st.latex(
        rf"a(t) = v'(t) = s''(t) = {sp.latex(acceleration_function)}"
    )
    st.write(
        "Describes how quickly velocity is changing."
    )


# =========================================================
# TURNING POINTS
# =========================================================

st.markdown(
    '<div class="section-title">11. Explore Turning Points</div>',
    unsafe_allow_html=True,
)

st.write(
    "A turning point can occur when instantaneous velocity is zero. "
    "For a differentiable position function, this means "
    "s′(t) = 0."
)

try:
    critical_points = sp.solve(
        sp.Eq(velocity_function, 0),
        t,
    )

    valid_points = []

    for point in critical_points:

        if point.is_real is False:
            continue

        try:
            point_value = float(point)

            if (
                start_time <= point_value <= end_time
                and np.isfinite(point_value)
            ):
                point_position = evaluate_at(
                    position_function,
                    point_value,
                )

                if point_position is not None:
                    valid_points.append(
                        (
                            point_value,
                            point_position,
                        )
                    )

        except (TypeError, ValueError):
            continue

    if valid_points:

        for point_time, point_position in valid_points:

            st.write(
                f"A critical point occurs at approximately "
                f"**t = {point_time:.2f} seconds**."
            )

            turn_col1, turn_col2, turn_col3 = st.columns(3)

            with turn_col1:
                st.metric(
                    "Time",
                    f"{point_time:.2f} s",
                )

            with turn_col2:
                st.metric(
                    "Position",
                    f"{point_position:.2f} m",
                )

            with turn_col3:
                st.metric(
                    "Velocity",
                    "0.00 m/s",
                )

    else:

        st.info(
            "No turning points where v(t) = 0 were found in "
            "the selected time interval."
        )

except Exception:

    st.info(
        "Turning points could not be determined symbolically "
        "for this function."
    )


# =========================================================
# INTERPRET THE MOTION
# =========================================================

st.markdown(
    '<div class="section-title">12. Interpret the Motion</div>',
    unsafe_allow_html=True,
)

if velocity > tolerance:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is positive, so the "
        "object's position is increasing. It is moving forward."
    )

elif velocity < -tolerance:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is negative, so the "
        "object's position is decreasing. It is moving backward."
    )

else:

    interpretation = (
        f"At **t = {time:.2f} s**, velocity is approximately zero. "
        "The object is momentarily stopped."
    )

st.info(interpretation)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True,
)

with st.expander("Step 1 — Define the position"):

    st.write(
        "The program begins with a position function describing "
        "the object's location over time."
    )

    st.latex(
        rf"s(t) = {sp.latex(position_function)}"
    )


with st.expander("Step 2 — Differentiate"):

    st.write(
        "The application uses symbolic differentiation to obtain "
        "the instantaneous velocity."
    )

    st.latex(
        rf"v(t) = s'(t) = {sp.latex(velocity_function)}"
    )


with st.expander("Step 3 — Differentiate again"):

    st.write(
        "Differentiating velocity gives acceleration."
    )

    st.latex(
        rf"a(t) = v'(t) = s''(t) = {sp.latex(acceleration_function)}"
    )


with st.expander("Step 4 — Choose a time"):

    st.write(
        "The slider selects a specific moment. The application "
        "evaluates the position, velocity, and acceleration at "
        "that moment."
    )


with st.expander("Step 5 — Build the tangent line"):

    st.write(
        "The tangent line uses instantaneous velocity as its slope."
    )

    st.latex(
        r"y = v(t_0)(t-t_0)+s(t_0)"
    )

    st.write(
        "This makes the geometric meaning of the derivative visible: "
        "the derivative is the slope of the tangent line."
    )


# =========================================================
# WHY I BUILT THIS
# =========================================================

st.markdown(
    '<div class="section-title">Why I Built This</div>',
    unsafe_allow_html=True,
)

st.write(
    "Many students can calculate derivatives without developing "
    "an intuition for what a derivative actually represents."
)

st.write(
    "I built Motion Explorer to connect the algebraic derivative "
    "to physical motion by showing position, tangent slope, "
    "velocity, and acceleration together in an interactive visualization."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    "Motion Explorer · Built with Python, Streamlit, NumPy, Plotly, and SymPy"
    "</div>",
    unsafe_allow_html=True,
)
