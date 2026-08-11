import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Motion Explorer",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    html, body, [class*="css"] {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .kicker {
        color: #8b8f98;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        color: #17181b;
        font-size: 2.7rem;
        font-weight: 750;
        letter-spacing: -0.055em;
        line-height: 1;
        margin: 0;
    }

    .hero-description {
        max-width: 760px;
        color: #6d727b;
        font-size: 0.96rem;
        line-height: 1.6;
        margin-top: 0.7rem;
        margin-bottom: 1.8rem;
    }

    /* --------------------------------------------------------
       SECTIONS
    -------------------------------------------------------- */

    .section-number {
        color: #a2a6ae;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        margin-top: 1.8rem;
        margin-bottom: 0.15rem;
    }

    .section-title {
        color: #202226;
        font-size: 1.18rem;
        font-weight: 680;
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin: 0;
    }

    .section-description {
        color: #747982;
        font-size: 0.84rem;
        line-height: 1.55;
        margin-top: 0.25rem;
        margin-bottom: 0.7rem;
    }

    /* --------------------------------------------------------
       FUNCTION DISPLAY
    -------------------------------------------------------- */

    .function-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #fafafa;
        padding: 0.75rem 1rem;
        margin-top: 0.35rem;
        margin-bottom: 0.7rem;
    }

    .function-label {
        color: #8a8f97;
        font-size: 0.64rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        border: 1px solid #e6e7ea;
        border-radius: 9px;
        background: rgba(250, 250, 251, 0.75);
        padding: 0.7rem 0.8rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #7c8189 !important;
        font-size: 0.67rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #202226 !important;
        font-size: 1.08rem !important;
        font-weight: 680 !important;
    }

    /* --------------------------------------------------------
       CONCEPT CARDS
    -------------------------------------------------------- */

    .concept-card {
        border-left: 2px solid #c8ccd2;
        padding: 0.2rem 0 0.2rem 0.85rem;
        margin: 0.55rem 0 0.85rem;
        color: #555a62;
        font-size: 0.87rem;
        line-height: 1.6;
    }

    .insight-card {
        border: 1px solid #e6e7ea;
        border-radius: 9px;
        padding: 0.85rem 1rem;
        background: #fafafa;
        height: 100%;
    }

    .insight-label {
        color: #8a8f97;
        font-size: 0.64rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .insight-value {
        color: #25272b;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    /* --------------------------------------------------------
       GRAPH
    -------------------------------------------------------- */

    div[data-testid="stPlotlyChart"] {
        margin-top: 0.15rem;
        margin-bottom: 0.3rem;
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    button {
        border-radius: 7px !important;
    }

    /* --------------------------------------------------------
       EXPANDERS
    -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 9px;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        margin-top: 3.5rem;
        padding-top: 0.9rem;
        border-top: 1px solid #e7e8eb;
        color: #9a9ea5;
        font-size: 0.68rem;
        display: flex;
        justify-content: space-between;
    }

    /* --------------------------------------------------------
       MOBILE
    -------------------------------------------------------- */

    @media (max-width: 700px) {

        .block-container {
            padding-top: 1.2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-title {
            font-size: 2.15rem;
        }

        .hero-description {
            font-size: 0.88rem;
        }

        .section-number {
            margin-top: 1.45rem;
        }

        .footer {
            flex-direction: column;
            gap: 0.3rem;
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


SAFE_GLOBALS = {
    "__builtins__": {},

    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,

    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,

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
    "log": sp.log,

    "sqrt": sp.sqrt,
    "Abs": sp.Abs,

    "pi": sp.pi,
    "E": sp.E,
}


# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    "Quadratic — turning point": "-t^2 + 8t",
    "Cubic — inflection": "t^3 - 6t^2 + 9t",
    "Sine — periodic motion": "3sin(t)",
    "Cosine — periodic motion": "2cos(t)",
    "Exponential growth": "e^(0.35t)",
    "Exponential decay": "e^(-0.35t)",
    "Logarithmic": "ln(t)",
    "Power": "t^2.5",
    "Rational": "1/t",
    "Damped oscillation": "e^(-0.2t)*sin(2t)",
}


DEFAULT_START = {
    "Quadratic — turning point": 0.0,
    "Cubic — inflection": 0.0,
    "Sine — periodic motion": 0.0,
    "Cosine — periodic motion": 0.0,
    "Exponential growth": 0.0,
    "Exponential decay": 0.0,
    "Logarithmic": 0.2,
    "Power": 0.1,
    "Rational": 0.5,
    "Damped oscillation": 0.0,
}


DEFAULT_END = {
    "Quadratic — turning point": 8.0,
    "Cubic — inflection": 6.0,
    "Sine — periodic motion": 12.0,
    "Cosine — periodic motion": 12.0,
    "Exponential growth": 8.0,
    "Exponential decay": 8.0,
    "Logarithmic": 8.0,
    "Power": 6.0,
    "Rational": 8.0,
    "Damped oscillation": 12.0,
}


# ============================================================
# COLORS
# ============================================================

GRID_COLOR = "rgba(128,128,128,0.18)"

POSITION_COLOR = "#2563eb"
VELOCITY_COLOR = "#059669"
ACCELERATION_COLOR = "#7c3aed"

TANGENT_COLOR = "#dc2626"
SECANT_COLOR = "#0891b2"

POINT_COLOR = "#f59e0b"
ZERO_COLOR = "#6b7280"

INCREASING_COLOR = "rgba(5,150,105,0.12)"
DECREASING_COLOR = "rgba(220,38,38,0.10)"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MotionPoint:
    time: float
    position: float
    velocity: float
    acceleration: float


@dataclass
class CriticalPoint:
    time: float
    position: float
    kind: str


# ============================================================
# PARSING
# ============================================================

def parse_function(expression: str) -> sp.Expr:

    expression = expression.strip()

    if not expression:
        raise ValueError(
            "Enter a position function."
        )

    if len(expression) > 180:
        raise ValueError(
            "Keep the function under 180 characters."
        )

    if not re.fullmatch(
        r"[0-9A-Za-z_+\-*/^().,\s]+",
        expression,
    ):
        raise ValueError(
            "The function contains an unsupported character."
        )

    identifiers = re.findall(
        r"[A-Za-z_][A-Za-z_0-9]*",
        expression,
    )

    unknown = sorted(
        {
            name
            for name in identifiers
            if name not in ALLOWED_LOCALS
        }
    )

    if unknown:
        raise ValueError(
            "Unsupported name: "
            + ", ".join(unknown)
        )

    try:

        parsed = parse_expr(
            expression,
            local_dict=ALLOWED_LOCALS,
            global_dict=SAFE_GLOBALS,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

    except Exception as exc:

        raise ValueError(
            "I couldn't interpret that function. "
            "Try examples such as -t^2 + 8t, sin(t), "
            "e^t, or ln(t)."
        ) from exc

    parsed = sp.sympify(parsed)

    if t not in parsed.free_symbols:
        raise ValueError(
            "The position function must depend on t."
        )

    return parsed


# ============================================================
# NUMERICAL EVALUATION
# ============================================================

def lambdify_expression(expr: sp.Expr) -> Callable:

    return sp.lambdify(
        t,
        expr,
        modules=["numpy"],
    )


def evaluate_array(
    expr: sp.Expr,
    values,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    fn = lambdify_expression(expr)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):

        try:
            result = fn(values)
        except Exception:
            return np.full(
                values.shape,
                np.nan,
                dtype=float,
            )

    result = np.asarray(result)

    if result.shape == ():
        result = np.full(
            values.shape,
            float(result),
        )

    if np.iscomplexobj(result):

        result = np.real_if_close(
            result,
            tol=1000,
        )

        if np.iscomplexobj(result):
            return np.full(
                values.shape,
                np.nan,
            )

    try:

        result = result.astype(
            float,
            copy=False,
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.full(
            values.shape,
            np.nan,
        )

    result[
        ~np.isfinite(result)
    ] = np.nan

    return result


def evaluate_at(
    expr: sp.Expr,
    value: float,
) -> Optional[float]:

    try:

        result = sp.N(
            expr.subs(
                t,
                float(value),
            )
        )

        if result.is_real is False:
            return None

        number = float(result)

        if not np.isfinite(number):
            return None

        return number

    except Exception:

        return None


# ============================================================
# DOMAIN MASK
# ============================================================

def valid_segments(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Return contiguous valid regions of a sampled curve.

    This prevents Plotly from drawing lines across
    discontinuities such as t = 0 for 1/t.
    """

    valid = np.isfinite(y)

    if not np.any(valid):
        return []

    segments = []

    start = None

    for i, is_valid in enumerate(valid):

        if is_valid and start is None:
            start = i

        if (
            start is not None
            and (
                not is_valid
                or i == len(valid) - 1
            )
        ):

            end = (
                i
                if is_valid
                else i - 1
            )

            if end - start >= 1:

                segments.append(
                    (
                        x[start:end + 1],
                        y[start:end + 1],
                    )
                )

            start = None

    return segments


# ============================================================
# GRAPH HELPERS
# ============================================================

def base_layout(
    height: int,
    x_title: str,
    y_title: str,
) -> dict:

    return {
        "height": height,

        "margin": {
            "l": 55,
            "r": 25,
            "t": 35,
            "b": 50,
        },

        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",

        "font": {
            "family": "Inter, sans-serif",
            "size": 12,
            "color": "#555a62",
        },

        "xaxis": {
            "title": x_title,
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
            "automargin": True,
        },

        "yaxis": {
            "title": y_title,
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
            "automargin": True,
        },

        "hovermode": "closest",

        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    }


def add_curve_segments(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    color: str,
    width: float = 3,
    hovertemplate: Optional[str] = None,
    showlegend: bool = True,
):

    segments = valid_segments(
        x,
        y,
    )

    for index, (sx, sy) in enumerate(
        segments
    ):

        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                name=(
                    name
                    if index == 0
                    else None
                ),
                showlegend=(
                    showlegend
                    if index == 0
                    else False
                ),
                line=dict(
                    color=color,
                    width=width,
                ),
                hovertemplate=hovertemplate,
            )
        )


def add_time_marker(
    fig: go.Figure,
    time: float,
):

    fig.add_vline(
        x=time,
        line_width=1.3,
        line_dash="dot",
        line_color=POINT_COLOR,
    )


# ============================================================
# CALCULUS ANALYSIS
# ============================================================

def differentiate(
    position_expr: sp.Expr,
):

    velocity_expr = sp.simplify(
        sp.diff(
            position_expr,
            t,
        )
    )

    acceleration_expr = sp.simplify(
        sp.diff(
            velocity_expr,
            t,
        )
    )

    return (
        velocity_expr,
        acceleration_expr,
    )


def classify_motion(
    velocity: float,
    tolerance: float = 1e-7,
) -> str:

    if velocity > tolerance:
        return "Moving forward"

    if velocity < -tolerance:
        return "Moving backward"

    return "Momentarily stopped"


def classify_acceleration(
    velocity: float,
    acceleration: float,
    tolerance: float = 1e-7,
) -> str:

    if (
        abs(velocity) < tolerance
        or abs(acceleration) < tolerance
    ):

        if acceleration > tolerance:
            return "Velocity increasing"

        if acceleration < -tolerance:
            return "Velocity decreasing"

        return "No instantaneous change"

    if velocity * acceleration > 0:
        return "Speed increasing"

    return "Speed decreasing"


# ============================================================
# NUMERICAL ROOT FINDING
# ============================================================

def bisect_root(
    expr: sp.Expr,
    left: float,
    right: float,
    iterations: int = 50,
) -> Optional[float]:

    f_left = evaluate_at(
        expr,
        left,
    )

    f_right = evaluate_at(
        expr,
        right,
    )

    if (
        f_left is None
        or f_right is None
    ):
        return None

    if abs(f_left) < 1e-10:
        return left

    if abs(f_right) < 1e-10:
        return right

    if f_left * f_right > 0:
        return None

    for _ in range(iterations):

        middle = (
            left + right
        ) / 2

        f_middle = evaluate_at(
            expr,
            middle,
        )

        if f_middle is None:
            return None

        if abs(f_middle) < 1e-10:
            return middle

        if f_left * f_middle <= 0:

            right = middle
            f_right = f_middle

        else:

            left = middle
            f_left = f_middle

    return (
        left + right
    ) / 2


def numerical_roots(
    expr: sp.Expr,
    lower: float,
    upper: float,
    samples: int = 3000,
):

    x = np.linspace(
        lower,
        upper,
        samples,
    )

    y = evaluate_array(
        expr,
        x,
    )

    roots = []

    for i in range(
        len(x) - 1
    ):

        y1 = y[i]
        y2 = y[i + 1]

        if (
            not np.isfinite(y1)
            or not np.isfinite(y2)
        ):
            continue

        if abs(y1) < 1e-8:
            roots.append(x[i])

        elif y1 * y2 < 0:

            root = bisect_root(
                expr,
                x[i],
                x[i + 1],
            )

            if root is not None:
                roots.append(root)

    if np.isfinite(y[-1]) and abs(y[-1]) < 1e-8:
        roots.append(x[-1])

    return deduplicate(
        roots,
        tolerance=1e-4,
    )


def symbolic_roots(
    expr: sp.Expr,
    lower: float,
    upper: float,
):

    roots = []

    try:

        solution = sp.solveset(
            sp.Eq(
                expr,
                0,
            ),
            t,
            domain=sp.S.Reals,
        )

        if isinstance(
            solution,
            sp.FiniteSet,
        ):

            for value in solution:

                if value.is_real is False:
                    continue

                try:

                    number = float(value)

                    if (
                        lower
                        <= number
                        <= upper
                    ):

                        roots.append(
                            number
                        )

                except Exception:
                    pass

    except Exception:
        pass

    return roots


def deduplicate(
    values,
    tolerance: float = 1e-4,
):

    values = sorted(
        float(v)
        for v in values
        if np.isfinite(v)
    )

    unique = []

    for value in values:

        if (
            not unique
            or abs(
                value
                - unique[-1]
            )
            > tolerance
        ):

            unique.append(
                value
            )

    return unique


# ============================================================
# CRITICAL POINTS
# ============================================================

def find_critical_points(
    position_expr: sp.Expr,
    velocity_expr: sp.Expr,
    lower: float,
    upper: float,
):

    candidates = []

    candidates.extend(
        symbolic_roots(
            velocity_expr,
            lower,
            upper,
        )
    )

    candidates.extend(
        numerical_roots(
            velocity_expr,
            lower,
            upper,
        )
    )

    candidates = deduplicate(
        candidates,
        tolerance=1e-3,
    )

    results = []

    for point in candidates:

        position = evaluate_at(
            position_expr,
            point,
        )

        velocity = evaluate_at(
            velocity_expr,
            point,
        )

        if (
            position is None
            or velocity is None
        ):
            continue

        epsilon = max(
            1e-4,
            min(
                0.01,
                (upper - lower) / 500,
            ),
        )

        left_velocity = evaluate_at(
            velocity_expr,
            max(
                lower,
                point - epsilon,
            ),
        )

        right_velocity = evaluate_at(
            velocity_expr,
            min(
                upper,
                point + epsilon,
            ),
        )

        if (
            left_velocity is not None
            and right_velocity is not None
        ):

            if (
                left_velocity > 0
                and right_velocity < 0
            ):

                kind = "Local maximum"

            elif (
                left_velocity < 0
                and right_velocity > 0
            ):

                kind = "Local minimum"

            else:

                kind = "Stationary point"

        else:

            kind = "Critical point"

        results.append(
            CriticalPoint(
                time=point,
                position=position,
                kind=kind,
            )
        )

    return results


# ============================================================
# MOTION INTERVALS
# ============================================================

def find_sign_intervals(
    x: np.ndarray,
    y: np.ndarray,
    positive_label: str,
    negative_label: str,
    zero_tolerance: float = 1e-7,
):

    valid = np.isfinite(y)

    signs = np.full(
        len(y),
        np.nan,
    )

    signs[
        valid
        & (y > zero_tolerance)
    ] = 1

    signs[
        valid
        & (y < -zero_tolerance)
    ] = -1

    intervals = []

    start = None
    current_sign = None

    for i, sign in enumerate(signs):

        if not np.isfinite(sign):

            if start is not None:

                intervals.append(
                    (
                        x[start],
                        x[i - 1],
                        current_sign,
                    )
                )

                start = None
                current_sign = None

            continue

        if start is None:

            start = i
            current_sign = sign
            continue

        if sign != current_sign:

            intervals.append(
                (
                    x[start],
                    x[i - 1],
                    current_sign,
                )
            )

            start = i
            current_sign = sign

    if start is not None:

        intervals.append(
            (
                x[start],
                x[-1],
                current_sign,
            )
        )

    return [
        (
            left,
            right,
            (
                positive_label
                if sign == 1
                else negative_label
            ),
        )
        for left, right, sign in intervals
    ]


# ============================================================
# GRAPH DATA
# ============================================================

def make_motion_grid(
    lower: float,
    upper: float,
    samples: int = 1200,
):

    return np.linspace(
        lower,
        upper,
        samples,
    )


def make_motion_point(
    position_expr,
    velocity_expr,
    acceleration_expr,
    time,
):

    return MotionPoint(
        time=time,
        position=evaluate_at(
            position_expr,
            time,
        ),
        velocity=evaluate_at(
            velocity_expr,
            time,
        ),
        acceleration=evaluate_at(
            acceleration_expr,
            time,
        ),
    )


# ============================================================
# SESSION STATE
# ============================================================

if "selected_time" not in st.session_state:
    st.session_state.selected_time = None

if "challenge_index" not in st.session_state:
    st.session_state.challenge_index = 0

if "challenge_answer" not in st.session_state:
    st.session_state.challenge_answer = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="kicker">Calculus laboratory</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="hero-title">Motion Explorer</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-description">
        A visual laboratory for understanding derivatives through motion.
        Build a position function, move through time, and watch position,
        velocity, acceleration, tangent slopes, and turning points respond
        together.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 01 — POSITION FUNCTION
# ============================================================

st.markdown(
    '<div class="section-number">01</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Define the motion</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Choose a model or enter your own position function."
    "</div>",
    unsafe_allow_html=True,
)

function_col1, function_col2 = st.columns(
    [1, 2.3]
)

with function_col1:

    preset = st.selectbox(
        "Example",
        list(PRESETS.keys()),
        index=0,
    )

with function_col2:

    function_input = st.text_input(
        "Position function",
        value=PRESETS[preset],
        placeholder="e.g. -t^2 + 8t",
    )


try:

    position_function = parse_function(
        function_input
    )

    (
        velocity_function,
        acceleration_function,
    ) = differentiate(
        position_function
    )

except ValueError as error:

    st.error(str(error))
    st.stop()


st.markdown(
    '<div class="function-card">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="function-label">'
    "Position"
    "</div>",
    unsafe_allow_html=True,
)

st.latex(
    rf"s(t) = {sp.latex(position_function)}"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 02 — TIME WINDOW
# ============================================================

st.markdown(
    '<div class="section-number">02</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Choose the time window</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "The graphs and analysis below use this interval."
    "</div>",
    unsafe_allow_html=True,
)

time_col1, time_col2 = st.columns(2)

with time_col1:

    start_time = st.number_input(
        "Start",
        value=float(
            DEFAULT_START.get(
                preset,
                0.0,
            )
        ),
        step=0.5,
    )

with time_col2:

    end_time = st.number_input(
        "End",
        value=float(
            DEFAULT_END.get(
                preset,
                8.0,
            )
        ),
        step=0.5,
    )


if end_time <= start_time:

    st.error(
        "The end time must be greater than the start time."
    )

    st.stop()


# ============================================================
# GRAPH DATA
# ============================================================

graph_t = make_motion_grid(
    start_time,
    end_time,
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
# INITIAL TIME
# ============================================================

if (
    st.session_state.selected_time is None
    or not (
        start_time
        <= st.session_state.selected_time
        <= end_time
    )
):

    st.session_state.selected_time = (
        start_time
        + 0.25
        * (
            end_time
            - start_time
        )
    )


time = st.slider(
    "Selected time",
    min_value=float(start_time),
    max_value=float(end_time),
    value=float(
        st.session_state.selected_time
    ),
    step=max(
        (end_time - start_time) / 400,
        0.001,
    ),
)

st.session_state.selected_time = time


# ============================================================
# CURRENT VALUES
# ============================================================

motion_point = make_motion_point(
    position_function,
    velocity_function,
    acceleration_function,
    time,
)


if (
    motion_point.position is None
    or motion_point.velocity is None
    or motion_point.acceleration is None
):

    st.warning(
        f"The selected time t = {time:.3f} "
        "is outside the real-valued domain of the function "
        "or one of its derivatives."
    )

    st.stop()


position = motion_point.position
velocity = motion_point.velocity
acceleration = motion_point.acceleration

motion = classify_motion(
    velocity
)

speed_behavior = classify_acceleration(
    velocity,
    acceleration,
)


# ============================================================
# 03 — CURRENT MOTION
# ============================================================

st.markdown(
    '<div class="section-number">03</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">At this moment</div>',
    unsafe_allow_html=True,
)

current_cols = st.columns(5)

with current_cols[0]:

    st.metric(
        "Position",
        f"{position:.3f} m",
    )

with current_cols[1]:

    st.metric(
        "Velocity",
        f"{velocity:.3f} m/s",
    )

with current_cols[2]:

    st.metric(
        "Acceleration",
        f"{acceleration:.3f} m/s²",
    )

with current_cols[3]:

    st.metric(
        "Speed",
        f"{abs(velocity):.3f} m/s",
    )

with current_cols[4]:

    st.metric(
        "Time",
        f"{time:.3f} s",
    )


insight_cols = st.columns(3)

with insight_cols[0]:

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">Direction</div>
            <div class="insight-value">{motion}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with insight_cols[1]:

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">Speed</div>
            <div class="insight-value">{speed_behavior}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with insight_cols[2]:

    acceleration_direction = (
        "Velocity is increasing"
        if acceleration > 1e-7
        else
        "Velocity is decreasing"
        if acceleration < -1e-7
        else
        "Velocity is momentarily constant"
    )

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">Acceleration</div>
            <div class="insight-value">
                {acceleration_direction}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 04 — POSITION GRAPH
# ============================================================

st.markdown(
    '<div class="section-number">04</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Position → tangent slope"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "The tangent line's slope is the instantaneous velocity."
    " Click the curve to explore another moment."
    "</div>",
    unsafe_allow_html=True,
)


tangent_values = (
    position
    + velocity
    * (
        graph_t
        - time
    )
)


position_fig = go.Figure()

add_curve_segments(
    position_fig,
    graph_t,
    position_values,
    name="Position s(t)",
    color=POSITION_COLOR,
    width=3,
    hovertemplate=(
        "t = %{x:.3f}<br>"
        "s(t) = %{y:.3f}"
        "<extra></extra>"
    ),
)

add_curve_segments(
    position_fig,
    graph_t,
    tangent_values,
    name="Tangent",
    color=TANGENT_COLOR,
    width=2,
    hovertemplate=None,
)

position_fig.add_trace(
    go.Scatter(
        x=[time],
        y=[position],
        mode="markers",
        name="Selected point",
        marker=dict(
            color=POINT_COLOR,
            size=12,
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            "t = %{x:.3f}<br>"
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
        500,
        "Time t",
        "Position s(t)",
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
# CLICK GRAPH TO MOVE TIME
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

            if abs(
                selected_x
                - st.session_state.selected_time
            ) > 0.001:

                st.session_state.selected_time = (
                    selected_x
                )

                st.rerun()

except Exception:
    pass


st.markdown(
    f"""
    <div class="concept-card">
        At <strong>t = {time:.3f}</strong>, the point on the
        position graph is <strong>({time:.3f}, {position:.3f})</strong>.
        The tangent slope is <strong>{velocity:.3f}</strong>,
        which is the instantaneous velocity.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 05 — VELOCITY GRAPH
# ============================================================

st.markdown(
    '<div class="section-number">05</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Velocity = first derivative"
    "</div>",
    unsafe_allow_html=True,
)

st.latex(
    rf"v(t)=s'(t)={sp.latex(velocity_function)}"
)

velocity_fig = go.Figure()

add_curve_segments(
    velocity_fig,
    graph_t,
    velocity_values,
    name="Velocity v(t)",
    color=VELOCITY_COLOR,
    width=3,
    hovertemplate=(
        "t = %{x:.3f}<br>"
        "v(t) = %{y:.3f}"
        "<extra></extra>"
    ),
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

add_time_marker(
    velocity_fig,
    time,
)

velocity_fig.add_hline(
    y=0,
    line_width=1,
    line_color=ZERO_COLOR,
)

velocity_fig.update_layout(
    **base_layout(
        390,
        "Time t",
        "Velocity v(t)",
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
# 06 — ACCELERATION GRAPH
# ============================================================

st.markdown(
    '<div class="section-number">06</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Acceleration = second derivative"
    "</div>",
    unsafe_allow_html=True,
)

st.latex(
    rf"a(t)=s''(t)={sp.latex(acceleration_function)}"
)

acceleration_fig = go.Figure()

add_curve_segments(
    acceleration_fig,
    graph_t,
    acceleration_values,
    name="Acceleration a(t)",
    color=ACCELERATION_COLOR,
    width=3,
    hovertemplate=(
        "t = %{x:.3f}<br>"
        "a(t) = %{y:.3f}"
        "<extra></extra>"
    ),
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

add_time_marker(
    acceleration_fig,
    time,
)

acceleration_fig.add_hline(
    y=0,
    line_width=1,
    line_color=ZERO_COLOR,
)

acceleration_fig.update_layout(
    **base_layout(
        390,
        "Time t",
        "Acceleration a(t)",
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
# 07 — CALCULUS CHAIN
# ============================================================

st.markdown(
    '<div class="section-number">07</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">The derivative chain</div>',
    unsafe_allow_html=True,
)

chain_cols = st.columns(3)

with chain_cols[0]:

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-label">Position</div>
            <div class="insight-value">
                Where the object is.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        rf"s(t)={sp.latex(position_function)}"
    )

with chain_cols[1]:

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-label">First derivative</div>
            <div class="insight-value">
                How position changes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        rf"v(t)=s'(t)"
    )

with chain_cols[2]:

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-label">Second derivative</div>
            <div class="insight-value">
                How velocity changes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        rf"a(t)=s''(t)"
    )


# ============================================================
# 08 — SECANT → TANGENT
# ============================================================

st.markdown(
    '<div class="section-number">08</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Secant → tangent laboratory"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Move h toward zero and watch the secant slope approach "
    "the derivative."
    "</div>",
    unsafe_allow_html=True,
)

h = st.slider(
    "Secant interval h",
    min_value=0.001,
    max_value=2.0,
    value=0.5,
    step=0.001,
)

second_time = min(
    time + h,
    end_time,
)

actual_h = (
    second_time
    - time
)

second_position = evaluate_at(
    position_function,
    second_time,
)

if (
    second_position is not None
    and actual_h > 0
):

    secant_slope = (
        second_position
        - position
    ) / actual_h

else:

    secant_slope = None


secant_fig = go.Figure()

add_curve_segments(
    secant_fig,
    graph_t,
    position_values,
    name="Position",
    color=POSITION_COLOR,
    width=3,
    hovertemplate=None,
)

if secant_slope is not None:

    secant_x = np.array(
        [
            time,
            second_time,
        ]
    )

    secant_y = (
        position
        + secant_slope
        * (
            secant_x
            - time
        )
    )

    secant_fig.add_trace(
        go.Scatter(
            x=secant_x,
            y=secant_y,
            mode="lines",
            name="Secant",
            line=dict(
                color=SECANT_COLOR,
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
                color=SECANT_COLOR,
                size=9,
            ),
        )
    )


secant_tangent = (
    position
    + velocity
    * (
        graph_t
        - time
    )
)

add_curve_segments(
    secant_fig,
    graph_t,
    secant_tangent,
    name="Tangent",
    color=TANGENT_COLOR,
    width=2,
    hovertemplate=None,
)

secant_fig.update_layout(
    **base_layout(
        470,
        "Time t",
        "Position s(t)",
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

    comparison = st.columns(3)

    with comparison[0]:

        st.metric(
            "Secant slope",
            f"{secant_slope:.6f}",
        )

    with comparison[1]:

        st.metric(
            "Tangent slope",
            f"{velocity:.6f}",
        )

    with comparison[2]:

        st.metric(
            "Absolute difference",
            f"{abs(secant_slope - velocity):.6f}",
        )


st.latex(
    r"""
    \frac{s(t+h)-s(t)}{h}
    \longrightarrow
    s'(t)
    \qquad
    \text{as }h\to0
    """
)


# ============================================================
# 09 — MOTION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-number">09</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Motion analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "The sign of velocity determines direction. "
    "Velocity and acceleration together determine whether speed "
    "is increasing or decreasing."
    "</div>",
    unsafe_allow_html=True,
)


motion_intervals = find_sign_intervals(
    graph_t,
    velocity_values,
    "Increasing position",
    "Decreasing position",
)


interval_cols = st.columns(2)

with interval_cols[0]:

    st.markdown("**Position increasing**")

    increasing = [
        interval
        for interval in motion_intervals
        if interval[2]
        == "Increasing position"
    ]

    if increasing:

        for left, right, _ in increasing:

            st.write(
                f"**{left:.3f} → {right:.3f} s**"
            )

    else:

        st.write(
            "No increasing interval detected."
        )

with interval_cols[1]:

    st.markdown("**Position decreasing**")

    decreasing = [
        interval
        for interval in motion_intervals
        if interval[2]
        == "Decreasing position"
    ]

    if decreasing:

        for left, right, _ in decreasing:

            st.write(
                f"**{left:.3f} → {right:.3f} s**"
            )

    else:

        st.write(
            "No decreasing interval detected."
        )


st.markdown(
    """
    <div class="concept-card">
        <strong>Speed is different from velocity.</strong>
        Velocity can be negative because direction matters.
        Speed is always nonnegative and equals
        <strong>|v(t)|</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10 — CRITICAL POINTS
# ============================================================

st.markdown(
    '<div class="section-number">10</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Turning points</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Critical points occur where velocity is zero or undefined "
    "while the original position function is defined."
    "</div>",
    unsafe_allow_html=True,
)


critical_points = find_critical_points(
    position_function,
    velocity_function,
    start_time,
    end_time,
)


if critical_points:

    cp_data = pd.DataFrame(
        [
            {
                "Type": point.kind,
                "Time": f"{point.time:.4f}",
                "Position": f"{point.position:.4f}",
            }
            for point in critical_points
        ]
    )

    st.dataframe(
        cp_data,
        hide_index=True,
        width="stretch",
    )

else:

    st.info(
        "No critical points were detected in this interval."
    )


# ============================================================
# 11 — CRITICAL POINT VISUALIZATION
# ============================================================

if critical_points:

    critical_fig = go.Figure()

    add_curve_segments(
        critical_fig,
        graph_t,
        position_values,
        name="Position",
        color=POSITION_COLOR,
        width=3,
        hovertemplate=(
            "t = %{x:.3f}<br>"
            "s(t) = %{y:.3f}"
            "<extra></extra>"
        ),
    )

    for point in critical_points:

        critical_fig.add_trace(
            go.Scatter(
                x=[point.time],
                y=[point.position],
                mode="markers+text",
                text=[point.kind],
                textposition="top center",
                name=point.kind,
                marker=dict(
                    size=10,
                    color=POINT_COLOR,
                    line=dict(
                        color="white",
                        width=2,
                    ),
                ),
                hovertemplate=(
                    "t = %{x:.4f}<br>"
                    "s(t) = %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    critical_fig.add_hline(
        y=0,
        line_width=1,
        line_color=ZERO_COLOR,
    )

    critical_fig.update_layout(
        **base_layout(
            430,
            "Time t",
            "Position s(t)",
        )
    )

    st.plotly_chart(
        critical_fig,
        width="stretch",
        key="critical_graph",
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )


# ============================================================
# 12 — DERIVATIVE INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-number">11</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Interpret this exact moment"
    "</div>",
    unsafe_allow_html=True,
)


if velocity > 1e-7:

    direction_text = (
        "Position is increasing, so the object is moving forward."
    )

elif velocity < -1e-7:

    direction_text = (
        "Position is decreasing, so the object is moving backward."
    )

else:

    direction_text = (
        "Velocity is approximately zero, so the object is "
        "momentarily stopped."
    )


if (
    velocity * acceleration
    > 1e-7
):

    speed_text = (
        "Velocity and acceleration have the same sign, "
        "so speed is increasing."
    )

elif (
    velocity * acceleration
    < -1e-7
):

    speed_text = (
        "Velocity and acceleration have opposite signs, "
        "so speed is decreasing."
    )

else:

    speed_text = (
        "The instantaneous velocity or acceleration is "
        "approximately zero."
    )


st.markdown(
    f"""
    <div class="concept-card">
        At <strong>t = {time:.3f} s</strong>,
        <strong>s(t) = {position:.3f} m</strong>,
        <strong>v(t) = {velocity:.3f} m/s</strong>, and
        <strong>a(t) = {acceleration:.3f} m/s²</strong>.
        <br><br>
        {direction_text}
        <br>
        {speed_text}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 13 — CHALLENGE MODE
# ============================================================

st.markdown(
    '<div class="section-number">12</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Challenge mode</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Test whether you can interpret the motion before revealing "
    "the calculation."
    "</div>",
    unsafe_allow_html=True,
)


questions = [
    (
        "direction",
        f"At t = {time:.2f}, is the object moving forward or backward?",
    ),
    (
        "speed",
        f"At t = {time:.2f}, is the object's speed increasing or decreasing?",
    ),
    (
        "velocity_zero",
        "What does v(t) = 0 mean physically?",
    ),
]


challenge_index = (
    st.session_state.challenge_index
    % len(questions)
)

question_id, question_text = questions[
    challenge_index
]

st.markdown(
    f"**{question_text}**"
)


if question_id == "direction":

    answer = st.radio(
        "Your answer",
        [
            "Moving forward",
            "Moving backward",
            "Momentarily stopped",
        ],
        key="challenge_direction",
    )

    correct = (
        answer == motion
    )

elif question_id == "speed":

    answer = st.radio(
        "Your answer",
        [
            "Speed increasing",
            "Speed decreasing",
            "Neither",
        ],
        key="challenge_speed",
    )

    if (
        velocity * acceleration
        > 1e-7
    ):

        expected = "Speed increasing"

    elif (
        velocity * acceleration
        < -1e-7
    ):

        expected = "Speed decreasing"

    else:

        expected = "Neither"

    correct = (
        answer == expected
    )

else:

    answer = st.radio(
        "Your answer",
        [
            "The object is momentarily stopped.",
            "The object has zero position.",
            "The object has zero acceleration.",
        ],
        key="challenge_zero",
    )

    correct = (
        answer
        == "The object is momentarily stopped."
    )


challenge_cols = st.columns(2)

with challenge_cols[0]:

    if st.button(
        "Check answer",
        use_container_width=True,
    ):

        if correct:

            st.success(
                "Correct."
            )

        else:

            st.error(
                "Not quite. Use the graphs and derivative definitions above."
            )

with challenge_cols[1]:

    if st.button(
        "Next challenge",
        use_container_width=True,
    ):

        st.session_state.challenge_index += 1
        st.rerun()


# ============================================================
# 14 — NUMERICAL DERIVATIVE EXPERIMENT
# ============================================================

st.markdown(
    '<div class="section-number">13</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    "Numerical derivative experiment"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Estimate the derivative from nearby position values instead "
    "of using symbolic differentiation."
    "</div>",
    unsafe_allow_html=True,
)


delta = st.slider(
    "Δt",
    min_value=0.001,
    max_value=1.0,
    value=0.1,
    step=0.001,
)


left_t = time - delta
right_t = time + delta

left_position = evaluate_at(
    position_function,
    left_t,
)

right_position = evaluate_at(
    position_function,
    right_t,
)


if (
    left_position is not None
    and right_position is not None
):

    numerical_velocity = (
        right_position
        - left_position
    ) / (
        2 * delta
    )

    numerical_error = abs(
        numerical_velocity
        - velocity
    )

    numerical_cols = st.columns(3)

    with numerical_cols[0]:

        st.metric(
            "Numerical estimate",
            f"{numerical_velocity:.6f} m/s",
        )

    with numerical_cols[1]:

        st.metric(
            "Exact derivative",
            f"{velocity:.6f} m/s",
        )

    with numerical_cols[2]:

        st.metric(
            "Absolute error",
            f"{numerical_error:.6f}",
        )

    st.latex(
        r"""
        s'(t)
        \approx
        \frac{s(t+\Delta t)-s(t-\Delta t)}
        {2\Delta t}
        """
    )

else:

    st.warning(
        "A symmetric numerical derivative could not be "
        "evaluated at this time."
    )


# ============================================================
# 15 — DOWNLOAD DATA
# ============================================================

st.markdown(
    '<div class="section-number">14</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Export the experiment</div>',
    unsafe_allow_html=True,
)

export_data = pd.DataFrame(
    {
        "time": graph_t,
        "position": position_values,
        "velocity": velocity_values,
        "acceleration": acceleration_values,
    }
)

csv_data = export_data.to_csv(
    index=False
)

st.download_button(
    "Download motion data (.csv)",
    data=csv_data,
    file_name="motion_explorer_data.csv",
    mime="text/csv",
)


# ============================================================
# 16 — HOW THE MATHEMATICS WORKS
# ============================================================

with st.expander(
    "How the mathematics works"
):

    st.markdown(
        """
        ### 1. Position

        The user defines a position function

        \[
        s(t)
        \]

        which tells us where the object is at time \(t\).

        ### 2. Velocity

        The first derivative gives instantaneous velocity:

        \[
        v(t)=s'(t)
        \]

        Its sign tells us the direction of motion.

        ### 3. Acceleration

        Differentiating again gives acceleration:

        \[
        a(t)=s''(t)
        \]

        This describes how velocity changes.

        ### 4. Tangent line

        At \(t=a\), the tangent line is

        \[
        y=s(a)+s'(a)(t-a)
        \]

        so the tangent's slope is exactly the instantaneous
        velocity.

        ### 5. Secant approximation

        For a small interval \(h\),

        \[
        \frac{s(a+h)-s(a)}{h}
        \]

        estimates the derivative.

        As \(h\to0\),

        \[
        \frac{s(a+h)-s(a)}{h}
        \rightarrow s'(a).
        \]

        ### 6. Critical points

        A turning point can occur where

        \[
        s'(t)=0.
        \]

        A sign change from positive to negative indicates a
        local maximum, while a sign change from negative to
        positive indicates a local minimum.
        """
    )


# ============================================================
# 17 — ABOUT
# ============================================================

with st.expander(
    "About Motion Explorer"
):

    st.write(
        """
        Motion Explorer is an interactive calculus laboratory
        designed around the idea that derivatives should be
        experienced geometrically and physically, not only
        manipulated symbolically.

        The application connects position, velocity,
        acceleration, tangent slopes, secant slopes,
        numerical differentiation, and critical-point analysis
        in one interactive environment.
        """
    )

    st.write(
        "Built with Python, Streamlit, SymPy, NumPy, Plotly, and pandas."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <span>Motion Explorer</span>
        <span>
            Python · Streamlit · SymPy · NumPy · Plotly
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
