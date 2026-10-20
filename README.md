# Calculus Motion Visualizer

The **original Python prototype of Motion Explorer**: an interactive Streamlit application for exploring the relationship between position, derivatives, tangent lines, velocity, and acceleration.

This repository preserves the earlier implementation. The expanded React/TypeScript version lives in a separate repository.

- **Current application:** [Motion Explorer live demo](https://calculus-motion-explorer.vercel.app/)
- **Current source:** [calculus-motion-explorer](https://github.com/SrinidhiKotteswaran/calculus-motion-explorer)

## What the prototype explores

- Enter a position function and inspect its motion over time
- View position, velocity, and acceleration graphs
- Move a time slider to inspect the selected point
- Compare the secant slope with the tangent slope
- Explore critical points and intervals of increasing or decreasing position
- Connect the derivative's geometric meaning to instantaneous velocity

The prototype uses SymPy for symbolic differentiation and NumPy/Plotly for numerical evaluation and visualization. Unlike the newer application, it does **not** implement the custom TypeScript symbolic math engine.

## Technology

- Python
- Streamlit
- SymPy
- NumPy
- Plotly

## Run locally

Requires Python and pip.

```bash
git clone https://github.com/SrinidhiKotteswaran/calculus-motion-visualizer.git
cd calculus-motion-visualizer
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL after the app starts.

## Why I rebuilt it

The first version helped me explore the educational idea: make calculus visible by connecting equations, slopes, graphs, and physical motion. I later wanted more control over the interaction model and wanted to understand the expression-processing machinery rather than rely on a symbolic math library for the core workflow.

That led to the React/TypeScript rebuild and its custom tokenizer, parser, abstract syntax tree (AST), symbolic differentiator, and simplifier. The expanded version also adds numerical methods, calculus analysis, integration, and generated challenges.

The build history is documented in the [current project's build journal](https://github.com/SrinidhiKotteswaran/calculus-motion-explorer/blob/main/docs/BUILD-JOURNAL.md).
