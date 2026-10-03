# Calculus Motion Visualizer

An original Python prototype for exploring the connection between derivatives, velocity, and physical motion through interactive simulations.

> **Project history:** This repository is the original prototype that led to [Motion Explorer](https://github.com/SrinidhiKotteswaran/calculus-motion-explorer), the substantially expanded React/TypeScript version.

**[View the current Motion Explorer →](https://calculus-motion-explorer-axfuj2ixr-srinidhi-kotteswaran.vercel.app/)**

## Why I built it

I wanted to explore calculus visually rather than treating derivatives as rules to apply on paper. I started with a small motion experiment and used it to investigate how position, velocity, tangent lines, and instantaneous rates of change relate to one another.

## Features

- Interactive time slider
- Dynamic position-time graphs
- Tangent-line visualization
- Instantaneous velocity calculation
- Physics-based motion examples

## How it works

A position function describes an object's location over time. The slope of the tangent line at a particular time represents the object's instantaneous velocity.

The application lets the user move through time and observe those relationships directly:

**Position → slope → velocity → motion**

## Technologies

- Python
- Streamlit
- SymPy
- NumPy
- Plotly

## Project evolution

This prototype began as an experiment with symbolic and numerical calculus in Python. It helped me identify what I wanted from a larger interactive tool: more control over the visualization, more computational experiments, and a broader set of calculus concepts.

I later rebuilt the project in React and TypeScript as Motion Explorer, adding numerical differentiation, floating-point error analysis, symbolic differentiation through a custom AST-based math engine, critical-point analysis, concavity, numerical integration, and interactive challenges.

Keeping the prototype public makes that progression visible rather than hiding the earlier version.
