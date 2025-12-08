# Mothership-Drone Routing Problem with Obstacles

Generates random MDRP+O instances with a specified number of convex polygon obstacles and targets, which are randomly placed on the obstacles. Solution procedure constructs an initial tour via GTSP, then iteratively refines launch/return locations by constructing and solving a SOCP problem.

This solution procedure is based on:
Poikonen, Stefan. "Hybrid Routing Models Utilizing Trucks or Ships to Launch Drones." PhD dissertation, University of Maryland, 2018. https://doi.org/10.13016/M2V698G3X

The GTSP heuristic solver is based on: Karapetyan & Gutin. "Lin-Kernighan heuristic adaptations for the generalized traveling salesman problem." European Journal of Operational Research, 2011.

## Files

- `config.py` — Problem parameters and run configuration
- `geometry.py` — Geometric primitives
- `instance.py` — Instance generation and I/O
- `initial_solution.py` — GTSP-based initial tour construction
- `visibility.py` — Line-of-sight graph and shortest paths
- `socp_refinement.py` — SOCP-based iterative improvement
- `plotting.py` — Visualization and route export
- `main.py` — Main script
- `gtsp_lk.py` — Lin-Kernighan heuristic for GTSP

## Dependencies

numpy, cvxpy, matplotlib
