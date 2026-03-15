# Neural Elastic Dynamics
*Note: This repository was previously titled "Learning Physical Invariants from Unsupervised Video". The current focus is specifically on learning exact physical invariants from state transitions of hybrid dynamical systems.*

## Overview

This research project investigates how neural networks can learn to model perfectly elastic collisions (hard-sphere dynamics) from discrete state transitions. The primary goal is to discover the architectural constraints and inductive biases required for a model to not just fit trajectories, but to truly **discover and preserve physical invariants** (energy, momentum, and phase-space volume).

## The Core Problem: Smooth Approximations vs. Discontinuous Physics

Standard neural networks (like MLPs or basic GNNs) trained with MSE loss struggle to learn hybrid dynamical systems. Hard-sphere collisions involve a smooth continuous flow interrupted by instantaneous jump discontinuities (impulses). 

When trained to predict the next state ($s_{t+1}$) from the current state ($s_t$), networks optimized via MSE tend to under-predict sharp changes representing collisions (a form of regression to the mean for rare events). For physics simulation, this manifests as a **systematic energy bleed**—the network learns a dissipative, damped system rather than a perfectly conservative one.

## Architectural Progression

This codebase systematically explores adding physical inductive biases to neural architectures, proving that equivariance alone is not enough to simulate correct physical dynamics:

1. **MLP Baseline (`mlp_dynamics.py`)**: Predicts next state/residuals directly. Suffers from massive long-term energy drift and fails to generalize to a different number of interacting objects.
2. **EGNN Baseline (`egnn_dynamics.py`)**: Adds $E(n)$ geometric equivariance. Improves trajectory shapes and momentum preservation but still over-smooths interactions and exhibits monotonic energy decay. 
3. **Force-Predicting EGNN (`force_egnn.py`)**: 
   - Predicts **acceleration** instead of state residuals.
   - Restricts forces strictly along the normal vector connecting two balls ($\hat{n}_{ij}$), guaranteeing zero tangential leakage by construction.
   - Employs **Symplectic Integration** (e.g., Leapfrog / Symplectic Euler), which preserves phase-space volume and successfully bounds energy oscillations, converting systematic drift into a bounded oscillation.
4. **Normal Impulse Network (`normal_impulse.py`)**:
   - Isolates the impulse prediction into a scalar function rather than a smooth feature mapping.
   - Factors the problem into **Classification** (Gate: *Are we colliding?*) and **Regression** (Magnitude: *How hard is the collision?*). This decomposition allows the network to handle the sharp step-function geometry of contact separately from the linear scaling of approach speed.
5. **Hybrid ODE Network (`hybrid_ode.py`)**:
   - Explicitly models the system as a true hybrid dynamical system.
   - Separates continuous zero-force flow ($\dot{x}=v, \dot{v}=0$) from an explicit discrete collision jump map.
   - Incorporates exact event-timing (calculating exact time of impact on the continuous timeline) to prevent numerical integration from smoothing over the precise impulse point.

## Key Insights

- **Geometry determines orbit, but invariants constrain the manifold**: An architecture can conserve energy perfectly (e.g., utilizing a Symplectic Integrator) but still predict the wrong trajectory if its geometric application of normal forces is incorrect.
- **Symplectic integrators can mask bad forces**: While a symplectic integrator fixes energy drift, it can implicitly compensate for a network that systematically underestimates force magnitudes.
- **Beware of gate-magnitude co-adaptation**: In factored impulse models, soft classification gates (e.g., predicting $0.5$ chance of collision) can cause the magnitude regressor to artificially double its output to compensate during training. Enforcing strict physical constraints at inference (like a hard `if collision` threshold) requires training the components independently or the system will explode.
- **Non-collision leakage dominates drift**: The primary source of energy drift in sophisticated architectures is often the tiny, non-zero "micro-kicks" applied unconditionally during free-flight (due to gate softness), rather than errors made during the actual collisions themselves.

## Evaluation and Metrics

The project includes custom evaluation pipelines in `physics/metrics.py` and diagnostic scripts (`eval_all.py`, `eval_impulse.py`). Standard MSE on absolute trajectories is fundamentally flawed for chaotic systems over long horizons. Instead, we measure:
- **Kinetic Energy Drift**: Determines if the model suffers from dissipative bias vs. conservative structure.
- **Momentum Preservation**: Evaluates adherence to Newton's Third Law in pairwise interaction forces ($F_{ij} = -F_{ji}$).
- **Impulse Diagnostic (Teacher-Forced)**: Directly evaluates predicted impulse magnitudes and directions against ground-truth collision frames, isolating the force error from the numerical integration error.