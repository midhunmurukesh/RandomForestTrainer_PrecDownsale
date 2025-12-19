# FastGLM Downscaler

**A high-performance, parallelized implementation of the Bernoulli-Gamma "Hurdle" model for statistical precipitation downscaling.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-v0.0.3-orange)

## Overview

Current Version: 0.0.3

**FastGLM Downscaler** is a lightweight Python framework designed to downscale coarse-resolution atmospheric variables (e.g., ERA5) to high-resolution precipitation fields.

Unlike deep learning approaches that operate as "black boxes," this project implements a statistically interpretable **Generalized Linear Model (GLM)** approach based on the framework described by *Legasa et al. (2023)*. It explicitly models precipitation as a two-stage process:
1.  **Occurrence:** Will it rain? (Binary classification)
2.  **Intensity:** How much will it rain? (Continuous regression)

The solver is written from scratch using **Iteratively Reweighted Least Squares (IRLS)** and `scipy`, avoiding heavy external dependencies and optimized for massive parallelization across grid pixels.

---

## Methodology

The downscaling is performed independently for every high-resolution pixel $(i, j)$ using a **Hurdle Model** (Bernoulli-Gamma distribution).

### 1. Occurrence Model (Bernoulli)
The probability of precipitation occurrence, $p$, is modeled using Logistic Regression:
$$\ln\left(\frac{p}{1-p}\right) = \mathbf{X}\beta_{occ}$$
where $\mathbf{X}$ represents the predictor variables (upscaled ERA5 fields + topography).

### 2. Intensity Model (Gamma)
Conditioned on a wet day ($Y > \text{threshold}$), the precipitation amount is modeled using a Gamma GLM with a Log link function:
$$\ln(\mu) = \mathbf{X}\beta_{int}$$
where $\mu$ is the expected intensity. The shape parameter $\alpha$ is estimated via the Method of Moments on the residuals.

### 3. Prediction
The final deterministic precipitation prediction is the expected value:
$$E[Y] = p \times \frac{\alpha}{\beta_{gamma}}$$

---

## Key Features

* **Custom IRLS Solver:** Implements a raw `numpy`-based Newton-Raphson solver for GLMs, which is significantly faster than wrapping `statsmodels` or `sklearn` for millions of pixel-wise regressions.
* **Multiprocessing:** Native support for `ProcessPoolExecutor` to parallelize fitting across CPU cores.
* **Bilinear Upscaling:** Integrated `scipy`-based interpolation to align low-res predictors with high-res targets.
* **Topography Aware:** Explicitly handles static auxiliary features like Digital Elevation Models (DEM).

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/yourusername/FastGLMDownscaler.git](https://github.com/yourusername/FastGLMDownscaler.git)
cd FastGLMDownscaler
pip install numpy scipy tqdm