---
bibliography: reference.bib
---

# Numerical Analysis Library

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This repository contains a collection of numerical methods implemented in python. These numerical techniques were originally developed as part of the math class _MATH 4610: Numerical Analysis_ (at Georgia State University), and are now packaged here for easy access and use. The library covers various mathematical topics, including root finding, interpolation, quadrature, orthogonal polynomials, and least squares. 

## Features

- **Root Finding**: Algorithms for finding roots of equations, including the Newton-Raphson method, Bisection method and others.
- **Interpolation**: Implementation of standard interpolation methods like Lagrange, Naville, and Hermite interpolation.
- **Quadrature**: Tools for numerical integration using techniques such as the Trapezoidal rule, Simpson's rule and Gaussian quadrature.
- **Least Squares Fit**: Includes ordinary least squares (OLS) technique for approximating functions of linear, exponential or power form, and polynomial least squares (PLS) approximation.


## Getting Started

To use the numerical methods library in your Python projects, follow these steps:

1. Clone this repository to your local machine.

```bash
git clone https://github.com/viraj-550/NumAnalysis.git
```

Install the required dependencies specified in the project's requirements file.

```python
pip install -r requirements.txt
```

Import the relevant modules and functions from the library into your Python scripts.

```python
from numanalysis import root, interpolate, quad, polynomial, least_squares
```

## Usage Examples

Here are some usage examples demonstrating how to use the library's functions:

```python
import numpy as np
from numanalysis import least_squares, quad

# Ordinary Least Squares (OLS) implementation
n = 100
X = np.linspace(0, 100, n)
f = lambda x: 3 * x + 8 + np.random.randn(n)

# Create OLS object
ols = least_squares.OLS(x_data = X, y_data = f(X))

result = ols.fit(formula = 'linear', show_coef = True)

# Gaussian Quadrature 

g = lambda x: 2 * x + 4 * x ** 2

# Create gaussian quadrature object
gauss = quad.Gaussian(g, bounds = [0, 5], degree = 2)

integral = gauss.fit()

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.


## Contact

For inquiries or support, please contact Viraj Chordiya (virajchordiya550@gmail.com).

## Reference

Burden, R. L., Faires, J. D., & Burden, A. M. (2016). _Numerical analysis_ (Tenth edition). Cengage Learning.
