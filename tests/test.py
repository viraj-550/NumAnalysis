import unittest
import numpy as np
import sys
sys.path.append('numanalysis')
import root, interpolate, quad, least_squares

class TestRootFinding(unittest.TestCase):
    def test_bisection(self):
        # Test bisection method with a simple function and known root
        polynomial = np.polynomial.Polynomial([-4, 0, 1])
        result = root.Bisection(polynomial).fit(0, 3)
        self.assertAlmostEqual(result, 2.0, delta = 1e-3)
        

    def test_fixed_point(self):
        # Test cases for the fixed-point iteration method
        func = lambda x: np.sqrt(10/(4 + x))
        result = root.FixedPoint(expr = func, expr_type = "function").fit(1.5)
        self.assertAlmostEqual(result, 1.365230013, delta = 1e-3)

    def test_newton(self):
        # Test cases for Newton's method
        polynomial = np.polynomial.Polynomial([-5, -2, 0, 2])
        result = root.Newton(polynomial).fit(3)
        self.assertAlmostEqual(result, 1.6006, delta = 1e-3)

    def test_secant(self):
        # Test cases for the secant method
        polynomial = np.polynomial.Polynomial([-5, -2, 0, 2])
        result = root.Secant(polynomial).fit(3, 4)
        self.assertAlmostEqual(result, 1.6006, delta = 1e-3)

class TestInterpolation(unittest.TestCase):
    
    def test_lagrange_interpolation(self):
        # Test cases for Lagrange interpolation
        f = lambda x: 8 * x + (2 * x ** 3) + (4 * x ** 4)
        X = np.array([1,2,3,4,5])
        Y = f(X)
        result = interpolate.Lagrange(X, Y).fit(3.5)
        self.assertAlmostEqual(result, f(3.5), delta = 1e-3)
        

    def test_hermite_interpolation(self):
        # Test cases for Hermite interpolation
        f = lambda x: 8 * x + (2 * x ** 3) + (4 * x ** 4)
        f_prime = lambda x: 8 + (6 * x ** 2) + (16 * x ** 3)
        X = np.array([1,2,3,4,5])
        Y = f(X)
        Y_prime = f_prime(X)
        result = interpolate.Hermite(X, Y, Y_prime).fit(3.5)
        self.assertAlmostEqual(result, f(3.5), delta = 1e-3)

    def test_naville_interpolation(self):
        # Test cases for Naville interpolation
        f = lambda x: 8 * x + (2 * x ** 3) + (4 * x ** 4)
        X = np.array([1,2,3,4,5])
        Y = f(X)
        result = interpolate.Naville(X, Y).fit(3.5)
        self.assertAlmostEqual(result, f(3.5), delta = 1e-3)

    def test_newton_divided_difference(self):
        # Test cases for Newton divided difference interpolation
        f = lambda x: 8 * x + (2 * x ** 3) + (4 * x ** 4)
        X = np.array([1,2,3,4,5])
        Y = f(X)
        result = interpolate.NewtonDivDiff(X, Y).fit(4.5)
        self.assertAlmostEqual(result, f(4.5), delta = 1e-3)

class TestQuad(unittest.TestCase):
    def test_simpson_rule(self):
        # Test cases for Simpson's rule
        f = lambda x: 2 * x + 4 * x ** 2
        result = quad.Simpson(f, [0, 5], 100).fit()
        self.assertAlmostEqual(result, 191.6666666667, delta = 1e-2)        
        

    def test_trapezoid_rule(self):
        # Test cases for the trapezoidal rule
        f = lambda x: 2 * x + 4 * x ** 2
        result = quad.Trapezoid(f, [0, 5], 100).fit()
        self.assertAlmostEqual(result, 191.6666666667, delta = 1e-2)  

    def test_gaussian_quadrature(self):
        # Test cases for Gaussian quadrature
        f = lambda x: 2 * x + 4 * x ** 2
        result = quad.Gaussian(f, [0, 5], 2).fit()
        self.assertAlmostEqual(result, 191.6666666667, delta=1e-2)  

class TestLeastSquares(unittest.TestCase):
    def test_ols(self):
        # Test cases for Ordinary Least Squares (OLS)
        n = 100
        X = np.linspace(0, 100, n)
        f = lambda x: 3 * x + 8 + np.random.randn(n)
        result_coef = least_squares.OLS(X, f(X)).fit(formula = 'linear', show_coef = True)[2]
        self.assertAlmostEqual(result_coef[1], 3, delta = 0.1)

    def test_polynomial_least_squares(self):
        # Test cases for Polynomial Least Squares
        n = 100
        X = np.linspace(0, 100, n)
        f = lambda x: 3 * x ** 2 + 2 * x + 8 + np.random.randn(n)
        result_coef = least_squares.PLS(X, f(X)).fit(degree = 2, show_coef = True)[2]
        self.assertAlmostEqual(result_coef[1], 2, delta = 0.1)


if __name__ == '__main__':
    unittest.main()
