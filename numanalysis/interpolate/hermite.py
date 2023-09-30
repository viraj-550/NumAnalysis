from numpy.polynomial import Polynomial as P
import numpy as np
import sympy
from interpolate.interpolate_base import ABCInterpolate

class Hermite(ABCInterpolate):

    def __init__(self, x_data, y_data, y_prime):
        self.y_prime = y_prime
        super().__init__(x_data, y_data)
            

    def fit(self, eval):
        """
        Hermite Interpolation Method. 
        Extention of Naville's method, used to successively generate polynomials.
        The interpolating polynomial P agrees with n+1 distinct points 
        (x0, f(x0)), ...,(xn, f(xn)) as well as its derivate 
        (x0, f'(x0)), ...,(xn, f'(xn)).
        
        Parameters:
        -----------
        x_data : array_like
            numbers x0, x1,..., xn
        y_data : array_like
            numbers f(x0), f(x1),..., f(xn)
        y_prime : array_like
            numbers f'(x0), f'(x1),..., f'(xn)
        val : float 
            point where polynomial P is to approximated

        Returns
        -------
        P(val) : float
            Polynomial P evaluated at `val`
        polynomial : np.Polynomial
            Polynomial.
        """

        n = len(self.x_data)
        z = np.zeros((2 * n))
        Q = np.zeros((2 * n, 2 * n))

        for i in range(n):
            # Divided Difference method
            z[2*i] = self.x_data[i]
            z[2*i + 1] = self.x_data[i]
            Q[2*i, 0] = self.y_data[i]
            Q[2*i + 1, 0] = self.y_data[i]
            Q[2*i + 1, 1] = self.y_prime[i]

            if i != 0:
                Q[2*i, 1] = (Q[2*i, 0] - Q[2*i - 1, 0]) / (z[2*i] - z[2*i-1])

        for i in range(2, 2*n):
            for j in range(2, i+1):
                Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (z[i] - z[i-j])
                
        n = int(len(z)/2)
        hermite_poly = Q[0, 0]

        for i in range(1, 2 * n):
            d = 1
            for j in range(i):
                d = d * (eval - z[j])

            hermite_poly += d * Q[i, i]
        return hermite_poly

    
    def poly(self):

        n = len(self.x_data)
        z = np.zeros((2 * n))
        Q = np.zeros((2 * n, 2 * n))

        for i in range(n):
            # Divided Difference method
            z[2*i] = self.x_data[i]
            z[2*i + 1] = self.x_data[i]
            Q[2*i, 0] = self.y_data[i]
            Q[2*i + 1, 0] = self.y_data[i]
            Q[2*i + 1, 1] = self.y_prime[i]

            if i != 0:
                Q[2*i, 1] = (Q[2*i, 0] - Q[2*i - 1, 0]) / (z[2*i] - z[2*i-1])

        for i in range(2, 2*n):
            for j in range(2, i+1):
                Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (z[i] - z[i-j])
                
        n = int(len(z)/2)
        hermite_poly = Q[0, 0]

        for i in range(1, 2 * n):
            d = 1
            for j in range(i):
                d = d * (self.x - z[j])

            hermite_poly += d * Q[i, i]

        coef = np.flip(np.asarray(sympy.Poly(hermite_poly).all_coeffs()))

        return P(coef)

