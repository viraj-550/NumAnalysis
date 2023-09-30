from numpy.polynomial import Polynomial as P
import numpy as np
import sympy
from interpolate.interpolate_base import ABCInterpolate

class NewtonDivDiff(ABCInterpolate):

    def __init__(self, x_data, y_data):
        super().__init__(x_data, y_data)
            
    def fit(self, eval):
        """
        Newton's Divided Difference Method. 
        Extension of Naville's method but this is used to successively generate polynomials.
        
        Parameters:
        -----------
        x_data : array_like
            numbers x0, x1,..., xn
        y_data : array_like
            numbers f(x0), f(x1), ..., f(xn)
        val : float 
            point where polynomial P is to approximated

        Returns
        -------
        P(val): float
            Polynomial evaluated at `val`
        polynomial : np.Polynomial
            Polynomial.
        """

        data_len = len(self.x_data)
        coef = np.zeros(data_len)
        F = np.zeros([data_len, data_len])

        # First column of matrix F is f(x0),f(x1),...,f(xn)
        F[:, 0] = self.y_data 

        for j in range(1, data_len):

            for i in range(j, data_len):

                F[i, j] = (F[i, j-1] - F[i-1, j-1]) / (self.x_data[i] - self.x_data[i-j])

        # Diagonals of matrix F
        for i in range(0, data_len):
            coef[i] = F[i,i]

        n = len(coef) - 1
        p = coef[n]

        for k in range(1, n+1):
            p = coef[n - k] + ((eval - self.x_data[n - k]) * p)
            
        return p

    def poly(self):

        data_len = len(self.x_data)
        coef = np.zeros(data_len)
        F = np.zeros([data_len, data_len])

        # First column of matrix F is f(x0),f(x1),...,f(xn)
        F[:, 0] = self.y_data 

        for j in range(1, data_len):

            for i in range(j, data_len):

                F[i, j] = (F[i, j-1] - F[i-1, j-1]) / (self.x_data[i] - self.x_data[i-j])

        # Diagonals of matrix F
        for i in range(0, data_len):
            coef[i] = F[i,i]

        n = len(coef) - 1
        p = coef[n]

        for k in range(1, n+1):
            p = coef[n - k] + ((self.x - self.x_data[n - k]) * p)
            
        coef = np.flip(np.asarray(sympy.Poly(p).all_coeffs()))
        return P(coef)