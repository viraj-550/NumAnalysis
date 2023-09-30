from numpy.polynomial import Polynomial as P
import numpy as np
import sympy
import pandas as pd
from interpolate.interpolate_base import ABCInterpolate

class Naville(ABCInterpolate):

    def __init__(self, x_data, y_data):
        super().__init__(x_data, y_data)
            

    def fit(self, eval):
        """
        Naville's Iterated Interpolation Method. 
        Interpolate tabulated data by evaluating an interpolating polynomial P 
        on n+1 distinct numbers x0,...,xn at point `x`.
        
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

        polynomial : np.Polynomial
            Polynomial.
        """
        data_len = len(self.x_data)
        naville_table = np.empty((data_len,data_len))
        naville_table[:, 0] = self.y_data

        for i in range(1,data_len):

            for j in range(i, data_len):

                # Iteration Table
                naville_table[j,i] = ((eval - self.x_data[j-i]) * naville_table[j,i - 1] - \
                                        (eval - self.x_data[j]) * naville_table[j-1,i-1]) / (self.x_data[j] - self.x_data[j-i])

        return naville_table[data_len-1,data_len-1]

    def poly(self):

        data_len = len(self.x_data)
        naville_pd = pd.DataFrame(np.empty((data_len, data_len)))
        naville_pd.iloc[:, 0] = self.y_data

        for i in range(1,data_len):

            for j in range(i, data_len):

                # Iteration Table
                naville_pd.iloc[j,i] = ((self.x - self.x_data[j-i]) * sympy.sympify(naville_pd.iloc[j,i - 1]) - \
                                        (self.x - self.x_data[j]) * sympy.sympify(naville_pd.iloc[j-1,i-1])) / (self.x_data[j] - self.x_data[j-i])

        poly = naville_pd.iloc[data_len-1,data_len-1]
        coef = np.flip(np.asarray(sympy.Poly(poly).all_coeffs()))

        return P(coef)