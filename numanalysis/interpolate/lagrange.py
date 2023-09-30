from numpy.polynomial import Polynomial as P
import numpy as np
import sympy
from interpolate.interpolate_base import ABCInterpolate

class Lagrange(ABCInterpolate):

    """
    
    
    """

    def __init__(self, x_data, y_data):
        super().__init__(x_data, y_data)
        

    def fit(self, eval):

        evaluated_poly = 0

        for i in range(len(self.x_data)):
            product = 1

            for j in range(len(self.x_data)):

                if i != j:
                    if  self.x_data[j] == self.x_data[i]:
                        raise ZeroDivisionError("X array has indistinct points")
                
                    product *= ((eval - self.x_data[j]) / (self.x_data[i] - self.x_data[j]))

                else:
                    continue
            
            evaluated_poly += (product * self.y_data[i]) 

        return evaluated_poly

    def poly(self):
        """
        Symbolic Representation of polynomial
        """
        poly = 0

        for i in range(len(self.x_data)):
            product = 1

            for j in range(len(self.x_data)):

                if i != j:
                    if  self.x_data[j] == self.x_data[i]:
                        raise ZeroDivisionError("X array has indistinct points")
                
                    product *= ((self.x - self.x_data[j]) / (self.x_data[i] - self.x_data[j]))

                else:
                    continue
            
            poly += (product * self.y_data[i])
        
        # Convert Sympy polynomial into Numpy Polynomial

        coefs = np.flip(np.asarray(sympy.Poly(poly).all_coeffs())) # extract coefficients
        return P(coefs)