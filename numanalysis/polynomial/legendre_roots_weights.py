import json

from numpy.polynomial import Polynomial as P
import numpy as np
import sympy
import math
import os
import sys
sys.path.append('numanalysis')

from root import Newton
from quad import Trapezoid


class Legendre:
    
    def __init__(self, degree: int):
        
        self.x = sympy.symbols("x")
        self.n = degree

        self._leg_polynomial = self._legendre_polynomial(degree)
        self._leg_coefs = self._legendre_coefs()
        self._leg_roots = self._legendre_roots()
        self._quad_coefs = self._quadrature_coefs()
        
    def _legendre_polynomial(self, n):
        
        """
        N order Legendre Polynomial generation via generating function:
        $P_{n + 1}(x) = ((2 * n + 1) x P_n(x) - nP_{n - 1}(x))/ (n + 1)$

        Parameters
        ----------
        n : int
            order of Legendre Polynomial.

        x : sympy.core.symbol.Symbol
            Representative variable 'x'.
        
        Result
        -------
        coef : array_like
            Coefficients of the Legendre Polynomial


        Reference:
        Arfken, George B.; Weber, Hans J. (2005). Mathematical Methods for Physicists. Elsevier Academic Press. ISBN 0-12-059876-0.
        """
        x = np.array(self.x)
        
        if (n == 0):
            return x * 0 + 1
        
        elif (n == 1):
            return x
        else:
            return (((2 * (n - 1) + 1) * x * self._legendre_polynomial(n - 1) \
                     - (n - 1) * self._legendre_polynomial(n - 2)) / (n))
        

    def _legendre_coefs(self):

        # extract coefficients
        coefs = np.flip(np.asarray(sympy.Poly(self._leg_polynomial).all_coeffs())) 
        return P(coefs)
    
    def _legendre_roots(self, tol = 1e-5):

        polyorder = self.n

        if polyorder < 2:
            raise ValueError("Polynomial order should be atleast 2")

        else:
            roots = []
            # The polynomials are alternately even and odd functions. So we evaluate only half the number of roots. 

            for i in range(1, int(polyorder / 2 + 1)):

                p0 = math.cos(math.pi * (i - 0.25) / (polyorder + 0.5))
                
                # Root finding using Newton - Raphson Method
                root = Newton(self._leg_coefs).fit(p0, tolerance = tol, max_iter = 10000)
                
                roots.append(root)
                

        # Use symmetry to get the other roots

            roots = np.array(roots)

            if polyorder % 2 == 0:
                roots = np.concatenate( (1.0 * roots, - roots[::-1]) )

                
            else:
                roots = np.concatenate((1.0 * roots, [0], - roots[::-1]))

            roots.sort()
        return roots

    def _poly_to_fun(self, poly):
            
            """
            Express polynomial as a function
            Parameter
            ----------
            x: float
                The value that the polynomial takes.       
            Returns
            --------
            float: float
                The value of the polynomial at x.

            """
            poly_coef = np.flip(np.asarray(sympy.Poly(poly).all_coeffs()))

            def eval(val):
                return sum([a * val ** i for i, a in enumerate(poly_coef)])
            
            return eval

    def _quadrature_coefs(self):

        quad_coef = []

        for i in range(len(self._leg_roots)):
            prod = 1

            for j in range(len(self._leg_roots)):

                if j != i:
                    p = (self.x - float(self._leg_roots[j])) / (float(self._leg_roots[i]) - float(self._leg_roots[j]))

                    prod *= p

                else:
                    continue
            
            prod_poly_fun = self._poly_to_fun(prod)

            c_i = Trapezoid(prod_poly_fun, [-1, 1], partition = 100).fit()

            quad_coef.append(c_i)
        
        return quad_coef
    
    def roots_weights(self, save_file = True):

        roots_weights_dict = {}

        for i in range(len(self._leg_roots)):
            roots_weights_dict[str(self._leg_roots[i])] = str(self._quad_coefs[i])

        if save_file:
            with open("legendre_roots_weights.json", "w") as lrw:
                json.dump(roots_weights_dict, lrw, indent = 1)
                return None
        else:      
            return roots_weights_dict