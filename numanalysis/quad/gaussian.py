
import numpy as np
import json
from polynomial import Legendre

class Gaussian:
    """
    Gaussian Quadrature Method:

    Approximate \int_a ^b f(x) \dx using legendre polynomials. 
    
    This method yields greater accuracy than the trapezoid or simpson method as the
    points for evaluation are chosen optimally, rather than equally spaced. 
    For a faster implementation, the legendre polynomial roots and weights can be
    stored in a .json file, and imported directly here using Legendre.roots_weights(degree, save_file = True).

    Parameters
    ----------

    fun : callable
        Function f(x) to be integrated
    bounds : array_like
        Upper and lower bound of the form [a, b]
    degree: int
        Degree of the polynomial to choose the points [x0, x1, .., x_k]

    Returns
    -------
    integral : float
        Approximation of the integral
    """
    

    def __init__(self, fun, bounds, degree):

        self.a = bounds[0]
        self.b = bounds[1]
        self.fun = fun
        self.n = degree
        self._leg_roots_weights = self._get_legendre_roots_weights()
    
    def _get_legendre_roots_weights(self):

        leg_roots_weights = Legendre(self. n).roots_weights(save_file = False)

        return leg_roots_weights
    
    def fit(self):

        quad_sum = 0
        
        for root, weight in self._leg_roots_weights.items():

            root = float(root)
            c_i = float(weight)
            

            # rescale [a, b] into [-1, 1]
            evaulated_fun = float(self.fun(((self.b - self.a) * root + self.a + self.b) / 2) * ((self.b - self.a)) / 2 )

            quad_sum += c_i * evaulated_fun
        
        return float(quad_sum)