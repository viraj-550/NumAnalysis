
import numpy as np
import sympy

class FixedPoint:
        
    def __init__(self, expr, expr_type: str):

        ''' 
        Fixed Point Iteration Method.
        Finding a solution to an equation of the form: p = g(p), 
        given an initial approximation p0. 
        
        Parameters:
        -----------
        expr : np.Polynomial or callable
            Equation that needs to be analyzed
        expr_type: str
            "function" - equation is of type function callable
            "polynomial" - equation is of type polynomial

        '''

        self.expr = expr
        self.expr_type = expr_type

        if self.expr_type == "polynomial":

            if "numpy.polynomial.polynomial.Polynomial" not in str(type(self._polynomial)):
                raise TypeError("Polynomial must be of type `numpy.polynomial.polynomial.Polynomial`")
            
            self.x = sympy.symbols("x")
            self._sympy_poly = sum([a * self.x ** i for i, a in enumerate(self.expr.coef)])
            self._fun = lambda eval: self._sympy_poly.subs(self.x, eval).evalf()
        
        elif self.expr_type == "function":
            self._fun = self.expr
        
        else:
            raise ValueError("Invalid expr_type. Choose from: ['function', 'polynomial']")
    
    def fit(self, p0 :float,  tolerance = 1e-7, max_iter = 100) -> float:

        ''' 
        Parameters:
        -----------
        _fun : polynomial function
            function of the form g(p) = p
        p0 : float
            initial point
        tolerance  : float (optional)
            acceptable error level
        max_iter: int (optional)
            maximum numer of interations

        
        Returns:
        -------
        p : float
            Solution to the equation g(p) = p.

        '''

        iter = 0
        p = self._fun(p0)

        while iter < max_iter:
            if abs(p - p0) < tolerance:
                return p   
            
            p0 = p
            p = self._fun(p0)
            iter += 1

            if iter == max_iter:
                raise NotImplementedError(f"Method failed, max iterations of ({max_iter}) reached!")