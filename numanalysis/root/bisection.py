import numpy as np

class Bisection:

    def __init__(self, polynomial):

        self._polynomial = polynomial

        # validate the polynomial class
        if "numpy.polynomial.polynomial.Polynomial" not in str(type(self._polynomial)):
            raise TypeError("Polynomial must be of type `numpy.polynomial.polynomial.Polynomial`")


    def _fun(self, x: float):
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
        return sum([a * x ** i for i, a in enumerate(self._polynomial.coef)])
    


    def fit(self, lower_bound :float, upper_bound :float,  tolerance = 1e-7, max_iter = 100) -> float:

        """
        Bisection Root Finding Method.
        Finding a solution to f(x) = 0 given an interval [a,b] where, 
        f(lower_bound) and f(upper_bound) have opposite signs.
        
        Parameters:
        -----------
        fun : function
            Polynomial function of the form f(x) = 0
        lower_bound : float
            Interval lower bound.
        lower_bound : float
            Interval upper bound.
        tolerance : float (optional)
            Acceptable error level.
        max_iter : int (optional)
            Maximum number of iterations.

        Returns:
        ---------
        x : float
            Root of the equation f(x) = 0.
    
        """

        if self._fun(lower_bound) * self._fun(upper_bound) < 0: 

            iter = 1

            while iter <= max_iter:
                if self._fun(lower_bound) > self._fun(upper_bound):
                    self._fun = - self._fun

                midpoint = (lower_bound + upper_bound) / 2 

                # check tolerance level

                if abs(self._fun(midpoint)) < tolerance: 
                    return midpoint
                    

                elif self._fun(midpoint) == 0:
                    return midpoint
                    

                elif self._fun(midpoint) > 0:
                    upper_bound = midpoint

                else:
                    lower_bound = midpoint
                
                iter +=1

                if iter == max_iter:
                    raise NotImplementedError(f"Method Failed, max iterations ({max_iter}) reached!")

        else:

            raise ValueError("Please use valid endpoints such that f(a) * f(b) < 0")