import numpy as np

class Newton:
        
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
    
    def _fun_prime(self, x: float):
        """
        Calcuate f'(x)

        Parameter
        ----------
        x: float
            Evaluating polynomial derivate at x.       
        Returns
        --------
        float: float
            The value of the derivate at x.
        """
        
        return sum([(a * i) * x ** (i - 1) if i > 0 else 0 for i, a in enumerate(self._polynomial.coef)])


    def fit(self, p0 :float,  tolerance = 1e-5, max_iter = 100):
        ''' 
        Newton Root Finding Method.
        Finding a solution to the equation f(x) = 0, 
        given an initial approximation p0. 
        
        Parameters:
        -----------
        fun : polynomial function
            function of the form f(x) = 0
        p0 : float
            initial point
        tolerance : float(optional)
            acceptable error level
        max_iter: int (optional) 
            maximum numer of interations

        Returns
        -------
        x : {float, array_like}
            Root of the equation f(x) = 0
        '''    
        iter = 0

        while iter <= max_iter:

            p = p0 - (self._fun(p0) / self._fun_prime(p0))

            if self._fun(p) == 0 or abs(self._fun(p)) < tolerance:
                return p

            iter += 1 

            # Reparameterize
            p0 = p

            if iter == max_iter:
                raise NotImplementedError(f"Method failed, max iterations of ({max_iter}) reached!")
    

