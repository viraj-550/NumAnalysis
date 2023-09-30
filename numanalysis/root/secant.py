
class Secant:
        
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
    

    def fit(self, p0 :float,  p1 :float, tolerance = 1e-5, max_iter = 100) -> float:

        '''
        Secant Method Approximation.
        Finding a solution to the equation f(x) = 0, 
        given an initial approximations p0 and p1. 
        
        Parameters:
        -----------
        fun : function 
            Polynomial function of the form f(x) = 0
        p0 : float
            First approximation
        p1 : float
            Second approximation
        tolerance: float (optional)
            Acceptable error level
        max_iter: int (optional)
            Maximum numer of interations

        Returns:
        --------
        x : float
            Solution to equation f(x) = 0
        '''

        fun_p0 = self._fun(p0)
        iter = 2
        while iter <= max_iter:

            fun_p1 = self._fun(p1)

            p = p1 - (fun_p1 * (p1 - p0)) / (fun_p1 - fun_p0)

            if self._fun(p) == 0 or abs(p - p0) < tolerance:
                return p
            
            iter += 1

            # Reparameterize
            p0 = p1
            fun_p0 = fun_p1
            p1 = p

            if iter == max_iter:
                raise NotImplementedError(f"Method Failed, max iterations of ({max_iter}) reached!")