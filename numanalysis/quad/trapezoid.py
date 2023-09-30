
import numpy as np

class Trapezoid:
    """
    Composite Trapezoid Method:
    Approximate \int_a ^b f(x) \dx

    Parameters
    ----------

    fun : callable
        Function f(x) to be integrated
    bounds : array_like
        Upper and lower bound of the form [a, b]
    partition: int
        Number of subintervals fo composite quadrature

    Returns
    -------
    integral : float
        Approximation of the integral
    """

    def __init__(self, fun: callable, bounds: np.ndarray, partition: int):

        self.fun = fun
        self.n = partition      
        self.a = bounds[0]
        self.b = bounds[1]

        self.h = (self.b - self.a) / self.n

    def fit(self):

        sum_0 = self.fun(self.a) + self.fun(self.b)
        sum_1 = 0
        iter = 1

        while iter < self.n:

            val = self.a + (iter * self.h)
            sum_1 += self.fun(val)

            iter += 1

        integral = (self.h / 2) * (sum_0 + 2 * sum_1) 

        return integral
    


        





