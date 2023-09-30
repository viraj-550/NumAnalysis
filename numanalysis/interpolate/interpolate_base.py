from abc import ABC, abstractmethod
import sympy

class ABCInterpolate(ABC):
    """
    An abstract interpolation class. 
    """

    def __init__(self, x_data, y_data):

        self.x_data = x_data
        self.y_data = y_data
        self.x = sympy.symbols("x")
        #self.poly = self._poly()

    @abstractmethod
    def fit(self, eval):
        pass

    @abstractmethod
    def poly(self):
        pass


    

