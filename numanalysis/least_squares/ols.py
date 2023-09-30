import numpy as np

class OLS:

    r'''
    Ordinary Least Squares approximation to the function:
    Y = f(X) + \epsilon.
    Formulae:
    Linear : Y = \beta_0 + \beta_1 X_1 + ... \beta_k X_k
    Exp    : Y = \beta_0 * exp(\beta_1 * x)
    Power  : Y = \beta_0 * x ** \beta_1

    Parameters
    -----------

    x_data : array_like[float]
        Values of the independent variable
    y_data : array_like[float]
        Values of the dependent variable
    formula: str['linear', 'exp', 'power']
        Type of formula to model the relationship between X and Y

    Returns
    -------
    pred : array_like
        predicted or fitted values \hat{y}
    error : float
        mean squared error
    '''
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data
        self.y_data = y_data

    def _mse(self, pred: np.ndarray) -> float:
        '''
        Calcuate mean squared error.

        Parameters
        ----------

        y : array_like
            y values
        n : int
            number of observations
        pred: array_like
            predicted values

        Returns
        --------
        mse : float
            Mean Squared Error $\sum_i^n (y - \hat{y})^2 / n $
        '''
        return np.square(np.subtract(self.y_data, pred)).mean()
    
  
    def fit(self, formula: str, show_coef = False):

        n = len(self.x_data)

        # Column stack 1's and x values in a N x 2 matrix
        X = np.c_[np.ones(n), self.x_data]
        X_t = np.transpose(X)
        A = np.linalg.inv(X_t @ X)
        B = X_t @ self.y_data

        fit = A @ B

        if formula == 'linear': # y = b + a * x
            b = fit[0]
            a = fit[1]
            pred = a + b * self.x_data
            
        elif formula == 'exp': # y = b * exp(a * x)
            b = np.exp(fit[0])
            a = fit[1]
            pred = b * np.exp(a * self.x_data)

        elif formula == 'power': # y = b * x ** a
            b = np.exp(fit[0])
            a = fit[1]
            pred = b * np.power(np.exp(self.x_data), a)
        
        else:
            raise NotImplementedError("Invalid Regression Formula! Please choose from linear, exp, or power")

        error = self._mse(pred)

        if show_coef:

            coef = [b, a]
            return pred, error, coef
        
        else:
            return pred, error
        