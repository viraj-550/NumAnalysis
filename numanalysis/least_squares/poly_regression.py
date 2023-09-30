import numpy as np

class PLS:

    r'''
    Non-linear Polynomial Least Squares approximation to the function:
    Y = f(X) + \epsilon.
    The model takes the form:
    Y = \beta_0 + \beta_1 X + \beta_2 X ** 2 + ... \beta_k X ** k

    Parameters
    -----------

    x_data : array_like[float]
        Values of the independent variable
    y_data : array_like[float]
        Values of the dependent variable
    degree: int
        Polynomial degree

    Returns
    -------
    pred : array_like
        predicted or fitted values \hat{y}
    error : float
        mean squared error
    '''

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def _mse(self, pred: np.ndarray) -> float:
        '''
        Mean Squared Error

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

    def fit(self, degree: int, show_coef = False):

        n = len(self.x_data)
        X = [np.ones(n), self.x_data]

        for i in range(2, degree + 1):
            X_i = np.power(self.x_data, i)
            X = np.append(X, [X_i], axis = 0)
            
        X = np.column_stack(X)
        X_t = np.transpose(X)
        A = np.linalg.inv(X_t @ X)
        B = X_t @ self.y_data

        fit = A @ B 

        pred = fit[0] + fit[1] * self.x_data
        for i in range(2, degree + 1):
            val = fit[i] * np.power(self.x_data, i)
            pred = pred + val

        error = self._mse(pred)

        if show_coef:

            coef = [fit[i] for i in range(degree + 1)]
            return pred, error, coef
        
        else:
            return pred, error