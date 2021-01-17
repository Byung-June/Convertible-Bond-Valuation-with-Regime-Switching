from scipy.stats import norm, multivariate_normal
import numpy as np


def _fixed_point(func, guess, n, epsilon=10 ** (-8)):
    _itr = 0
    # print("Guess:", guess)
    if n <=0:
        raise  RuntimeError('_Does not converge fast enough! ')

    _test = func(guess)
    if abs(_test - guess) < epsilon:
        return _test

    while (n > _itr) and (abs(_test - guess) >= epsilon):
        _itr += 1
        _test = func(_test)
        if (abs(_test - guess)) < epsilon:
            return _test
        else:
            guess = _test

    if abs(_test - guess) >= 0.0001:
        raise RuntimeError("_Does not converge fast enough! _Adjust 'n' to a larger value.")

    if guess < 0:
        return _fixed_point(func, guess * 2, epsilon=10 ** (-8), n=n-10)
    return guess


def _normal_cdf(x, dim):
    if dim == 1:
        cdf = norm.cdf(x)
    elif dim == 2:
        temp = -np.sqrt(1 / 2)
        sigma_matrix = np.array([[1, temp], [temp, 1]])
        cdf = multivariate_normal.cdf(x, cov=sigma_matrix)
    elif dim == 3:
        sigma_matrix = np.sqrt([[1/1, 1/2, 1/3], [1/2, 2/2, 2/3], [1/3, 2/3, 3/3]])
        cdf = multivariate_normal.cdf(x, cov=sigma_matrix)
    else:
        raise ValueError('dimension check!')
    return cdf