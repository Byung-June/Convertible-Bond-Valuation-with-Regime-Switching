from scipy.stats import norm, multivariate_normal
import numpy as np


def _fixed_point(func, guess, epsilon=10 ** (-8), n=10):
    _itr = 0
    # print("Guess:", guess)

    _test = func(guess)
    if abs(_test - guess) < epsilon:
        return _test

    while (n > _itr) and (abs(_test - guess) >= epsilon):
        _itr += 1
        guess = _test
        _test = func(_test)
        # print("Guess:", guess)

        if (abs(_test - guess)) < epsilon:
            return _test
    if abs(_test - guess) >= 0.01:
        raise RuntimeError("_Does not converge fast enough! _Adjust 'n' to a larger value.")
    return None


def _normal_cdf(x, dim):
    if dim == 1:
        cdf = norm.cdf(x)
    elif dim == 2:
        cdf = multivariate_normal.cdf(x, cov=-np.sqrt(1 / 2))
    elif dim == 3:
        sigma_matrix = np.sqrt([[1/1, 1/2, 1/3], [1/2, 2/2, 2/3], [1/3, 2/3, 3/3]])
        cdf = multivariate_normal.cdf(x, cov=sigma_matrix)
    else:
        raise ValueError('dimension check!')
    return cdf