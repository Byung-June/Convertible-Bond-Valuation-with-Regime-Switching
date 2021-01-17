import numpy as np
from scipy import integrate, stats
import scipy.special as sc
from utils import _fixed_point, _normal_cdf
import math
from tqdm import tqdm


class Valuation:

    def __init__(self, m=2, maturity=1,
                 Q=np.array([[-0.5, 0.5], [0.6, -0.6]]),
                 d=np.array([0.05, 0.03]),
                 sigma_e=np.array([0.13, 0.26]),
                 c_r=np.array([-0.48, -0.432]),
                 f_bar=np.array([[0.47, 0.423], [0.1, 0.09]]),
                 phi=np.array([0.56, 0.02]),
                 mu_r=np.array([[0.47, 0.423], [0.1, 0.09]]),
                 sigma_r=np.array([[0.02, 0.04], [0.05, 0.1]]),
                 pi=np.array([[-0.03, -0.027], [0, 0]]),
                 c_s=np.array([-0.061, -0.055]),
                 alpha=np.array([0.056, 0.050]),
                 s_bar=np.array([0.079, 0.071]),
                 nu_bar=np.array([0.531 * 10**(-6), 1.062 * 10**(-6)]),
                 gamma=np.array([0.077, 0.069]),
                 zeta=np.array([0.006, 0.012]),
                 rho=0.011,
                 delta=np.array([-0.475, -0.134]),
                 eta=np.array([[9.956, 8.960], [-19.365, -17.429]]),
                 f=np.array([[0.47, 0.423], [0.1, 0.09]]),
                 s_star=np.array([0.079, 0.071]),
                 nu=np.array([0.531 * 10**(-6), 1.062 * 10**(-6)]),

                 S_0 = 1,
                 mu_S = np.array([0.08, 0.04]),
                 F = 1
                 ):

        self.m = m
        self.maturity = maturity
        self.Q = Q
        self.lambda_h = Q[0, 1]
        self.lambda_l = Q[1, 0]
        self.d = d
        self.sigma_e = sigma_e
        self.c_r = c_r
        self.f_bar = f_bar
        self.phi = phi
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.pi = pi
        self.c_s = c_s
        self.alpha = alpha
        self.s_bar = s_bar
        self.nu_bar = nu_bar
        self.gamma = gamma
        self.zeta = zeta
        self.rho = rho
        self.delta = delta
        self.eta = eta

        self.S_0 = S_0
        self.mu_S = mu_S
        self.F = F

        self.f_star = f * delta
        self.s_star = s_star
        self.nu = nu
        self.bond_price = self.bond()
        self.c_bar = 1/maturity * np.log(F/self.bond_price)


    def prob_dist_occupation(self, v_h, state):
        z = 2 * (self.lambda_h * self.lambda_l * v_h * (self.maturity - v_h)) ** (1/2)
        I_1 = sc.iv(1, z)
        I_0 = sc.iv(0, z)
        _a = np.exp(-self.lambda_l * (self.maturity - v_h) - self.lambda_h * v_h)

        if state == 0:
            if v_h == self.maturity:
                f = np.exp(-self.lambda_h * self.maturity)
            elif v_h == 0:
                f = 0
            else:
                _b = (self.lambda_h * self.lambda_l * v_h / (self.maturity - v_h)) ** (1/2)
                f = _a * (_b * I_1 + self.lambda_h * I_0)
        else:
            if v_h == self.maturity:
                f = 0
            elif v_h == 0:
                f = np.exp(-self.lambda_l * self.maturity)
            else:
                _b = (self.lambda_h * self.lambda_l * (self.maturity - v_h) / v_h) ** (1/2)
                f = _a * (_b * I_1 + self.lambda_l * I_0)
        return f

    def _integrand_euro(self, v_h, state, boundary=False):
        v_l = self.maturity - v_h
        d_h, d_l = self.d
        S_T = self.S_0 * np.exp(-(d_h * v_h + d_l * v_l))
        P_T = self.F

        sigma_h, sigma_l = self.sigma_e * np.sqrt((1 - np.exp(-2 * self.sigma_e * self.maturity)) / (2 * self.sigma_e))
        sigma_C = (sigma_h * v_h + sigma_l * v_l)/self.maturity
        mu_C = np.log(S_T/P_T) - (sigma_C ** 2) * self.maturity / 2

        rv = stats.norm(loc=0, scale=1)
        d1 = (mu_C + sigma_C ** 2 * self.maturity) / (sigma_C * (self.maturity ** (1/2)))
        d1 = rv.cdf(d1)
        d2 = mu_C / (sigma_C * (self.maturity ** (1/2)))
        d2 = rv.cdf(d2)
        if boundary:
            return S_T * d1 - P_T * d2
        else:
            return (S_T * d1 - P_T * d2) * self.prob_dist_occupation(v_h, state)

    def european_option(self):
        europe_ex2 = []
        for state in [0, 1]:
            europe_ex = integrate.quad(self._integrand_euro, 0, self.maturity, args=(state, False,))[0]
            # a = self._integrand_euro(0, state=state, boundary=True)
            # a = self._integrand_euro(self.maturity, state=state, boundary=True)
            # a = self.prob_dist_occupation(0, state)
            # a = self.prob_dist_occupation(self.maturity, state)
            temp = europe_ex \
                   + self._integrand_euro(0, state=state, boundary=True) \
                   * self.prob_dist_occupation(0, state) \
                   + self._integrand_euro(self.maturity, state=state, boundary=True) \
                   * self.prob_dist_occupation(self.maturity, state)
            europe_ex2.append(temp)

        return europe_ex2

    def _d_1(self, q, v_h, maturity, state):
        v_l = maturity - v_h
        d_h, d_l = self.d
        c_bar_h, c_bar_l = self.c_bar
        sigma_h, sigma_l = self.sigma_e * np.sqrt((1 - np.exp(-2 * self.sigma_e * self.maturity)) / (2 * self.sigma_e))
        sigma_C = (sigma_h * v_h + sigma_l * v_l) / self.maturity
        temp = (np.log(1/q * self.S_0 / self.bond_price[state]) - (d_h * v_h + d_l * v_l) + (c_bar_h * v_h + c_bar_l * v_l)) \
               + 0.5 * sigma_C ** 2 * maturity
        temp /= sigma_C * np.sqrt(maturity)
        return temp

    def _d_2(self, q, v_h, maturity, state):
        v_l = maturity - v_h
        sigma_h, sigma_l = self.sigma_e * np.sqrt((1 - np.exp(-2 * self.sigma_e * self.maturity)) / (2 * self.sigma_e))
        sigma_C = (sigma_h * v_h + sigma_l * v_l) / self.maturity
        return self._d_1(q, v_h, maturity, state) - sigma_C * np.sqrt(maturity)

    def american_option(self, n):
        result = []
        if n==1:
            result = self.european_option()

        elif n==2:
            def omega_1(v_h, state):
                v_l = self.maturity - v_h
                d_h, d_l = self.d
                c_bar_h, c_bar_l = self.c_bar

                r_critical = (1 - np.exp(- 0.5 * (c_bar_h * v_h + c_bar_l * v_l))
                          * _normal_cdf(self._d_1(1, v_h, self.maturity, state), dim=1)) / \
                         (1 - np.exp(- 0.5 * (d_h * v_h + d_l * v_l))
                          * _normal_cdf(self._d_1(1, v_h, self.maturity, state), dim=1))

                N_1 = _normal_cdf(self._d_1(r_critical, v_h/2, self.maturity/2, state), dim=1)
                N_2 = _normal_cdf([-self._d_1(r_critical, v_h/2, self.maturity/2, state), self._d_1(1, v_h, self.maturity, state)], dim=2)

                return np.exp(-0.5 * (d_h * v_h + d_l * v_l)) * N_1 + np.exp(-(d_h * v_h + d_l * v_l)) * N_2

            def omega_2(v_h, state):
                v_l = self.maturity - v_h
                d_h, d_l = self.d
                c_bar_h, c_bar_l = self.c_bar

                r_critical = (1 - np.exp(- 0.5 * (c_bar_h * v_h + c_bar_l * v_l))
                          * _normal_cdf(self._d_1(1, v_h, self.maturity, state), dim=1) ) / \
                         (1 - np.exp(- 0.5 * (d_h * v_h + d_l * v_l))
                          * _normal_cdf(self._d_1(1, v_h, self.maturity, state), dim=1) )

                N_1 = _normal_cdf(self._d_2(r_critical, v_h/2, self.maturity/2, state), dim=1)
                N_2 = _normal_cdf(
                    [-self._d_2(r_critical, v_h/2, self.maturity/2, state), self._d_2(1, v_h, self.maturity, state)], dim=2
                )

                return np.exp(-0.5 * (c_bar_h * v_h + c_bar_l * v_l)) * N_1 + np.exp(-(c_bar_h * v_h + c_bar_l * v_l)) * N_2

            def _integrand_american_S(v_h, state):
                return omega_1(v_h, state) * self.prob_dist_occupation(v_h, state)

            def _integrand_american_P(v_h, state):
                return omega_2(v_h, state) * self.prob_dist_occupation(v_h, state)

            result = []
            for state in [0, 1]:
                american_ex = self.S_0 * integrate.quad(_integrand_american_S, 0, self.maturity, args=(state,))[0] \
                              + self.bond_price[state] * integrate.quad(_integrand_american_P, 0, self.maturity, args=(state,))[0]
                temp = american_ex \
                       + (_integrand_american_S(0, state) + _integrand_american_P(0, state)) \
                       * self.prob_dist_occupation(0, state) \
                       + (_integrand_american_S(self.maturity, state) + _integrand_american_P(self.maturity, state)) \
                       * self.prob_dist_occupation(self.maturity, state)
                result.append(temp)

        elif n==3:
            def omega_1(v_h, state):
                v_l = self.maturity - v_h
                d_h, d_l = self.d
                c_bar_h, c_bar_l = self.c_bar

                r_critical_13 = _fixed_point(
                    func=lambda x:
                    (1
                     - np.exp(- 1 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                     * _normal_cdf(self._d_2(x, v_h / 3, self.maturity / 3, state), dim=1)
                     - np.exp(- 2 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                     * _normal_cdf(
                                [-self._d_2(x, v_h / 3, self.maturity / 3, state),
                                 self._d_2(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state)],
                                dim=2)
                     ) /
                    (1
                     - np.exp(- 1 / 3 * (d_h * v_h + d_l * v_l))
                     * _normal_cdf(self._d_1(x, v_h / 3, self.maturity / 3, state), dim=1)
                     - np.exp(- 2 / 3 * (d_h * v_h + d_l * v_l))
                     * _normal_cdf(
                                [-self._d_1(x, v_h / 3, self.maturity / 3, state),
                                 self._d_1(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state)],
                                dim=2)
                     ),
                    guess=2, n=100
                )
                r_critical_23 = (1 - np.exp(- 1/3 * (c_bar_h * v_h + c_bar_l * v_l))
                             * _normal_cdf(self._d_1(1, 2/3* v_h, 2/3* self.maturity, state), dim=1)) \
                            / (1 - np.exp(- 1/3 * (c_bar_h * v_h + c_bar_l * v_l))
                               * _normal_cdf(self._d_2(1, 2/3* v_h, 2/3* self.maturity, state), dim=1))

                N_1 = _normal_cdf(self._d_1(r_critical_13, v_h/3, self.maturity/3, state), dim=1)
                if math.isnan(N_1):
                    _normal_cdf(self._d_1(r_critical_13, v_h / 3, self.maturity / 3, state), dim=1)
                N_2 = _normal_cdf(
                    [-self._d_1(r_critical_13, v_h/3, self.maturity/3, state),
                     self._d_1(1, 2/3*v_h, self.maturity * 2/3, state)],
                    dim=2
                )
                N_3 = _normal_cdf(
                    [-self._d_1(r_critical_13, v_h/3, self.maturity/3, state),
                     -self._d_1(r_critical_23, 2/3*v_h, self.maturity * 2/3, state),
                     self._d_1(1, v_h, self.maturity, state)],
                    dim=3
                )

                return (
                        np.exp(-1 / 3 * (d_h * v_h + d_l * v_l)) * N_1
                        + np.exp(-2 / 3 * (d_h * v_h + d_l * v_l)) * N_2
                        + np.exp(-(d_h * v_h + d_l * v_l)) * N_3
                )

            def omega_2(v_h, state):
                v_l = self.maturity - v_h
                d_h, d_l = self.d
                c_bar_h, c_bar_l = self.c_bar

                r_critical_13 = _fixed_point(
                    func=lambda x:
                    (1
                     - np.exp(- 1 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                     * _normal_cdf(self._d_2(x, v_h / 3, self.maturity / 3, state), dim=1)
                     - np.exp(- 2 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                     * _normal_cdf(
                                [-self._d_2(x, v_h / 3, self.maturity / 3, state),
                                 self._d_2(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state)],
                                dim=2)
                     ) /
                    (1
                     - np.exp(- 1 / 3 * (d_h * v_h + d_l * v_l))
                     * _normal_cdf(self._d_1(x, v_h / 3, self.maturity / 3, state), dim=1)
                     - np.exp(- 2 / 3 * (d_h * v_h + d_l * v_l))
                     * _normal_cdf(
                                [-self._d_1(x, v_h / 3, self.maturity / 3, state),
                                 self._d_1(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state)],
                                dim=2)
                     ),
                    guess=2, n=100
                )
                r_critical_23 = (1 - np.exp(- 1 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                             * _normal_cdf(self._d_1(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state), dim=1)) \
                            / (1 - np.exp(- 1 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                               * _normal_cdf(self._d_2(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state), dim=1))

                N_1 = _normal_cdf(self._d_2(r_critical_13, v_h / 3, self.maturity / 3, state), dim=1)
                if math.isnan(N_1):
                    _normal_cdf(self._d_1(r_critical_13, v_h / 3, self.maturity / 3, state), dim=1)
                N_2 = _normal_cdf(
                    [-self._d_2(r_critical_13, v_h / 3, self.maturity / 3, state),
                     self._d_2(1, 2 / 3 * v_h, self.maturity * 2 / 3, state)],
                    dim=2
                )
                N_3 = _normal_cdf(
                    [-self._d_2(r_critical_13, v_h / 3, self.maturity / 3, state),
                     -self._d_2(r_critical_23, 2 / 3 * v_h, self.maturity * 2 / 3, state),
                     self._d_2(1, v_h, self.maturity, state)],
                    dim=3
                )

                return (
                        np.exp(-1 / 3 * (d_h * v_h + d_l * v_l)) * N_1
                        + np.exp(-2 / 3 * (d_h * v_h + d_l * v_l)) * N_2
                        + np.exp(-(d_h * v_h + d_l * v_l)) * N_3
                )

            def _integrand_american_S(v_h, state):
                return omega_1(v_h, state) * self.prob_dist_occupation(v_h, state)

            def _integrand_american_P(v_h, state):
                return omega_2(v_h, state) * self.prob_dist_occupation(v_h, state)

            result = []
            for state in [0, 1]:
                american_ex = self.S_0 * integrate.quad(_integrand_american_S, 0, self.maturity, args=(state,))[0] \
                              + self.bond_price[state] * integrate.quad(_integrand_american_P, 0, self.maturity, args=(state,))[0]
                temp = american_ex \
                       + (_integrand_american_S(0, state) + _integrand_american_P(0, state)) \
                       * self.prob_dist_occupation(0, state) \
                       + (_integrand_american_S(self.maturity, state) + _integrand_american_P(self.maturity, state)) \
                       * self.prob_dist_occupation(self.maturity, state)
                result.append(temp)

        return np.array(result)

    def bond(self, default_free=False):

        if default_free:
            M_c = self._mathcal_M(np.exp(self.c_r), 1)
            temp = self._A(0, self.maturity) + self._A(1, self.maturity) \
                   + self._B(0, self.maturity) * self.f_star[0] + self._B(1, self.maturity) * self.f_star[1]

        else:
            temp = np.multiply(self.delta, self.f_bar)
            M_c = self._mathcal_M(np.exp(self.c_r + self.c_s - temp[0] - temp[1]), 1)
            temp = self._A(0, self.maturity) + self._A(1, self.maturity) \
                   + self._B(0, self.maturity) * self.f_star[0] + self._B(1, self.maturity) * self.f_star[1]
            temp2 = self._F(self.maturity)

            temp += -self.s_star * self._D(self.maturity) \
                    + self.nu * temp2[0] \
                    + temp2[1]

        temp = M_c * np.exp(temp)
        return np.array(temp)

    def _regime_X(self, k):
        temp = np.zeros(self.m)
        temp[k] = 1
        return temp.transpose()

    def _mathcal_M(self, vec, tau):
        if len(vec) != 2:
            raise ValueError('dimension larger then 2 should be implemented later')
        x_h, x_l = vec

        def _integrand(v_h, state):
            return (x_h * v_h + x_l * (tau - v_h)) * self.prob_dist_occupation(v_h, state)

        result = []
        for state in [0, 1]:
            temp = integrate.quad(_integrand, 0, self.maturity, args=(state,))[0]
            temp += x_h * self.prob_dist_occupation(0, state) + x_l * self.prob_dist_occupation(self.maturity, state)
            result.append(temp)

        return np.array(result)

    def _B(self, j, tau):
        return (1 - np.exp(-self.phi[j] * tau)) / self.phi[j]

    def _A(self, j, tau):
        n = 1000
        dt = tau/n
        C1 = np.ones(n+1)
        C2 = np.ones(n+1)
        for i in range(1, n+1):
            temp = - (0.5 * self.sigma_r[j] ** 2 * self._B(j, dt * (i-1)) ** 2 - self.mu_r[j] * self.phi[j])
            C1[i] = C1[i - 1] * (1 + temp[0] * dt) + (self.lambda_l * C2[i - 1] - self.lambda_h * C1[i - 1]) * dt
            C2[i] = C2[i - 1] * (1 + temp[1] * dt) + (self.lambda_l * C2[i - 1] - self.lambda_h * C1[i - 1]) * dt

        return np.log([C1[-1], C2[-1]])

    def _D(self, tau):
        return (1 - np.exp(-self.alpha * tau))/2

    def _F(self, tau):
        theta = (self.gamma + self.zeta * self.eta[1]) / self.alpha + self.rho * self.zeta / self.alpha
        temp = np.array([np.matmul(self.Q, self._regime_X(k)).sum() for k in range(self.m)])
        beta = 0.5 * theta - 0.5\
               * np.sqrt(
                    theta ** 2 - self.zeta ** 2 / self.alpha ** 4
                    + 2 * self.eta[0] * self.zeta ** 2 / self.alpha ** 3
                    - 2 * self.zeta * temp
        )
        beta_j = np.array([beta, theta - beta])

        # j = 1, 2 -> (2 x m) dimensional matrices
        g_j = 2 * beta_j - theta + 1
        kappa = self.zeta / (self.alpha ** 2) * np.sqrt(1-self.rho ** 2)
        h_j = 1/2 - 1.0j/2 * self.rho / np.sqrt(1 - self.rho ** 2) # m dim
        d_j = g_j / 2 - 0.5j * self.rho / np.sqrt(1 - self.rho ** 2) \
              * (self.zeta / (self.alpha ** 2) * (1 - self.alpha * self.eta[0]) - self.rho * (theta - 1))

        alpha_bar = 2 * beta - theta + 1
        beta_bar = self.rho * self.zeta / self.alpha ** 2
        gamma_bar = self.zeta * self.rho * beta / self.alpha ** 2 \
                    - self.zeta ** 2 / (2 * self.alpha ** 4) * (1 - self.alpha * self.eta[0])
        delta_bar = self.zeta ** 2 / (4 * self.alpha ** 4)

        def _Q_part(x, c, n, derivative=False):
            """
            :param x: (m X 1)
            :param c: (m X 1)
            :param n:
            :param derivative:
            :return:
            """
            a_0 = np.ones(self.m)
            a_1 = - (beta_bar * c + gamma_bar) * a_0 / (1+c) / (alpha_bar + c)
            epsilon_bar = beta_bar * c + gamma_bar
            a_2 = ((beta_bar + epsilon_bar) * epsilon_bar - delta_bar * (1 + c) * (alpha_bar + c)) \
                  / ((1+c) * (2+c) * (alpha_bar + c) * (alpha_bar + c + 1)) * a_0
            a_3 = (
                    -epsilon_bar * (beta_bar + epsilon_bar) * (2 * beta_bar + epsilon_bar)
                    - ((2 * beta_bar + epsilon_bar) * (1 + c) * (alpha_bar + c)
                       + epsilon_bar * (2 + c) * (alpha_bar + c + 1))
            ) / (
                    (1 + c) * (2 + c) * (3 + c) * (alpha_bar + c) * (alpha_bar + c + 1) * (alpha_bar + c + 2)
            )

            a_series = [a_0, a_1, a_2, a_3]
            for i in range(4, n + 1):
                temp = (-((i-1) * beta_bar + epsilon_bar) * a_series[i-1] - delta_bar * a_series[i-2]) \
                       / (i+c) / (alpha_bar + c + i - 1)
                a_series.append(temp)
            a_series = np.array(a_series)

            if derivative:
                x_series = (x ** (c - 1)) * np.array([(i + c) * x ** i for i in range(n + 1)])
            else:
                x_series = (x ** c) * np.array([x ** i for i in range(n + 1)])

            temp = a_series * x_series
            temp = np.sum(temp, axis=0)
            return temp

        def _Q(x, n, derivative=False):
            """
            :param x: (m X 1)
            :param n:
            :param derivative:
            :return: (m X 1)
            """
            ones = np.ones(self.m)
            zeros = np.zeros(self.m)
            b = (_Q_part(ones, c=zeros, n=n, derivative=True) + _Q_part(ones, c=zeros, n=n, derivative=False) * beta) \
                / (_Q_part(ones, c=1-alpha_bar, n=n, derivative=False) * _Q_part(ones, c=zeros, n=n, derivative=True)
                   - _Q_part(ones, c=zeros, n=n, derivative=False) * _Q_part(ones, c=1-alpha_bar, n=n, derivative=True))
            a = (1 - _Q_part(ones, c=1-alpha_bar, n=n, derivative=False) * b) / _Q_part(ones, c=0, n=n, derivative=False)

            if derivative:
                return (
                    a * _Q_part(x, c=np.zeros(self.m), n=n, derivative=True)
                    + b * _Q_part(x, c=1 - alpha_bar, n=n, derivative=True)
                )
            else:
                return a * _Q_part(x, c=np.zeros(self.m), n=n) + b * _Q_part(x, c=1-alpha_bar, n=n)

        def _H(tau, n, derivative=False):
            """
            :param tau: maturity
            :param n: series sum terminal
            :param derivative:
            :return: (m X 1)
            """
            if derivative:
                return (
                    -self.alpha * beta * _H(tau, n, derivative=False)
                    - self.alpha * np.exp(-self.alpha * (beta + 1) * tau)
                    * _Q(np.exp(-self.alpha * tau), n=n, derivative=True)
                )
            else:
                return np.exp(-self.alpha * beta * tau) * _Q(np.exp(-self.alpha * tau), n=n)
        n = 100
        temp = _H(tau, n, derivative=True)
        F = - 2 /self.zeta ** 2 * temp / _H(tau, n, derivative=False)
        K = self.s_bar * (self._D(tau) - tau) - 2 * self.gamma * self._D(tau) / self.zeta ** 2 * temp
        return [F, K]


if __name__=="__main__":
    # # delta
    # base_param = np.array([-0.475, -0.134])
    # scale = np.arange(0.3, 1, 0.02)

    # # alpha
    # base_param = np.array([0.056, 0.050])
    # scale = np.arange(1, 1.3, 0.02)

    # # s_bar
    # base_param = np.array([0.079, 0.071])
    # scale = np.arange(0.1, 1.3, 0.1)

    # # zeta
    # base_param = np.array([0.006, 0.012])
    # scale = np.arange(0.9, 1.0, 0.001)

    # nu_bar
    base_param = np.array([0.531 * 10 ** (-6), 1.062 * 10 ** (-6)])
    scale = np.arange(-2., 10., 1.)

    bond_list = []
    european_list = []
    american_list = []
    american_list2 = []
    american_list3 = []


    for multiplier in tqdm(scale):
        # print('multiplier', multiplier)
        scaled = base_param * 10. ** np.array([multiplier, 0])
        # scaled = base_param * np.array([multiplier, 1])
        # scaled = base_param * multiplier
        test_result = Valuation(nu_bar=scaled)

        bond = test_result.bond()
        bond_list.append(bond)
        # print('bond', bond)

        european_option = test_result.european_option()
        european_list.append(european_option)
        # print('europian', european_option)
        american_option1 = test_result.american_option(1)
        american_option2 = test_result.american_option(2)
        american_option3 = test_result.american_option(3)
        american_list2.append(american_option2)
        american_list3.append(american_option3)

        american_option = 1/2 * american_option1 - 4 * american_option2 + 9/2 * american_option3
        american_list.append(american_option)
        # print('american', american_option)

    bond_list = np.array(bond_list)
    european_list = np.array(european_list)
    american_list = np.array(american_list)

    import matplotlib.pyplot as plt
    plt.plot(scale, bond_list[:, 0])
    # plt.plot(scale, np.log(bond_list[:, 1]))
    plt.plot(scale, european_list[:, 0])
    plt.plot(scale, american_list[:, 0])
    plt.legend(['Bond', 'European', 'American'])
    # plt.xlabel('Parameter Multiplier')
    plt.xlabel('Parameter Multiplier (10^Multiplier)')
    plt.ylabel('Value')
    # plt.ylabel('Value (log scale)')
    plt.show()

    plt.plot(scale, bond_list[:, 1])
    # plt.plot(scale, np.log(bond_list[:, 1]))
    plt.plot(scale, european_list[:, 1])
    plt.plot(scale, american_list[:, 1])
    plt.legend(['Bond', 'European', 'American'])

    # plt.xlabel('Parameter Multiplier')
    plt.xlabel('Parameter Multiplier (10^Multiplier)')
    plt.ylabel('Value')
    # plt.ylabel('Value (log scale)')
    plt.show()

    print('bond', bond_list)
    print('european', european_list)
    print('american', american_list)
    print('american2', np.array(american_list2))
    print('american3', np.array(american_list3))
