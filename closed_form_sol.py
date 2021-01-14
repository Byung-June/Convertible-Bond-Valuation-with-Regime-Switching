import numpy as np
from scipy import integrate, stats
import scipy.special as sc
from utils import _fixed_point, _normal_cdf
from scipy.optimize import root, fsolve
from mpmath import hyp1f1
from mystic.solvers import BuckshotSolver


class Valuation:

    def __init__(self, m=2, maturity=1,
                 Q=np.array([[-1, 1], [0.5, -0.5]]),
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
            S_T * d1 - P_T * d2
        else:
            return (S_T * d1 - P_T * d2) * self.prob_dist_occupation(v_h, state)

    def european_option(self):
        europe_ex2 = []
        for state in [0, 1]:
            europe_ex = integrate.quad(self._integrand_euro, 0, self.maturity, args=(state, False,))[0]
            temp = europe_ex \
                   + self._integrand_euro(0, state=None, boundary=True) \
                   * self.prob_dist_occupation(0, state) \
                   + self._integrand_euro(self.maturity, state=None, boundary=True) \
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
                temp = american_ex[0] \
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
                    guess=1
                )
                r_critical_23 = (1 - np.exp(- 1/3 * (c_bar_h * v_h + c_bar_l * v_l))
                             * _normal_cdf(self._d_1(1, 2/3* v_h, 2/3* self.maturity, state), dim=1)) \
                            / (1 - np.exp(- 1/3 * (c_bar_h * v_h + c_bar_l * v_l))
                               * _normal_cdf(self._d_2(1, 2/3* v_h, 2/3* self.maturity, state), dim=1))

                N_1 = _normal_cdf(self._d_1(r_critical_13, v_h/3, self.maturity/3, state), dim=1)
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

            def omega_2(v_h):
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
                    guess=1
                )
                r_critical_23 = (1 - np.exp(- 1 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                             * _normal_cdf(self._d_1(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state), dim=1)) \
                            / (1 - np.exp(- 1 / 3 * (c_bar_h * v_h + c_bar_l * v_l))
                               * _normal_cdf(self._d_2(1, 2 / 3 * v_h, 2 / 3 * self.maturity, state), dim=1))

                N_1 = _normal_cdf(self._d_2(r_critical_13, v_h / 3, self.maturity / 3, state), dim=1)
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

        return result

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
            temp += -self.s_star * self._D(self.maturity) \
                    + self.nu * self._F(self.maturity) \
                    + self._K(self.maturity)

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
        return (1 - np.exp(self.phi[j] * tau)) / self.phi[j]

    def _A(self, j, tau):
        def _a_j(u, k):
            a_bar_j = [self.mu_r[j][i] * (1 + self.delta[j]) * self._B(j, u)
                       - 1 / 2 * ((1 + self.delta[j]) * self.sigma_r[j][i] * self._B(j, u)) ** 2 for i in range(self.m)]
            _a_j = np.diag(a_bar_j) - self.Q.transpose()
            _a_j = np.matmul(_a_j, np.array([1, 1]))
            return _a_j[k]
        temp = [integrate.quad(func=_a_j, a=0, b=tau, args=(k, ))[0] for k in range(self.m)]
        return np.log(temp)

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
        beta = np.array([beta, theta - beta])

        # j = 1, 2 -> (2 x m) dimensional matrices
        g_j = 2 * beta - theta + 1
        kappa = self.zeta / (self.alpha ** 2) * np.sqrt(1-self.rho ** 2)
        h_j = 1/2 - 1.0j/2 * self.rho / np.sqrt(1 - self.rho ** 2) # m dim
        d_j = g_j / 2 - 0.5j * self.rho / np.sqrt(1 - self.rho ** 2) \
              * (self.zeta / (self.alpha ** 2) * (1 - self.alpha * self.eta[0]) - self.rho * (theta - 1))

        M_1_j_k = []
        M_2_j_k = []
        for j in [0, 1]:
            M_1_j_k.append(
                [complex(1 + hyp1f1(d_j[j][state], g_j[j][state], 1.0j * kappa[state] * np.exp(-self.alpha[state] * tau)))
                 for state in [0, 1]]
            )
            M_2_j_k.append(
                [complex(
                    1 + hyp1f1(d_j[j][state]+1, g_j[j][state]+1, 1.0j * kappa[state] * np.exp(-self.alpha[state] * tau)))
                 for state in [0, 1]]
            )

        def _find_c(_variables, k, show_constants=False):
            r1, c1 = _variables
            M_1_j_k = []
            M_2_j_k = []
            for j in [0, 1]:
                M_1_j_k.append(
                    [complex(1 + hyp1f1(d_j[j][state], g_j[j][state], 1.0j * kappa[state]))
                     for state in [0, 1]]
                )
                M_2_j_k.append(
                    [complex(1 + hyp1f1(d_j[j][state] + 1, g_j[j][state] + 1, 1.0j * kappa[state]))
                        for state in [0, 1]]
                )

            temp_a = - j * kappa[k] * h_j
            temp_c1 = (beta[0][k] * M_1_j_k[0][k] + 1j * kappa[k] * d_j[0][k] / g_j[0][k] * M_2_j_k[0][k])
            temp_c2 = (beta[1][k] * M_1_j_k[1][k] + 1j * kappa[k] * d_j[1][k] / g_j[1][k] * M_2_j_k[1][k])

            variable2 = - (temp_a * M_1_j_k[1][k] + temp_c2) / (temp_a * M_1_j_k[0][k] + temp_c1) * complex(r1, c1)
            r2, c2 = variable2.real, variable2.imag
            #
            # zero1 = temp_a \
            #         + (complex(r1, c1) * temp_c1 + complex(r2, c2) * temp_c2) \
            #         / (complex(r1, c1) *  M_1_j_k[0][k] + complex(r2, c2) *  M_1_j_k[1][k])

            temp_c12 = beta[0][k] * M_1_j_k[0][k] \
                       + 1j * kappa[k] * d_j[0][k] / g_j[0][k] * M_2_j_k[0][k] * np.exp(-self.alpha[k] * tau)
            temp_c22 = beta[1][k] * M_1_j_k[1][k] \
                       + 1j * kappa[k] * d_j[1][k] / g_j[1][k] * M_2_j_k[1][k] * np.exp(-self.alpha[k] * tau)

            zero2 = temp_a * np.exp(self.alpha[k] * tau) \
                    + (complex(r1, c1) * np.exp(-beta[0][k] * self.alpha[k] * tau) * temp_c12
                                + complex(r2, c2) * np.exp(-beta[1][k] * self.alpha[k] * tau) * temp_c22) \
                    / (complex(r1, c1) *  M_1_j_k[0][k] * np.exp(-beta[0][k] * self.alpha[k] * tau)
                       + complex(r2, c2) *  M_1_j_k[1][k] * np.exp(-beta[1][k] * self.alpha[k] * tau))
            if show_constants:
                return r1, c1, r2, c2
            else:
                return zero2.imag

        _find_c([1, 1], 1)
        wrapper = lambda variables: np.sum(np.array(_find_c(variables, 1)) ** 2)
        variables = np.array([root(_find_c, x0=np.array([1, 1, 1, 1]), args=(state,), maxfev=10000) for state in [0, 1]])

        c_1 = np.array([complex(variables[0, 0], variables[0, 1]), complex(variables[1, 0], variables[1, 1])])
        c_2 = np.array([complex(variables[0, 2], variables[0, 3]), complex(variables[1, 2], variables[1, 3])])

        temp1 = c_1 * np.exp(-beta[0] * self.alpha * tau) \
                * (beta[0] * M_1_j_k[0] + 1j * kappa * np.exp(-self.alpha * tau) * d_j[0]/g_j[0] * M_2_j_k[0]) \
                + c_2 * np.exp(-beta[1] * self.alpha * tau) \
                * (beta[1] * M_1_j_k[1] + 1j * kappa * np.exp(-self.alpha * tau) * d_j[1]/g_j[1] * M_2_j_k[1])
        temp2 = c_1 * np.exp(beta[0] * self.alpha * tau) * M_1_j_k[0] \
                + c_2 * np.exp(beta[1] * self.alpha * tau) * M_1_j_k[1]

        temp = - 2j * self.alpha * kappa * h_j / (self.zeta ** 2) * np.exp(self.alpha * tau) \
               + 2 * self.alpha / (self.zeta ** 2) * temp1 / temp2

        return temp

    def _K(self, tau):
        def _integrand(t, state):
            _temp = self.alpha[state] * self.s_bar[state] * self._D(t)[state] \
                   + self.gamma[state] * self.nu_bar[state] * self._F(t)[state]
            return _temp
        temp = np.array([integrate.quad(_integrand, 0, tau, args=(state,)) for state in [0, 1]])
        return temp

if __name__=="__main__":
    test_result = Valuation()
    test_result.bond()
