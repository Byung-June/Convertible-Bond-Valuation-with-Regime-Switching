import numpy as np
from scipy import integrate
import scipy.special as sc


class ClosedSolution:

    def __init__(self, m=2,
                 Q=np.array([[-1, 1], [0.5, -0.5]]),
                 d=np.array([0.05, 0.03]),
                 sigma_e=np.array([0.13, 0.26]),
                 c_r=np.array([-0.48, -0.432]),
                 f_bar=np.array([[0.47, 0.423], [0.1, 0.09]]),
                 phi=np.array([[0.56, 0.504], [0.02, 0.018]]),
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
                 f=np.array([]),
                 s_star=np.array([]),
                 nu=np.array([])
                 ):

        self.m = m
        self.Q = Q
        self.lambda_H = Q[0, 1]
        self.lambda_L = Q[1, 0]
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


    def bond(self):
        M_c_b = self.mathcal_M(self.c_r, 1)
        temp = np.multiply(self.delta, self.f_bar)
        M_c_p = self.mathcal_M(self.c_r + self.c_s - temp[0] - temp[1], 1)


    def regime_X(self, k):
        temp = np.zeros(self.m)
        temp[k] = 1
        return temp.transpose()

    def mathcal_M(self, vec, tau):
        return [np.matmul((np.diag(vec) - self.Q * 1.0j) * tau, self.regime_X(k)).sum() for k in range(self.m)]


    def B(self, j, tau):
        return (1 - np.exp(self.phi[j] * tau)) / self.phi[j]

    def A(self, j, tau):
        def a_j(u, k):
            a_bar_j = [self.mu_r[j][i] * (1 + self.delta[j]) * self.B(j, u)
                       - 1 / 2 * ((1 + self.delta[j]) * self.sigma_r[j][i] * self.B(j, u)) ** 2 for i in range(self.m)]
            a_j = np.diag(a_bar_j) - self.Q.transpose
            return a_j[k]
        temp = [integrate.quad(func=a_j, a=0, b=tau, args=(k, )) for k in range(self.m)]
        return np.log(temp)

    def D(self, tau):
        return (1 - np.exp(-self.alpha * tau))/2

    def F(self, tau, k):
        theta = (self.gamma + self.zeta * self.eta[1]) / self.alpha + self.rho * self.zeta / self.alpha
        temp = [np.matmul(self.Q, self.regime_X(k)).sum() for k in range(self.m)]
        beta = 0.5 * theta - 0.5 \
               * np.sqrt(
            theta ** 2 - self.zeta ** 2 / self.alpha ** 4
            + 2 * self.eta[0] * self.zeta ** 2 / self.alpha ** 3
            - 2 * self.zeta * temp)
        beta = np.array([beta, theta - beta])

        # j = 1, 2 -> (2 x m) dimensional matrices
        g_j = 2 * beta - theta + 1
        d_j = g_j/2 - 0.5j * self.rho/np.sqrt(1 - self.rho ** 2) \
              * (self.zeta/(self.alpha**2) * (1 - self.alpha * self.eta[0]) - self.rho * (theta-1))
        h_j = self.zeta/(self.alpha**2) * np.sqrt(1 - self.rho ** 2) # m dim

        M_1_j_k = 1 + sc.hyp1f1(d_j, g_j, 1.0j * h_j * np.exp(-self.alpha * tau))
        M_2_j_k = 1 + sc.hyp1f1(d_j + 1, g_j + 1, 1.0j * h_j * np.exp(-self.alpha * tau))

        c_j =

        temp = -np.sqrt(1-self.rho **2+self.rho) / self.zeta / self.alpha * np.exp(self.alpha * tau) \
               + 2 * self.alpha / self.zeta ** 2 * \
               (c_j * np.exp(beta * self.alpha * tau) * (beta[0] * M_1_j_k[0] + 1.0j * h_j[0] * np.exp(-self.alpha * tau) * d_j[0]/g_j[0] * M_2_j_k[0])
               )



if __name__=="__main__":



