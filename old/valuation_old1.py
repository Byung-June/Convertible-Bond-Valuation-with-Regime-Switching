from scipy import special, stats, integrate
import numpy as np
import pandas as pd
import time


class Analytic:

    def __init__(self, lambdas, current_state, maturity,
                 S_0, vmu, vdividend, vsigma_S, rho,
                 F):

        self.lambda_h, self.lambda_l = lambdas
        self.current_state = current_state
        self.S_0 = S_0
        self.vmu = vmu
        self.vdividend = vdividend
        self.vsigma_S = vsigma_S
        self.rho = rho
        self.F = F
        self.maturity = maturity

    def integrand(self, v_h):
        v_l = self.maturity - v_h
        d_h, d_l = self.vdividend
        S_T = self.S_0 * np.exp(-(d_h * v_h + d_l * v_l))
        P_T = self.F

        sigma_h, sigma_l = self.vsigma_S
        sigma_C = (sigma_h * v_h + sigma_l * v_l)
        mu_C = np.log(S_T/P_T) - (sigma_C ** 2) * self.maturity / 2

        rv = stats.norm(loc=0, scale=1)
        d1 = (mu_C + sigma_C ** 2 * self.maturity) / (sigma_C * (self.maturity ** (1/2)))
        d1 = rv.cdf(d1)
        d2 = mu_C / (sigma_C * (self.maturity ** (1/2)))
        d2 = rv.cdf(d2)
        # print('d1, d2', S_T, d1, d2)
        # integrand = (S_T * d1 - P_T * d2 * self.interest) * self.prob_dist_occupation(v_h)
        integrand = (S_T * d1 - P_T * d2) * self.prob_dist_occupation(v_h)

        return integrand

    def integrand2(self, v_h):
        v_l = self.maturity - v_h
        d_h, d_l = self.vdividend
        S_T = self.S_0 * np.exp(-(d_h * v_h + d_l * v_l))
        P_T = self.F

        sigma_h, sigma_l = self.vsigma_S
        sigma_C = (sigma_h * v_h + sigma_l * v_l)/self.maturity
        mu_C = np.log(S_T/P_T) - sigma_C ** 2 * self.maturity / 2

        rv = stats.norm(loc=0, scale=1)
        d1 = (mu_C + sigma_C ** 2 * self.maturity) / (sigma_C * (self.maturity ** (1/2)))
        d1 = rv.cdf(d1)
        d2 = mu_C / (sigma_C * (self.maturity ** (1/2)))
        d2 = rv.cdf(d2)
        # print('d1, d2', d1, d2)
        return S_T * d1 - P_T * d2

    def prob_dist_occupation(self, v_h):
        z = 2 * (self.lambda_h * self.lambda_l * v_h * (self.maturity - v_h)) ** (1/2)
        I_1 = special.iv(1, z)
        I_0 = special.iv(0, z)
        A = np.exp(-self.lambda_l * (self.maturity - v_h) - self.lambda_h * v_h)

        if self.current_state == 0:
            if v_h == self.maturity:
                f = np.exp(-self.lambda_h * self.maturity)
            elif v_h == 0:
                f = 0
            else:
                B = (self.lambda_h * self.lambda_l * v_h / (self.maturity - v_h)) ** (1/2)
                f = A * (B * I_1 + self.lambda_h * I_0)
        else:
            if v_h == self.maturity:
                f = 0
            elif v_h == 0:
                f = np.exp(-self.lambda_l * self.maturity)
            else:
                B = (self.lambda_h * self.lambda_l * (self.maturity - v_h) / v_h) ** (1/2)
                f = A * (B * I_1 + self.lambda_l * I_0)
        return f

    def europe(self):
        europe_ex = integrate.quad(self.integrand, 0, self.maturity)
        europe_ex2 = europe_ex[0] + self.integrand2(0) * self.prob_dist_occupation(0) \
                     + self.integrand2(self.maturity) * self.prob_dist_occupation(self.maturity)

        # check_prob = integrate.quad(self.prob_dist_occupation, 0, self.maturity)
        # # print('check', check_prob)
        # # print( self.prob_dist_occupation(0),  self.prob_dist_occupation(self.maturity))
        # check2 = check_prob[0] + self.prob_dist_occupation(0) + self.prob_dist_occupation(self.maturity)
        # print('check_prob', check_prob, check2)

        return europe_ex2


if __name__ == "__main__":
    S_list = [1.0, 1.1, 1.2]
    for S in S_list:
        S_0 = S
        df_result = np.empty((1, 4))
        print('analytic', S_0)

        for current_state in [0, 1]:
            start = time.time()
            analytic = Analytic(lambdas=[1, 1], current_state=current_state, maturity=1,
                                S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.1, 0.15], rho=0.5, F=1.1)
            result1 = analytic.europe()
            time1 = time.time() - start
            print(1, "time :", time1, result1)

            start = time.time()
            analytic = Analytic(lambdas=[2, 2], current_state=current_state, maturity=1,
                                S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.1, 0.15], rho=0.5, F=1.1)
            result2 = analytic.europe()
            time2 = time.time() - start
            print(2, "time :", time2, result2)

            start = time.time()
            analytic = Analytic(lambdas=[1, 1], current_state=current_state, maturity=1,
                                S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.15, 0.2], rho=0.5, F=1.1)
            result3 = analytic.europe()
            time3 = time.time() - start
            print(3, "time :", time3, result3)

            start = time.time()
            analytic = Analytic(lambdas=[2, 2], current_state=current_state, maturity=1,
                                S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.15, 0.2], rho=0.5, F=1.1)
            result4 = analytic.europe()
            time4 = time.time() - start
            print(4, "time :", time4, result4)

            a = [result1, result2, result3, result4]
            print('average time: ', np.mean([time1, time2, time3, time4]))
            df_result = np.append(df_result, [a], axis=0)
        df_result = np.delete(df_result, [0, 0], axis=0)
        df_result = pd.DataFrame(data=df_result).T
        print('result', df_result)
