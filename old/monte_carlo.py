import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import time
import os


class MonteCarlo:

    def __init__(self, lambdas, vstate, current_state, maturity, dt,
                 S_0, vmu, vdividend, vsigma_S, rho,
                 F, process_num,
                 simulation_num=10000):

        self.lambda_h, self.lambda_l = lambdas
        p_hh, p_hl = self.transition(self.lambda_h, self.lambda_l, dt)
        p_ll, p_lh = self.transition(self.lambda_l, self.lambda_h, dt)
        self.p_mat = np.array([[p_hh, p_hl], [p_lh, p_ll]])
        # print(self.p_mat)
        self.vstate = vstate
        self.current_state = current_state
        self.step = int(maturity//dt)
        self.simulation_num = simulation_num
        self.S_0 = S_0
        self.vmu = vmu
        self.vdividend = vdividend
        self.vsigma_S = vsigma_S
        self.rho = rho
        self.F = F
        self.dt = dt
        self.process_num = process_num

    def simul_monte(self):
        simulation_num = self.simulation_num

        samples = np.zeros(simulation_num)
        list_test = range(simulation_num)
        pool = Pool(processes=self.process_num)
        samples = pool.map(self.single, tqdm(list_test))
        pool.close()
        pool.join()
        # mean = np.sum(samples) / simulation_num

        samples1 = [samples[i][0] for i in range(len(samples))]
        samples2 = [samples[i][1] for i in range(len(samples))]
        samples3 = [samples[i][2] for i in range(len(samples))]
        mean = np.sum(samples1) / simulation_num
        mean2 = np.sum(samples2) / simulation_num
        mean3 = np.sum(samples3) / simulation_num
        # print('mean ', mean, mean2, mean3)
        # print('sample3', samples3)
        return mean

    def single(self, i):
        # Markov chain
        chain_hist = self.markov_chain(self.step)
        # print('Markov chain, chain_hist, occupation_time)

        sbm = np.random.normal(scale=np.sqrt(self.dt), size=self.step + 1)
        # print(sbm)

        # N share of stock
        S_T, int_sigma, ito_simga = self.stock(self.S_0, sbm, self.vdividend, self.vsigma_S, chain_hist, self.dt)
        samples = max(0, S_T - self.F)
        sample2 = np.exp(-int_sigma / 2 + ito_simga)
        # sample3 = np.exp(ito_simga)
        sample3 = S_T
        samples = [samples, sample2, sample3]

        return samples

    @staticmethod
    def stock(S_0, bm, vdividend, vsigma_S, chain_hist, dt):
        int_dividend = np.sum([vdividend[i] for i in chain_hist], axis=0) * dt
        sigma = [vsigma_S[i] for i in chain_hist]
        int_sigma = np.sum(np.square(sigma), axis=0) * dt
        ito_simga = np.sum(np.multiply(sigma, bm))
        S_T = S_0 * np.exp(- int_dividend - int_sigma/2 + ito_simga)
        # print('st', S_T)
        return S_T, int_sigma, ito_simga

    @staticmethod
    def transition(lambda_i, lambda_j, dt):
        p_ii = lambda_i/(lambda_i + lambda_j) * np.exp(-(lambda_i+lambda_j) * dt) + lambda_j/(lambda_i + lambda_j)
        p_ij = 1-p_ii
        return p_ii, p_ij

    @staticmethod
    def sim_next_markov(vmultinomial):
        cs = np.cumsum(vmultinomial)
        cs = np.insert(cs, 0, 0, axis=0)
        r = np.random.uniform(0.0, 1.0)
        m = (np.where(cs < r))[0]
        next_state = m[len(m)-1]
        return next_state

    def markov_chain(self, step):
        p_mat = self.p_mat
        vstate = self.vstate
        current_state = self.current_state
        chain_hist = np.array([current_state])
        # occupation_time = np.empty((1, 2), dtype=float)

        for x in range(step):
            current_row = np.ma.masked_values((p_mat[current_state]), 0.0)
            next_state = self.sim_next_markov(current_row)
            # keep track of state history
            chain_hist = np.append(chain_hist, [next_state], axis=0)
            current_state = next_state

        # Cumulative occupation time
        for state in vstate:
            if state == current_state:
                state_num = np.count_nonzero(chain_hist == state) - 1
            else:
                state_num = np.count_nonzero(chain_hist == state)
            state_num = np.array([[state, state_num]])
            # occupation_time = np.append(occupation_time, state_num, axis=0)

        # occupation_time = np.delete(occupation_time, [0,0], axis=0)

        return chain_hist


if __name__ == "__main__":
    # lambdas, vstate, current_state, step, dt
    process_num = 4
    simulation_num = 500000
    S_list = [1.0, 1.1, 1.2]
    for S in S_list:
        S_0 = S
        df_result = np.empty((1, 4))
        print(S_0, simulation_num)

        for current_state in [0, 1]:
            start = time.time()
            monte = MonteCarlo(lambdas=[1, 1], vstate=[0, 1], current_state=current_state, maturity=1, dt=0.01,
                               S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.1, 0.15], rho=0.5, F=1.1,
                               process_num=process_num, simulation_num=simulation_num)
            result1 = monte.simul_monte()
            time1 = time.time() - start
            print(1, "time :", time1, result1)

            start = time.time()
            monte = MonteCarlo(lambdas=[2, 2], vstate=[0, 1], current_state=current_state, maturity=1, dt=0.01,
                               S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.1, 0.15], rho=0.5, F=1.1,
                               process_num=process_num, simulation_num=simulation_num)
            result2 = monte.simul_monte()
            time2 = time.time() - start
            print(2, "time :", time2, result2)

            start = time.time()
            monte = MonteCarlo(lambdas=[1, 1], vstate=[0, 1], current_state=current_state, maturity=1, dt=0.01,
                               S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.15, 0.2], rho=0.5, F=1.1,
                               process_num=process_num, simulation_num=simulation_num)
            result3 = monte.simul_monte()
            time3 = time.time() - start
            print(3, "time :", time3, result3)

            start = time.time()
            monte = MonteCarlo(lambdas=[2, 2], vstate=[0, 1], current_state=current_state, maturity=1, dt=0.01,
                               S_0=S_0, vmu=[0.08, 0.04], vdividend=[0.02, 0.02], vsigma_S=[0.15, 0.2], rho=0.5, F=1.1,
                               process_num=process_num, simulation_num=simulation_num)
            result4 = monte.simul_monte()
            time4 = time.time() - start
            print(4, "time :", time4, result4)

            a = [result1, result2, result3, result4]
            print('average time: ', np.mean([time1, time2, time3, time4]))
            df_result = np.append(df_result, [a], axis=0)
        df_result = np.delete(df_result, [0, 0], axis=0)
        df_result = pd.DataFrame(data=df_result).T
        print(df_result)
        name = 'result_S0_' + str(S_0) + '_' + str(simulation_num) + '2.csv'
        df_result.to_csv(name)