# Need this temporarily to import CFRL before it is officially published to PyPI
import sys
sys.path.append("E:/learning/university/MiSIL/CFRL Python Package/CFRL")

import pandas as pd
import numpy as np
import torch
from pycfrl.preprocessor.preprocessor import Preprocessor
from pycfrl.agents.agents import FQI
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from pycfrl.evaluation.evaluation import evaluate_reward_through_simulation
from pycfrl.evaluation.evaluation import evaluate_fairness_through_simulation
from examples.baseline_agents import RandomAgent
import time



def run_exp_one(N, T, seed):
    np.random.seed(seed) # ensure reproducibility
    torch.manual_seed(seed) # ensure reproducibility

    start_time = time.process_time()

    class ConcatenatePreprocessor(Preprocessor):
            def __init__(self) -> None:
                pass

            def preprocess(
                    self, 
                    z: list | np.ndarray, 
                    xt: list | np.ndarray
                ) -> tuple[np.ndarray]:
                if xt.ndim == 1:
                    xt = xt[np.newaxis, :]
                    z = z[np.newaxis, :]
                    xt_new = np.concatenate([xt, z], axis=1)
                    return xt_new.flatten()
                elif xt.ndim == 2:
                    xt_new = np.concatenate([xt, z], axis=1)
                    return xt_new
                
            def preprocess_single_step(
                    self, 
                    z: list | np.ndarray, 
                    xt: list | np.ndarray, 
                    xtm1: list | np.ndarray | None = None, 
                    atm1: list | np.ndarray | None = None, 
                    rtm1: list | np.ndarray | None = None, 
                    verbose: bool = False
                ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
                z = np.array(z)
                xt = np.array(xt)
                if verbose:
                    print("Preprocessing a single step...")

                xt_new = self.preprocess(z, xt)
                if rtm1 is None:
                    return xt_new
                else:
                    return xt_new, rtm1
                

            def preprocess_multiple_steps(
                    self, 
                    zs: list | np.ndarray, 
                    xs: list | np.ndarray, 
                    actions: list | np.ndarray, 
                    rewards: list | np.ndarray | None = None, 
                    verbose: bool = False
                ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
                zs = np.array(zs)
                xs = np.array(xs)
                actions = np.array(actions)
                rewards = np.array(rewards)
                if verbose:
                    print("Preprocessing multiple steps...")
            
                # some convenience variables
                N, T, xdim = xs.shape
                
                # define the returned arrays; the arrays will be filled later
                xs_tilde = np.zeros([N, T, xdim + zs.shape[-1]])
                rs_tilde = np.zeros([N, T - 1])

                # preprocess the initial step
                np.random.seed(0)
                xs_tilde[:, 0, :] = self.preprocess_single_step(zs, xs[:, 0, :])

                # preprocess subsequent steps
                if rewards is not None:
                    for t in range (1, T):
                        np.random.seed(t)
                        xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(zs, 
                                                                                        xs[:, t, :], 
                                                                                        xs[:, t-1, :], 
                                                                                        actions[:, t-1], 
                                                                                        rewards[:, t-1]
                                                                                        )
                    return xs_tilde, rs_tilde                
                else:
                    for t in range (1, T):
                        np.random.seed(t)
                        xs_tilde[:, t, :] = self.preprocess_single_step(zs, 
                                                                        xs[:, t, :], 
                                                                        xs[:, t-1, :], 
                                                                        actions[:, t-1]
                                                                        )
                    return xs_tilde
                
    def f_x0(
            zs: list | np.ndarray, 
            ux0: list | np.ndarray, 
            z_coef: int | float = 1
        ) -> np.ndarray:

        zs = np.array(zs)
        ux0 = np.array(ux0)
        gamma0 = np.array([-0.3, 1 * z_coef, 1])
        n = zs.shape[0]
        M = np.concatenate(
            [
                np.ones([n, 1]),
                zs,
                ux0,
            ],
            axis=1,
        )
        x0 = M @ gamma0
        x0 = x0.reshape(-1, 1)
        return x0

    def f_xt(
            zs: list | np.ndarray, 
            xtm1: list | np.ndarray, 
            atm1: list | np.ndarray, 
            uxt: list | np.ndarray, 
            z_coef: int | float = 1
        ) -> np.ndarray:

        zs = np.array(zs)
        xtm1 = np.array(xtm1)
        atm1 = np.array(atm1)
        uxt = np.array(uxt)
        gamma = np.array([-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]) #-0.3
        n = xtm1.shape[0]
        M = np.concatenate(
            [
                np.ones([n, 1]),
                (zs - 0.5),
                xtm1,
                atm1.reshape(-1, 1) - 0.5,
                xtm1 * (atm1.reshape(-1, 1) - 0.5),
                xtm1 * (zs - 0.5),
                (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                uxt,
            ],
            axis=1,
        )
        xt = M @ gamma
        xt = xt.reshape(-1, 1)
        return xt

    def f_rt(
            zs: list | np.ndarray, 
            xt: list | np.ndarray, 
            at: list | np.ndarray, 
            urtm1: list | np.ndarray, 
            z_coef: int | float = 1
        ) -> np.ndarray:

        zs = np.array(zs)
        xt = np.array(xt)
        at = np.array(at)
        urtm1 = np.array(urtm1)
        lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef])
        n = xt.shape[0]
        at = at.reshape(-1, 1)
        M = np.concatenate(
            [np.ones([n, 1]), xt, zs, at, xt * zs, xt * at, zs * at], axis=1
        )
        rt = M @ lmbda
        return rt

    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0, 
                               f_xt=f_xt, 
                               f_rt=f_rt)
    zs_in = np.random.binomial(n=1, p=0.5, size=N).reshape(-1, 1)
    behavior_agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                    zs=zs_in, 
                                                    state_dim=1, 
                                                    T=T, 
                                                    policy=behavior_agent)

    cp = ConcatenatePreprocessor()
    agent = FQI(num_actions=2, model_type='nn', preprocessor=cp, 
                is_loss_monitored=False, is_early_stopping_nn=False, 
                is_q_monitored=False, is_early_stopping_q=False)
    agent.train(zs=zs, 
                xs=states, 
                actions=actions, 
                rewards=rewards, 
                max_iter=100, 
                preprocess=True)

    value = evaluate_reward_through_simulation(env=env, 
                                               z_eval_levels=[[0], [1]], 
                                               state_dim=1, 
                                               N=N, 
                                               T=T, 
                                               policy=agent, 
                                               gamma=0.9)
    cf_metric = evaluate_fairness_through_simulation(env=env, 
                                                     z_eval_levels=[[0], [1]], 
                                                     state_dim=1, 
                                                     N=N, 
                                                     T=T, 
                                                     policy=agent)
    
    end_time = time.process_time()
    df_nt = pd.DataFrame(
                    {'workflow': ['synthetic_data'], 
                     'N': [N], 
                     'T': [T], 
                     'time': [end_time - start_time], 
                     'seed': [seed]
                    }
                )
    return df_nt



def run_exp(Ns, Ts, start_seed, nreps, export=True, 
            export_path='./temporary_outputs.csv'):
    out = pd.DataFrame()
    for n in Ns:
        for t in Ts:
            #times = []
            for i in range(nreps):
                df_nt = run_exp_one(N=n, T=t, seed=start_seed+i)
                #times.append(time)
                print(df_nt)
                out = pd.concat([out, df_nt])

    if export:
        out.to_csv(export_path)
    
    return out



# Run the computing time experiment
df = run_exp(Ns=[100], Ts=[10], start_seed=1, nreps=10, 
             export=True)
print(df)