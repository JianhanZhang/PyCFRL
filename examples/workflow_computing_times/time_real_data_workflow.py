# Need this temporarily to import CFRL before it is officially published to PyPI
import sys
sys.path.append("E:/learning/university/MiSIL/CFRL Python Package/CFRL")

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pycfrl.reader.reader import read_trajectory_from_dataframe, convert_trajectory_to_dataframe
from pycfrl.reader.reader import export_trajectory_to_csv
from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from pycfrl.agents.agents import FQI
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from pycfrl.environment.environment import SimulatedEnvironment
from pycfrl.evaluation.evaluation import evaluate_reward_through_fqe, evaluate_fairness_through_model
from examples.baseline_agents import RandomAgent
import time



def run_exp_one(N, T, seed):
    np.random.seed(seed) # ensure reproducibility
    torch.manual_seed(seed) # ensure reproducibility

    # Generate the trajectories; this section is not timed
    # define the environment
    def f_x0_uni(
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

    def f_xt_uni(
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

    def f_rt_uni(
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

    # generate sensitive attributes
    zs = np.zeros((N))
    z_levels = np.array([[0], [1]])
    Z_idx = np.random.choice(range(z_levels.shape[0]), size=N, replace=True)
    for i in range(N):
        zs[i] = z_levels[Z_idx[i]].item()
    zs = zs.reshape(-1, 1)

    # generate the IDs
    ids = np.array([i + 1 for i in range(N)]).reshape(-1, 1)

    # generate trajectories with the given sensitive attributes
    env = SyntheticEnvironment(state_dim=1, 
                            z_coef=1, 
                            f_x0=f_x0_uni, 
                            f_xt=f_xt_uni, 
                            f_rt=f_rt_uni)
    zs_in = zs
    agent = RandomAgent(2)

    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=T, 
                                                     policy=agent)

    # export the generated trajectories
    export_trajectory_to_csv(path='./temporary_sample_data.csv', 
                             zs=zs, 
                             states=states, 
                             actions=actions, 
                             rewards=rewards, 
                             ids=ids)

    # Run the preprocessing only workflow; this section is timed
    start_time = time.process_time()

    trajectory = pd.read_csv('./temporary_sample_data.csv')
    zs, states, actions, rewards, ids = read_trajectory_from_dataframe(
                                                    data=trajectory, 
                                                    z_labels=['z1'], 
                                                    state_labels=['state1'], 
                                                    action_label='action', 
                                                    reward_label='reward', 
                                                    id_label='ID', 
                                                    T=T
                                                    )
    zs, states, actions, rewards, ids = read_trajectory_from_dataframe(
                                                    data=trajectory, 
                                                    z_labels=['z1'], 
                                                    state_labels=['state1'], 
                                                    action_label='action', 
                                                    reward_label='reward', 
                                                    id_label='ID', 
                                                    T=T
                                                    )
    
    (
    zs_train, zs_test, 
    states_train, states_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test
    ) = train_test_split(zs, states, actions, rewards, test_size=0.2)

    sp = SequentialPreprocessor(z_space=[[0], [1]], 
                                num_actions=2, 
                                cross_folds=5, 
                                mode='single', 
                                reg_model='nn', 
                                is_loss_monitored=True, 
                                is_early_stopping=True)
    states_tilde, rewards_tilde = sp.train_preprocessor(zs=zs_train, 
                                                        xs=states_train, 
                                                        actions=actions_train, 
                                                        rewards=rewards_train)
    
    agent = FQI(num_actions=2, model_type='nn', preprocessor=sp, 
                is_loss_monitored=True, is_early_stopping_nn=True, 
                is_q_monitored=True, is_early_stopping_q=True)
    agent.train(zs=zs_train, 
                xs=states_tilde, 
                actions=actions_train, 
                rewards=rewards_tilde, 
                max_iter=100, 
                preprocess=False)
    
    env = SimulatedEnvironment(num_actions=2, 
                               state_model_type='nn', 
                               reward_model_type='nn', 
                               is_loss_monitored=True, 
                               is_early_stopping=True)
    env.fit(zs=zs, states=states, actions=actions, rewards=rewards)

    value = evaluate_reward_through_fqe(zs=zs_test, 
                                        states=states_test, 
                                        actions=actions_test, 
                                        rewards=rewards_test, 
                                        policy=agent, 
                                        model_type='nn', 
                                        gamma=0.9, 
                                        is_loss_monitored=True, 
                                        is_early_stopping_nn=True, 
                                        is_q_monitored=True, 
                                        is_early_stopping_q=True)
    cf_metric = evaluate_fairness_through_model(env=env, 
                                                zs=zs_test, 
                                                states=states_test, 
                                                actions=actions_test, 
                                                policy=agent)
    
    end_time = time.process_time()
    df_nt = pd.DataFrame(
                    {'workflow': ['real_data'], 
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
df = run_exp(Ns=[100], Ts=[10], start_seed=1, nreps=1, 
             export=True)
print(df)
