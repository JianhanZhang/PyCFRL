from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from examples.baseline_preprocessors import SequentialPreprocessorOracle, UnawarenessPreprocessor
from examples.baseline_preprocessors import ConcatenatePreprocessor
from pycfrl.agents.agents import FQI
from examples.baseline_agents import RandomAgent, BehaviorAgent
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from pycfrl.evaluation.evaluation import evaluate_reward_through_simulation
from pycfrl.evaluation.evaluation import evaluate_fairness_through_simulation
#from experiments.evaluation_test import evaluate_fairness_through_simulation
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import os, copy
import torch
import multiprocessing as mp
from tqdm import tqdm
from scipy.special import expit
#from experiments.agents_test import FQI

# Behavior policy for generating training trajectories
class BehaviorPolicy:
    def __init__(self, seed=1) -> None:
        np.random.seed(seed)
        self.eta = np.array([-1.39, 0.0, 2.77, 0.0])

    def act(self, z, xt, uat, **kwargs):
        n = xt.shape[0]
        M = np.concatenate(
            [np.ones([n, 1]), xt, z, xt * z],
            axis=1,
        )
        ps = expit(M @ self.eta)
        action_behavior = (uat.flatten() <= ps.flatten()).astype(int)
        action_random = (uat.flatten() <= 0.5).astype(int)

        idx_random = np.random.uniform(0, 1, size=[n]) <= 0.0
        action = action_behavior * (1 - idx_random) + action_random * idx_random
        return action

# Environment that generates the synthetic data
def f_x0(zs, ux0, z_coef=1):
    ux0 = ux0[:, 0].reshape(-1, 1)
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

def f_xt(zs, xtm1, atm1, uxt, z_coef=1):
    uxt = uxt[:, 0].reshape(-1, 1)
    gamma = np.array([-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]) 
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

def f_rt(zs, xt, at, urtm1, z_coef=1):
    lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef, 1])
    n = xt.shape[0]
    at = at.reshape(-1, 1)
    M = np.concatenate(
        [np.ones([n, 1]), xt, zs, at, xt * zs, xt * at, zs * at, urtm1], axis=1
    )
    rt = M @ lmbda
    return rt



# Function to run one experiment replication
def run_exp_one(methods, method_policy, N, T, z_coef, seed, f_x0, f_xt, f_rt):
    #torch.set_num_threads(1)
    #T_eval = 20
    T_eval = 10
    N_eval = 10000
    eval_seed = seed * 10
    env = SyntheticEnvironment(z_coef=z_coef, state_dim=1, f_x0=f_x0, f_xt=f_xt, f_rt=f_rt)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z = np.random.binomial(1, 0.5, size=[N]).reshape(N, -1)
    Z_pre = np.random.binomial(1, 0.5, size=[N]).reshape(N, -1)
    #working_policy = RandomAgent(2)
    working_policy = BehaviorPolicy(seed=seed)

    fqi_model = None
    if method_policy == "FQI_LM":
        learning_algorithm = FQI
        fqi_model = "lm"
        model_type = 'lm'
        max_iters = 100
    elif method_policy == "FQI_NN":
        learning_algorithm = FQI
        fqi_model = "nn"
        model_type = 'nn'
        max_iters = 100
    else:
        raise ValueError("Method policy not found")
    
    (
        zs,
        xs,
        actions,
        rewards,
    ) = sample_trajectory(
        env, Z, 1, T, seed=seed, policy=working_policy 
    )

    (
        zs_pre,
        xs_pre,
        actions_pre,
        rewards_pre,
    ) = sample_trajectory(
        env, Z_pre, 1, T, seed=seed+10, policy=working_policy 
    )

    # set up the methods to be evaluated
    policies = []
    for method in methods:
        if method == "random":
            policies.append(RandomAgent(2))
        elif method == "behavior":
            policies.append(BehaviorAgent(seed=seed))
        elif method == "full":
            preprocessor = ConcatenatePreprocessor(
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]])
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                #state_size=2, 
                num_actions=2,
                hidden_dims=[32, 32], 
                is_early_stopping_nn=False, 
                is_loss_monitored=False, 
                #loss_monitoring_patience=2,
                #loss_monitoring_min_delta=2, 
                #q_monitoring_patience=2, 
                #q_monitoring_min_delta=2
                #test_size_nn=0.3
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)
        elif method == "unaware":
            preprocessor = UnawarenessPreprocessor(
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]])
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                num_actions=2,
                hidden_dims=[32, 32], 
                is_early_stopping_nn=False, 
                is_loss_monitored=False
                #test_size_nn=0.3
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)
        elif method == "ours":
            preprocessor = SequentialPreprocessor(
                z_space=np.array([[0], [1]]), 
                num_actions=2, 
                reg_model=model_type,
                is_normalized=False,
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs_pre),
                                            zs=copy.deepcopy(zs_pre),
                                            actions=copy.deepcopy(actions_pre),
                                            rewards=copy.deepcopy(rewards_pre), 
                                           )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                num_actions=2,
                hidden_dims=[32, 32], 
                is_early_stopping_nn=False, 
                is_loss_monitored=False
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)
        else:
            raise ValueError("Method not found")

    # evaluate reward and fairness
    df_n = pd.DataFrame()
    
    for i, policy in enumerate(policies):
        discounted_cumulative_reward = evaluate_reward_through_simulation(
            env=env, z_eval_levels=np.array([[0], [1]]), state_dim=1, N=N_eval, T=T_eval, 
            policy=policy, seed=eval_seed
        )
        cf_outs = evaluate_fairness_through_simulation(
            env, np.array([[0], [1]]), 1, N_eval, T_eval, policy, seed=eval_seed
        )

        df_n = pd.concat(
            [
                df_n,
                pd.DataFrame(
                    {
                        "N": [N],
                        "T": [T],
                        "cf": [cf_outs],
                        "reward": [discounted_cumulative_reward],
                        "method": [methods[i]],
                        "zcoef": [z_coef],
                        "seed": [seed], 
                    }
                ),
            ]
        )

    return df_n


# Function to run multiple experiment replications at once
def run_exp_N(rep, f_x0, f_xt, f_rt, start_seed=1, export=False):
    NREP = rep
    Ns = [100, 200, 500, 1000, 2000]
    #Ns = [100]
    Ts = [10]
    # z_coefs = [0, 0.5, 1.0, 1.5, 2.0]
    z_coefs = [1]
    method_policy = "FQI_NN"

    methods = [
        "random",
        "full",
        "unaware",
        "ours",
    ]

    mp.set_start_method("spawn")

    df_n = pd.DataFrame()
    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for _ in range(NREP):
                    df_n_one = run_exp_one(methods=methods, 
                          method_policy=method_policy, N=N, T=T, z_coef=z_coef, seed=_+start_seed, 
                          f_x0=f_x0, f_xt=f_xt, f_rt=f_rt)
                    df_n = pd.concat([df_n, df_n_one])
                    print(df_n)
    
    if export:
        df_n.to_csv('./experiments/synthetic_data_results/results_exp_N_test.csv')
    
    return df_n



# Run the experiments
df_n = run_exp_N(rep=50, f_x0=f_x0, f_xt=f_xt, f_rt=f_rt, start_seed=1, export=True)
print(df_n)