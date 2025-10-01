import sys
sys.path.append("E:/learning/university/MiSIL/CFRL Python Package/CFRL")

from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from .baseline_preprocessors import SequentialPreprocessorOracle, UnawarenessPreprocessor
from .baseline_preprocessors import ConcatenatePreprocessor
from pycfrl.agents.agents import FQI
from .baseline_agents import RandomAgent, BehaviorAgent
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from pycfrl.evaluation.evaluation import evaluate_reward_through_simulation
from pycfrl.evaluation.evaluation import evaluate_fairness_through_simulation
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import os, copy
import torch
import multiprocessing as mp
from tqdm import tqdm

def run_exp1_one(methods, method_policy, N, T, z_coef, seed):
    torch.set_num_threads(1)
    T_eval = 20
    N_eval = 10000
    eval_seed = seed * 10
    IS_CF_LOGITS = False
    env = SyntheticEnvironment(z_coef=z_coef, state_dim=1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z = np.random.binomial(1, 0.5, size=[N]).reshape(N, -1)
    working_policy = BehaviorAgent(seed=seed)
    #working_policy = RandomAgent(2)

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
        max_iters = 100 # ORIGINAL: 100
    else:
        raise ValueError("Method policy not found")
    '''elif method_policy == "DDQN":
        learning_algorithm = OfflineDoubleDQNWrapped
        max_iters = 5000
    elif method_policy == "CQL":
        learning_algorithm = OfflineCQLWrapped
        max_iters = 5000
    else:
        raise ValueError("Method policy not found")'''
    (
        zs,
        xs,
        actions,
        rewards,
    ) = sample_trajectory(
        env, Z, 1, T, seed=seed, policy=working_policy 
    ) # STATE_DIM CHANGED

    '''env = SimpleEnv(env_name='lm4', z_coef=z_coef, seed=seed)
    (
        zs,
        xs,
        actions,
        rewards,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = env.sample_trajectory(
        N, T, Z, seed=seed, policy=working_policy, include_counter=True
    )'''

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
                num_actions=2,
                #name="Full",
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                #seed=seed,
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
                #name="Unaware",
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                #seed=seed,
            )
            policies.append(agent)
        elif method == "ours":
            preprocessor = SequentialPreprocessor(
                z_space=np.array([[0], [1]]), 
                num_actions=2, 
                reg_model=model_type,
                is_normalized=False,
                is_loss_monitored=True,
                is_early_stopping=False,
                loss_monitoring_min_delta=0.01,
                loss_monitoring_patience=3 
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs),
                                            zs=copy.deepcopy(zs),
                                            actions=copy.deepcopy(actions),
                                            rewards=copy.deepcopy(rewards), 
                                           )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                num_actions=2,
                is_loss_monitored=True,
                is_early_stopping_nn=True,
                loss_monitoring_patience=3,
                is_q_monitored=True,
                is_early_stopping_q=False,
                q_monitoring_patience=3,
                #min_delta_q=0.05
                #name="ours",
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                #seed=seed,
            )
            policies.append(agent)
        elif method == "ours_cf2":
            preprocessor = SequentialPreprocessor(
                z_space=np.array([[0], [1]]), 
                num_actions=2, 
                reg_model=model_type,
                is_normalized=False,
                cross_folds=2,
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs),
                                            zs=copy.deepcopy(zs),
                                            actions=copy.deepcopy(actions),
                                            rewards=copy.deepcopy(rewards),
                                           )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                num_actions=2,
                #name="ours_cf2",
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                #seed=seed,
            )
            policies.append(agent)
        elif method == "oracle":
            preprocessor = SequentialPreprocessorOracle(
                env=env,
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]])
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs),
                                            zs=copy.deepcopy(zs),
                                            actions=copy.deepcopy(actions),
                                            rewards=copy.deepcopy(rewards),
                                           )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                num_actions=2,
                #name="Oracle_reward",
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                #seed=seed,
            )
            policies.append(agent)
        elif method == "sdp_nostate_yesreward":
            '''preprocessor = SDP_nostate_yesreward(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                reg_model="nn",
                is_normalized=False,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                model_type=fqi_model,
                action_size=2,
                name="sdp_nostate_yesreward",
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)'''
        elif method == "sdp_yesstate_noreward":
            '''preprocessor = SDP_yesstate_noreward(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                reg_model="nn",
                is_normalized=False,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                model_type=fqi_model,
                action_size=2,
                name="sdp_yesstate_noreward",
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)'''
        elif method == "sdp_yesstate_noreward_Oracle":
            '''preprocessor = SDP_yesstate_noreward_Oracle(
                env=env,
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                model_type=fqi_model,
                action_size=2,
                name="sdp_yesstate_noreward_oracle",
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)'''
        elif method == "sdp_nostate_yesreward_Oracle":
            '''preprocessor = SDP_nostate_yesreward_Oracle(
                env=env,
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                model_type=fqi_model,
                action_size=2,
                name="sdp_nostate_yesreward_oracle",
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
            )
            policies.append(agent)'''
        else:
            raise ValueError("Method not found")

    # evaluate reward and fairness
    df_n = pd.DataFrame()
    df_t = pd.DataFrame()
    '''for i, policy in enumerate(policies):
        discounted_cumulative_reward = evaluate_reward_through_simulation(
            env=env, z_levels=np.array([[0], [1]]), state_dim=1, N=N_eval, T=T_eval, 
            policy=policy, seed=eval_seed
        )
        #Z_eval = np.random.binomial(1, 0.5, size=[N_eval]).reshape(N_eval, -1)
        cf_outs = evaluate_fairness_through_simulation(
            env, np.array([[0], [1]]), 1, N_eval, T_eval, policy, seed=eval_seed
        )'''
    
    for i, policy in enumerate(policies):
        discounted_cumulative_reward = evaluate_reward_through_simulation(
            env=env, z_eval_levels=np.array([[0], [1]]), state_dim=1, N=N_eval, T=T_eval, 
            policy=policy, seed=eval_seed
        ) # STATE_DIM IS CHANGED
        '''cf_outs_given_observed_actions = (
            evaluate_fairness_through_simulation_given_observed_actions(
                env, N_eval, T_eval, policy, seed=eval_seed, is_logit=IS_CF_LOGITS
            )
        )'''
        cf_outs = evaluate_fairness_through_simulation(
            env, np.array([[0], [1]]), 1, N_eval, T_eval, policy, seed=eval_seed
        ) # STATE_DIM IS CHANGED

        df_n = pd.concat(
            [
                df_n,
                pd.DataFrame(
                    {
                        "N": [N],
                        "T": [T],
                        "cf": [cf_outs],
                        #"cf_g": [cf_outs_given_observed_actions[0]],
                        "reward": [discounted_cumulative_reward],
                        "method": [methods[i]],
                        "zcoef": [z_coef],
                        "seed": [seed], 
                    }
                ),
            ]
        )
        '''df_t = pd.concat(
            [
                df_t,
                pd.DataFrame(
                    {
                        "N": [N] * T_eval,
                        "T": np.ones(T_eval) * T,
                        "t": range(T_eval),
                        "cf": cf_outs[1],
                        "method": np.repeat([policy.name], repeats=T_eval),
                        "zcoef": [z_coef] * T_eval,
                    }
                ),
            ]
        )'''

    return df_n#, df_t



def run_exp(rep, start_seed=1, export=False):
    #set_env()
    NREP = rep
    #CORES = 20
    #Ns = [100, 200, 500, 1000, 2000]
    Ns = [100]
    Ts = [10]
    # z_coefs = [0, 0.5, 1.0, 1.5, 2.0]
    z_coefs = [1]
    #z_coefs = [0.0, 1.0, 2.0]
    #env_name = "lm4"
    # env_name = "nlm2"
    #suffix = datetime_suffix()
    is_greedy = False
    method_policy = "FQI_NN"

    methods = [
        #"random",
        #"behavior",
        # "sdp_nostate_yesreward",
        # "sdp_yesstate_noreward",
        # "sdp_yesstate_noreward_Oracle",
        # "sdp_nostate_yesreward_Oracle",
        #"full",
        #"unaware",
        "ours",
        #"oracle",
    ]

    mp.set_start_method("spawn")

    df_n = pd.DataFrame()
    #df_t = pd.DataFrame()
    '''jobs = []
    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for _ in range(NREP):
                    jobs.append(
                        (
                            #env_name,
                            methods,
                            method_policy,
                            N,
                            T,
                            #is_greedy,
                            z_coef,
                            _ + 1,
                        )
                    )
    with mp.Pool(CORES) as p:
        out = list(
            tqdm(
                p.imap(run_exp1_one, jobs),
                total=len(jobs),
                desc="run exp",
            )
        )'''
    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for _ in range(NREP):
                    df_n_one = run_exp1_one(methods=methods, 
                          method_policy=method_policy, N=N, T=T, z_coef=z_coef, seed=_+start_seed)
                    df_n = pd.concat([df_n, df_n_one])
                    print(df_n)

    '''for d in out:
        df_n = pd.concat([df_n, d])
        #df_t = pd.concat([df_t, d[1]])'''
    
    if export:
        df_n.to_csv('./examples/simulation_outputs/results.csv')
    
    return df_n

    # unaggregate outfile
    '''df_n.round(3).to_csv(
        "../out/exp1_{}_N_{}_T_{}_greedy{}_zoef{}_method{}_{}.csv".format(
            env_name,
            "_".join([str(N) for N in Ns]),
            "_".join([str(T) for T in Ts]),
            is_greedy,
            "_".join([str(z_coef) for z_coef in z_coefs]),
            method_policy,
            suffix,
        ),
        index=False,
    )

    # aggregate df_t outfile
    df_t_agg = (
        df_t.groupby(["method", "N", "T", "t", "zcoef"])
        .mean()
        .sort_values(["N", "T", "t", "zcoef"], ascending=True)
        .reset_index()
    )
    df_t_agg.round(3).to_csv(
        "../out/df_t/exp1_{}_t_N_{}_T_{}_greedy{}_zoef{}_method{}_{}.csv".format(
            env_name,
            "_".join([str(N) for N in Ns]),
            "_".join([str(T) for T in Ts]),
            is_greedy,
            "_".join([str(z_coef) for z_coef in z_coefs]),
            method_policy,
            suffix,
        ),
        index=False,
    )

    # aggregate df_n outfile
    df_n_agg = (
        df_n.groupby(["method", "N", "T", "zcoef"])
        .mean()
        .sort_values(["N", "T", "method", "zcoef"], ascending=True)
        .reset_index()
    )
    print("NREP:{}, ENV{}, greedy{}".format(NREP, env_name, is_greedy))
    print(df_n_agg)
    df_n_agg.style.format(precision=3).hide().to_latex(
        buf="../out/latex_tables/exp1_{}_N_{}_T_{}_greedy{}_method{}_{}.csv".format(
            env_name,
            "_".join([str(N) for N in Ns]),
            "_".join([str(T) for T in Ts]),
            is_greedy,
            method_policy,
            suffix,
        ),
        position="H",
        position_float="centering",
        hrules=True,
    )'''



# run the experiments
'''s = int(input('Enter the seed that is to be used: '))
df_n = run_exp1_one(methods=['ours'], 
                          method_policy='FQI_NN', N=100, T=10, z_coef=1, seed=s)'''
df_n = run_exp(rep=5, start_seed=1, export=True)
print(df_n)