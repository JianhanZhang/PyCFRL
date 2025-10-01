import pandas as pd
import numpy as np
from pycfrl.environment.environment import SimulatedEnvironment, sample_simulated_env_trajectory
from pycfrl.reader.reader import read_trajectory_from_csv, read_trajectory_from_dataframe
from sklearn.model_selection import train_test_split
from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from .baseline_preprocessors import UnawarenessPreprocessor, ConcatenatePreprocessor
from pycfrl.agents.agents import FQI
from .baseline_agents import BehaviorAgent, RandomAgent
#from policy_learning_add import RandomAgent
from pycfrl.evaluation.evaluation import evaluate_fairness_through_model, evaluate_reward_through_fqe
import torch

def run_exp_one(seed, model_type, z_label, methods):
    # Step 1
    # 1.2 read data
    impute_data = pd.read_csv('data/impute_data.csv')
    impute_data["rldecision"] = impute_data["rldecision"].replace(0, 1)
    impute_data["rldecision"] = impute_data["rldecision"].replace(1, 0)
    impute_data["rldecision"] = impute_data["rldecision"].replace(2, 1)
    impute_data["rldecision"] = impute_data["rldecision"].replace(3, 2)
    #impute_data["rldecision"] = impute_data["rldecision"].replace(3, 1) # TO BE DELETED
    impute_data["sex_b"] = (impute_data["sex"] == 2).astype(int)
    impute_data["age_b"] = (impute_data["age"] >= 40).astype(int)
    impute_data["education_b"] = (impute_data["education"] == "college").astype(int)
    impute_data["race_b"] = (impute_data["race"] == "white").astype(int)
    impute_data = impute_data.sort_values(["pt_id", "week"])
    zs, xs, actions, rewards, ids = read_trajectory_from_dataframe(data=impute_data, 
                                                        z_labels=[z_label], 
                                                        state_labels=["weekly_pain_score", "weekly_pain_interference"], 
                                                        action_label='rldecision', 
                                                        reward_label='reward', 
                                                        id_label='pt_id', 
                                                        T=11, 
                                                        #T=3
                                                        )

    # 1.3 train-test split
    np.random.seed(seed)
    (zs_train, zs_test, 
    xs_train, xs_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test, 
    ids_train, ids_test) = train_test_split(zs, xs, actions, rewards, ids, test_size=0.2)

    # 1.4 train preprocessor
    '''sp = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                action_space=[[0], [1], [2]], 
                                #action_space=[[0], [1]], # TO BE DELETED
                                cross_folds=10, 
                                reg_model='nn', 
                                is_early_stopping=True, 
                                early_stopping_min_delta=0.001)
    up = UnawarenessPreprocessor(
                    z_space=np.unique(zs_train, axis=0), 
                    action_space=np.array([[0], [1], [2]])
                )
    fp = ConcatenatePreprocessor(
                    z_space=np.unique(zs_train, axis=0), 
                    action_space=np.array([[0], [1], [2]])
                )
    xs_tilde, rs_tilde = sp.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)'''



    # Step 2 policy learning
    #prepro_type = input('Please choose the type of preprocessor: ')
    agents = []
    for prepro_type in methods:
        if prepro_type == 'ours':
            sp = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                        num_actions=3, 
                                        #action_space=[[0], [1]], # TO BE DELETED
                                        cross_folds=10, 
                                        reg_model=model_type, 
                                        batch_size=128, 
                                        learning_rate=0.001, # LR THAT JITAO USED
                                        #learning_rate=0.003, # LR THAT I FOUND BETTER WHEN USING JITAO'S NN
                                        is_loss_monitored=True, 
                                        is_early_stopping=True, 
                                        #patience=10, 
                                        #min_delta=0.001
                                        )
            np.random.seed(seed+1)
            torch.manual_seed(seed+1) # NEWLY ADDED
            xs_tilde, rs_tilde = sp.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)
            agent = FQI(model_type=model_type, num_actions=3, preprocessor=sp, learning_rate=0.1, 
                        is_loss_monitored=True, is_early_stopping_nn=True, 
                        is_q_monitored=True, is_early_stopping_q=True
                       )
            #agent = FQI(model_type='nn', action_space=[[0], [1]], preprocessor=sp) # TO BE DELETED
            np.random.seed(seed+2)
            torch.manual_seed(seed+2) # NEWLY ADDED
            agent.train(zs_train, xs_tilde, actions_train, rs_tilde, max_iter=200, preprocess=False)
        elif prepro_type == 'unaware':
            up = UnawarenessPreprocessor(
                            z_space=np.unique(zs_train, axis=0), 
                            action_space=np.array([[0], [1], [2]])
                        )
            agent = FQI(model_type=model_type, num_actions=3, preprocessor=up, learning_rate=0.1)
            np.random.seed(seed+2)
            torch.manual_seed(seed+2) # NEWLY ADDED
            agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=200, preprocess=True)
        elif prepro_type == 'full':
            fp = ConcatenatePreprocessor(
                            z_space=np.unique(zs_train, axis=0), 
                            action_space=np.array([[0], [1], [2]])
                        )
            agent = FQI(model_type=model_type, num_actions=3, preprocessor=fp, learning_rate=0.1)
            np.random.seed(seed+2)
            torch.manual_seed(seed+2) # NEWLY ADDED
            agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=200, preprocess=True)
        elif prepro_type == 'random':
            agent = RandomAgent(3)
        else:
            print('Undefined preprocessor.')
            exit(1)
        agents.append(agent)
    #agent.train(zs_train, xs_tilde, actions_train, rs_tilde, max_iter=200, preprocess=False)
    #agent = RandomAgent(3)



    # Step 3
    # 3.2 train transition kernel
    np.random.seed(seed+3)
    torch.manual_seed(seed+3) # NEWLY ADDED
    env = SimulatedEnvironment(state_model_type=model_type, 
                            reward_model_type=model_type, 
                            num_actions=3, 
                            is_action_onehot=True, 
                            is_loss_monitored=True, 
                            is_early_stopping=True)
    env.fit(zs, xs, actions, rewards)
    #env.reset(zs)

    # 3.3 generate evaluation samples
    '''metadata = SingleTableMetadata()
    df = pd.read_csv('./impute_data.csv')
    df = df[['sex']]
    metadata.detect_from_dataframe(data=df)
    metadata.update_column(column_name='sex', sdtype='categorical')
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=100,
    )
    synthesizer.fit(df)
    synthetic_zs = synthesizer.sample(num_rows=5, output_file_path="disable").to_numpy()
    synthetic_zs = np.array([[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]]) # for debug convenience'''

    # 3.4 generate counterfactual trajectories
    '''sample_counterfactual_simulated_env_trajectories(env, 
                                                    zs, 
                                                    [[0], [1]], 
                                                    2, 
                                                    zs.shape[0], 
                                                    12, 
                                                    agent)'''


    out = None
    for i, method in enumerate(methods):
        # Step 4 calculate the CF metric
        '''cf_metric = evaluate_fairness_through_simulated_environment(env, 
                                                    zs, 
                                                    [[0], [1]], 
                                                    2, 
                                                    zs.shape[0], 
                                                    12, 
                                                    fqi)'''
        cf_metric = evaluate_fairness_through_model(env, 
                                                    zs_test, 
                                                    #[[0], [1]], 
                                                    xs_test, 
                                                    #2, 
                                                    actions_test, 
                                                    #zs.shape[0], 
                                                    #5, 
                                                    agents[i], 
                                                    seed=seed+4)
        
        # Step 5 FQE
        np.random.seed(seed+4)
        torch.manual_seed(seed+4) # NEWLY ADDED
        value = evaluate_reward_through_fqe(zs=zs_test, 
                                            states=xs_test, 
                                            actions=actions_test, 
                                            rewards=rewards_test, 
                                            model_type=model_type, 
                                            policy=agents[i], 
                                            seed=seed+4, 
                                            is_loss_monitored=True,
                                            is_early_stopping_nn=True,
                                            early_stopping_patience_nn=10,
                                            is_q_monitored=True,
                                            is_early_stopping_q=True,
                                            #max_iter=1000
                                           )
        
        #print('fairness:', cf_metric)
        tmp = pd.DataFrame(
                {
                    "method": [method],
                    "z_label": [z_label],
                    "cf_metric": [cf_metric],
                    "value": [value],
                    #"mean_actions_z1gez0": [mean_actions_z1gez0],
                    #"mean_actions_z1lez0": [mean_actions_z1lez0],
                    #"mean_actions_z1eqz0": [mean_actions_z1eqz0],
                    "model_type": [model_type], 
                    "seed": [seed],
                }
            )
        out = pd.concat([out, tmp], axis=0, ignore_index=True)

    return out



def run_exp(rep, model_type, z_labels, methods, print_res=True, export_res=False, 
            export_path='./res.csv'):
    res = None
    for i in range(rep):
        for z_label in z_labels: 
            res_add = run_exp_one(seed=i+10, model_type=model_type, z_label=z_label, methods=methods)
            #res_add = run_exp(seed=12, method='unaware')
            #res_add = run_exp(seed=i+50, method='unaware')
            res = pd.concat([res, res_add], axis=0, ignore_index=True)
            if print_res:
                print(res_add)
    #print(res)
    if export_res:
        res.to_csv(export_path)



'''run_exp(rep=10, methods=['ours'], z_labels=['sex_b'], 
        print_res=True, export_res=True, 
        export_path='./result_rda_real_data_analysis_after.csv')'''
run_exp(rep=1, model_type='nn', methods=['ours'], z_labels=['sex_b'], 
        print_res=True, export_res=False)