import pandas as pd
import numpy as np
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from pycfrl.environment.environment import SimulatedEnvironment, sample_simulated_env_trajectory
from pycfrl.reader.reader import read_trajectory_from_csv, read_trajectory_from_dataframe
from sklearn.model_selection import train_test_split
from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from examples.baseline_preprocessors import UnawarenessPreprocessor, ConcatenatePreprocessor
from pycfrl.agents.agents import FQI
from examples.baseline_agents import BehaviorAgent, RandomAgent
#from policy_learning_add import RandomAgent
from pycfrl.evaluation.evaluation import evaluate_fairness_through_model, evaluate_reward_through_fqe
import torch
import pytest

# an environment with univariate zs and states
# x0 = 0.5 + zs + ux0 (assuming z_coef=1)
def f_x0_uni(
        zs: np.ndarray, 
        ux0: np.ndarray, 
        z_coef: int = 1
    ) -> np.ndarray:
    gamma0 = np.array([0.5, 1 * z_coef, 1])
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

# xt = -0.5 + zs + 3 * xtm1 + 2 * atm1 + uxt (assuming z_coef=1)
def f_xt_uni(
        zs: np.ndarray, 
        xtm1: np.ndarray, 
        atm1: np.ndarray, 
        uxt: np.ndarray, 
        z_coef: int = 1
    ) -> np.ndarray:
    gamma = np.array([-0.5, 1 * z_coef, 3, 2, 1])
    n = xtm1.shape[0]
    M = np.concatenate(
        [
            np.ones([n, 1]),
            zs,
            xtm1,
            atm1.reshape(-1, 1), 
            uxt,
        ],
        axis=1,
    )
    xt = M @ gamma
    xt = xt.reshape(-1, 1)
    return xt

# xt = -0.5 + zs + xt + at + urt (assuming z_coef=1)
def f_rt_uni(
        zs: np.ndarray, 
        xt: np.ndarray, 
        at: np.ndarray, 
        urtm1: np.ndarray, 
        z_coef: int = 1
    ) -> np.ndarray:
    lmbda = np.array([-0.5, 1, 1 * z_coef, 1, 1])
    n = xt.shape[0]
    at = at.reshape(-1, 1)
    M = np.concatenate(
        [np.ones([n, 1]), xt, zs, at, urtm1], axis=1
    )
    rt = M @ lmbda
    return rt



# an environment with bivariate zs and trivariate states
# x0_i = 0.5 + zs_1 + zs_2 + ux0_i (assuming z_coef=1)
def f_x0_multi(
        zs: np.ndarray, 
        ux0: np.ndarray, 
        z_coef: int = 1
    ) -> np.ndarray:
    gamma0 = np.array([np.repeat(np.array([0.5]), repeats=3), 
                       np.repeat(np.array(1 * z_coef), repeats=3), 
                       np.repeat(np.array(1 * z_coef), repeats=3)])
    n = zs.shape[0]
    M = np.concatenate(
        [
            np.ones([n, 1]),
            zs,
        ],
        axis=1,
    )
    x0 = M @ gamma0
    x0 = x0.reshape(-1, 3)
    x0 = x0 + ux0
    return x0

# xt_i = -0.5 + zs_1 + zs_2 + 3 * (xtm1_1 + xtm1_2 + xtm1_3) + 2 * atm1 + uxt (assuming z_coef=1)
def f_xt_multi(
        zs: np.ndarray, 
        xtm1: np.ndarray, 
        atm1: np.ndarray, 
        uxt: np.ndarray, 
        z_coef: int = 1
    ) -> np.ndarray:
    gamma = np.array([np.repeat(np.array([-0.5]), repeats=3), 
                      np.repeat(np.array(1 * z_coef), repeats=3), 
                      np.repeat(np.array(1 * z_coef), repeats=3), 
                      np.array([3, 3, 3]),
                      np.array([3, 3, 3]), 
                      np.array([3, 3, 3]), 
                      np.array([2, 2, 2])])
    n = xtm1.shape[0]
    M = np.concatenate(
        [
            np.ones([n, 1]),
            zs,
            xtm1,
            atm1.reshape(-1, 1), 
        ],
        axis=1,
    )
    xt = M @ gamma
    xt = xt.reshape(-1, 3)
    xt = xt + uxt
    return xt

# xt = -0.5 + zs_1 + zs_2 + xt_1 + xt_2 + xt_3 + at + urt (assuming z_coef=1)
def f_rt_multi(
        zs: np.ndarray, 
        xt: np.ndarray, 
        at: np.ndarray, 
        urtm1: np.ndarray, 
        z_coef: int = 1
    ) -> np.ndarray:
    lmbda = np.array([-0.5, 1, 1, 1, 1 * z_coef, 1 * z_coef, 1, 1])
    n = xt.shape[0]
    at = at.reshape(-1, 1)
    M = np.concatenate(
        [np.ones([n, 1]), xt, zs, at, urtm1], axis=1
    )
    rt = M @ lmbda
    return rt



def test_real_data_analysis_univariate_zs_states_nn():
    # simulate synthetic data tarjectories
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    Z = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2], 
         [0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env_true, 
                                                     zs=Z, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (30, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (30, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (30, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

    # create training and testing sets
    (zs_train, zs_test, 
    xs_train, xs_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test) = train_test_split(zs, states, actions, rewards, test_size=0.2)

    # train the preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                          num_actions=2, 
                                          cross_folds=3, 
                                          reg_model='nn', 
                                          batch_size=128, 
                                          learning_rate=0.001, 
                                          epochs=100,  
                                          is_early_stopping=True, 
                                          early_stopping_patience=10, 
                                          early_stopping_min_delta=0.001)
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs_train, 
                                                         xs=xs_train, 
                                                         actions=actions_train, 
                                                         rewards=rewards_train)
    assert(xs_tilde.shape == (24, 11, 3))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (24, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))
    agent = FQI(model_type='nn', num_actions=2, preprocessor=preprocessor, epochs=10)
    agent.train(zs=zs_train, xs=xs_tilde, actions=actions_train, rewards=rs_tilde, 
                max_iter=10, preprocess=False)

    env = SimulatedEnvironment(state_model_type='nn', 
                               reward_model_type='nn', 
                               num_actions=2)
    env.fit(zs=zs_train, states=xs_train, actions=actions_train, rewards=rewards_train)

    # evaluate the value and CF metric
    value = evaluate_reward_through_fqe(zs=zs_test, 
                                        states=xs_test, 
                                        actions=actions_test, 
                                        rewards=rewards_test, 
                                        model_type='nn', 
                                        policy=agent, 
                                        epochs=10, 
                                        max_iter=10)
    cf_metric = evaluate_fairness_through_model(env=env, 
                                                zs=zs_test, 
                                                states=xs_test, 
                                                actions=actions_test, 
                                                policy=agent)
    print('test_real_data_analysis_univariate_zs_states_nn():')
    print('Value:', value)
    print('CF metric:', cf_metric)

def test_real_data_analysis_multivariate_zs_states_nn():
    # simulate synthetic data tarjectories
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    Z = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], 
         [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], 
         [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env_true, 
                                                     zs=Z, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (24, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (24, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (24, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

    # create training and testing sets
    (zs_train, zs_test, 
    xs_train, xs_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test) = train_test_split(zs, states, actions, rewards, test_size=0.25)

    # train the preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                          num_actions=2, 
                                          cross_folds=3, 
                                          reg_model='nn', 
                                          batch_size=128, 
                                          learning_rate=0.001, 
                                          epochs=10, 
                                          is_early_stopping=True, 
                                          early_stopping_patience=10, 
                                          early_stopping_min_delta=0.001)
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs_train, 
                                                         xs=xs_train, 
                                                         actions=actions_train, 
                                                         rewards=rewards_train)
    assert(xs_tilde.shape == (18, 11, 6))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (18, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))
    agent = FQI(model_type='nn', num_actions=2, preprocessor=preprocessor, epochs=10)
    agent.train(zs=zs_train, xs=xs_tilde, actions=actions_train, rewards=rs_tilde, 
                max_iter=10, preprocess=False)

    env = SimulatedEnvironment(state_model_type='nn', 
                               reward_model_type='nn', 
                               num_actions=2)
    env.fit(zs=zs_train, states=xs_train, actions=actions_train, rewards=rewards_train)

    # evaluate the value and CF metric
    value = evaluate_reward_through_fqe(zs=zs_test, 
                                        states=xs_test, 
                                        actions=actions_test, 
                                        rewards=rewards_test, 
                                        model_type='nn', 
                                        policy=agent, 
                                        epochs=10, 
                                        max_iter=10)
    cf_metric = evaluate_fairness_through_model(env=env, 
                                                zs=zs_test, 
                                                states=xs_test, 
                                                actions=actions_test, 
                                                policy=agent)
    print('test_real_data_analysis_multivariate_zs_states_nn():')
    print('Value:', value)
    print('CF metric:', cf_metric)



@pytest.mark.skip(reason="Skipped because 'lm' is currently not supported.")
def test_real_data_analysis_univariate_zs_states_lm():
    # simulate synthetic data tarjectories
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    Z = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2], 
         [0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env_true, 
                                                     zs=Z, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (30, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (30, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (30, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

    # create training and testing sets
    (zs_train, zs_test, 
    xs_train, xs_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test) = train_test_split(zs, states, actions, rewards, test_size=0.2)

    # train the preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                          num_actions=2, 
                                          cross_folds=3, 
                                          reg_model='lm', 
                                          batch_size=128, 
                                          learning_rate=0.001, 
                                          epochs=100,  
                                          is_early_stopping=True, 
                                          early_stopping_patience=10, 
                                          early_stopping_min_delta=0.001)
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs_train, 
                                                         xs=xs_train, 
                                                         actions=actions_train, 
                                                         rewards=rewards_train)
    assert(xs_tilde.shape == (24, 11, 3))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (24, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))
    agent = FQI(model_type='lm', num_actions=2, preprocessor=preprocessor, epochs=10)
    agent.train(zs=zs_train, xs=xs_tilde, actions=actions_train, rewards=rs_tilde, 
                max_iter=10, preprocess=False)

    env = SimulatedEnvironment(state_model_type='lm', 
                               reward_model_type='lm', 
                               num_actions=2)
    env.fit(zs=zs_train, states=xs_train, actions=actions_train, rewards=rewards_train)

    # evaluate the value and CF metric
    value = evaluate_reward_through_fqe(zs=zs_test, 
                                        states=xs_test, 
                                        actions=actions_test, 
                                        rewards=rewards_test, 
                                        model_type='lm', 
                                        policy=agent, 
                                        epochs=10, 
                                        max_iter=10)
    cf_metric = evaluate_fairness_through_model(env=env, 
                                                zs=zs_test, 
                                                states=xs_test, 
                                                actions=actions_test, 
                                                policy=agent)
    print('test_real_data_analysis_univariate_zs_states_lm():')
    print('Value:', value)
    print('CF metric:', cf_metric)

@pytest.mark.skip(reason="Skipped because 'lm' is currently not supported.")
def test_real_data_analysis_multivariate_zs_states_lm():
    # simulate synthetic data tarjectories
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    Z = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], 
         [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], 
         [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env_true, 
                                                     zs=Z, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (24, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (24, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (24, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

    # create training and testing sets
    (zs_train, zs_test, 
    xs_train, xs_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test) = train_test_split(zs, states, actions, rewards, test_size=0.25)

    # train the preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                          num_actions=2, 
                                          cross_folds=3, 
                                          reg_model='lm', 
                                          batch_size=128, 
                                          learning_rate=0.001, 
                                          epochs=10, 
                                          is_early_stopping=True, 
                                          early_stopping_patience=10, 
                                          early_stopping_min_delta=0.001)
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs_train, 
                                                         xs=xs_train, 
                                                         actions=actions_train, 
                                                         rewards=rewards_train)
    assert(xs_tilde.shape == (18, 11, 6))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (18, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))
    agent = FQI(model_type='lm', num_actions=2, preprocessor=preprocessor, epochs=10)
    agent.train(zs=zs_train, xs=xs_tilde, actions=actions_train, rewards=rs_tilde, 
                max_iter=10, preprocess=False)

    env = SimulatedEnvironment(state_model_type='lm', 
                               reward_model_type='lm', 
                               num_actions=2)
    env.fit(zs=zs_train, states=xs_train, actions=actions_train, rewards=rewards_train)

    # evaluate the value and CF metric
    value = evaluate_reward_through_fqe(zs=zs_test, 
                                        states=xs_test, 
                                        actions=actions_test, 
                                        rewards=rewards_test, 
                                        model_type='lm', 
                                        policy=agent, 
                                        epochs=10, 
                                        max_iter=10)
    cf_metric = evaluate_fairness_through_model(env=env, 
                                                zs=zs_test, 
                                                states=xs_test, 
                                                actions=actions_test, 
                                                policy=agent)
    print('test_real_data_analysis_multivariate_zs_states_lm():')
    print('Value:', value)
    print('CF metric:', cf_metric)


# run the tests
test_real_data_analysis_univariate_zs_states_nn()
test_real_data_analysis_multivariate_zs_states_nn()
#test_real_data_analysis_univariate_zs_states_lm()
#test_real_data_analysis_multivariate_zs_states_lm()
print('All real data analysis integration tests passed!')