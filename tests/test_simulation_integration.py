from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from examples.baseline_preprocessors import SequentialPreprocessorOracle, UnawarenessPreprocessor
from examples.baseline_preprocessors import ConcatenatePreprocessor
from pycfrl.agents.agents import FQI
from examples.baseline_agents import RandomAgent, BehaviorAgent
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



@pytest.mark.skip(reason="Skipped because 'lm' is currently not supported.")
def test_simulation_univariate_zs_states_lm():
    # simulate synthetic data tarjectories
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    Z = [[0], [1], [2], [0], [1], [2], [0], [1], [2]]
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=Z, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (9, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (9, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (9, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))
    
    # train preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                          num_actions=2, 
                                          reg_model='lm', 
                                          is_normalized=False)
    preprocessor.train_preprocessor(xs=states, zs=zs, actions=actions, rewards=rewards)
    agent = FQI(preprocessor=preprocessor, 
                model_type='lm', 
                num_actions=2, 
                epochs=10)
    agent.train(xs=states, zs=zs, actions=actions, rewards=rewards, max_iter=10)
    
    # evalate discounted cumulated reward and CF metric
    dcr = evaluate_reward_through_simulation(env=env, 
                                             z_eval_levels=[[0], [1], [2]], 
                                             state_dim=1, 
                                             N=10, 
                                             T=10, 
                                             policy=agent)
    cf_metric = evaluate_fairness_through_simulation(env=env, 
                                                     z_eval_levels=[[0], [1], [2]], 
                                                     state_dim=1, 
                                                     N=10, 
                                                     T=10, 
                                                     policy=agent)
    #assert(np.issubdtype(dcr.dtype, np.floating))
    #assert(np.issubdtype(cf_metric.dtype, np.floating))
    print('test_simulation_univariate_zs_states_lm():')
    print('Discounted cumulative reward:', dcr)
    print('CF metric:', cf_metric)

@pytest.mark.skip(reason="Skipped because 'lm' is currently not supported.")
def test_simulation_multivariate_zs_states_lm():
    # simulate synthetic data tarjectories
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    Z = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=Z, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (8, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (8, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (8, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))
    
    # train preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=[[0, 1], [1, 0]], 
                                          num_actions=2, 
                                          reg_model='lm', 
                                          is_normalized=False)
    preprocessor.train_preprocessor(xs=states, zs=zs, actions=actions, rewards=rewards)
    agent = FQI(preprocessor=preprocessor, 
                model_type='lm', 
                num_actions=2, 
                epochs=10)
    agent.train(xs=states, zs=zs, actions=actions, rewards=rewards, max_iter=10)
    
    # evalate discounted cumulated reward and CF metric
    dcr = evaluate_reward_through_simulation(env=env, 
                                             z_eval_levels=[[0, 1], [1, 0]], 
                                             state_dim=3, 
                                             N=10, 
                                             T=10, 
                                             policy=agent)
    cf_metric = evaluate_fairness_through_simulation(env=env, 
                                                     z_eval_levels=[[0, 1], [1, 0]], 
                                                     state_dim=3, 
                                                     N=10, 
                                                     T=10, 
                                                     policy=agent)
    #assert(np.issubdtype(dcr.dtype, np.floating))
    #assert(np.issubdtype(cf_metric.dtype, np.floating))
    print('test_simulation_univariate_zs_states_lm():')
    print('Discounted cumulative reward:', dcr)
    print('CF metric:', cf_metric)



def test_simulation_univariate_zs_states_nn():
    # simulate synthetic data tarjectories
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    Z = [[0], [1], [2], [0], [1], [2], [0], [1], [2]]
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=Z, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (9, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (9, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (9, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))
    
    # train preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                          num_actions=2, 
                                          reg_model='nn', 
                                          is_normalized=False)
    preprocessor.train_preprocessor(xs=states, zs=zs, actions=actions, rewards=rewards)
    agent = FQI(preprocessor=preprocessor, 
                model_type='nn', 
                num_actions=2, 
                epochs=10)
    agent.train(xs=states, zs=zs, actions=actions, rewards=rewards, max_iter=10)
    
    # evalate discounted cumulated reward and CF metric
    dcr = evaluate_reward_through_simulation(env=env, 
                                             z_eval_levels=[[0], [1], [2]], 
                                             state_dim=1, 
                                             N=10, 
                                             T=10, 
                                             policy=agent)
    cf_metric = evaluate_fairness_through_simulation(env=env, 
                                                     z_eval_levels=[[0], [1], [2]], 
                                                     state_dim=1, 
                                                     N=10, 
                                                     T=10, 
                                                     policy=agent)
    #assert(np.issubdtype(dcr.dtype, np.floating))
    #assert(np.issubdtype(cf_metric.dtype, np.floating))
    print('test_simulation_univariate_zs_states_nn():')
    print('Discounted cumulative reward:', dcr)
    print('CF metric:', cf_metric)

def test_simulation_multivariate_zs_states_nn():
    # simulate synthetic data tarjectories
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    Z = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    working_policy = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=Z, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=working_policy)
    assert(np.array_equal(zs, Z))
    assert(states.shape == (8, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (8, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (8, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))
    
    # train preprocessor and FQI
    preprocessor = SequentialPreprocessor(z_space=[[0, 1], [1, 0]], 
                                          num_actions=2, 
                                          reg_model='nn', 
                                          is_normalized=False)
    preprocessor.train_preprocessor(xs=states, zs=zs, actions=actions, rewards=rewards)
    agent = FQI(preprocessor=preprocessor, 
                model_type='nn', 
                num_actions=2, 
                epochs=10)
    agent.train(xs=states, zs=zs, actions=actions, rewards=rewards, max_iter=10)
    
    # evalate discounted cumulated reward and CF metric
    dcr = evaluate_reward_through_simulation(env=env, 
                                             z_eval_levels=[[0, 1], [1, 0]], 
                                             state_dim=3, 
                                             N=10, 
                                             T=10, 
                                             policy=agent)
    cf_metric = evaluate_fairness_through_simulation(env=env, 
                                                     z_eval_levels=[[0, 1], [1, 0]], 
                                                     state_dim=3, 
                                                     N=10, 
                                                     T=10, 
                                                     policy=agent)
    #assert(np.issubdtype(dcr.dtype, np.floating))
    #assert(np.issubdtype(cf_metric.dtype, np.floating))
    print('test_simulation_univariate_zs_states_nn():')
    print('Discounted cumulative reward:', dcr)
    print('CF metric:', cf_metric)



# run the tests
#test_simulation_univariate_zs_states_lm()
#test_simulation_multivariate_zs_states_lm()
test_simulation_univariate_zs_states_nn()
test_simulation_multivariate_zs_states_nn()
print('All simulation integration tests passed!')