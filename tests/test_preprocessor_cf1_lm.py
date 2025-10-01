from pycfrl.preprocessor.preprocessor import SequentialPreprocessor
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from examples.baseline_agents import RandomAgent
import numpy as np

import pytest
pytest.skip("Skipping this entire test file because 'lm' is currently not supported.", 
            allow_module_level=True)

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



def test_train_preprocessor_univariate_zs_states_single_lm():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    preprocessor = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                          num_actions=2, 
                                          reg_model='lm', 
                                          epochs=10, 
                                          cross_folds=1, 
                                          mode='single')
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs, 
                                                         xs=states, 
                                                         actions=actions, 
                                                         rewards=rewards)
    
    assert(xs_tilde.shape == (9, 11, 3))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (9, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))

def test_train_preprocessor_multivariate_zs_states_single_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    preprocessor = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                          num_actions=2, 
                                          reg_model='lm', 
                                          epochs=10, 
                                          cross_folds=1, 
                                          mode='single')
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs, 
                                                         xs=states, 
                                                         actions=actions, 
                                                         rewards=rewards)
    
    assert(xs_tilde.shape == (8, 11, 12))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (8, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))

def test_train_preprocessor_univariate_zs_states_sensitive_lm():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    preprocessor = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                          num_actions=2, 
                                          reg_model='lm', 
                                          epochs=10, 
                                          cross_folds=1, 
                                          mode='sensitive')
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs, 
                                                         xs=states, 
                                                         actions=actions, 
                                                         rewards=rewards)
    
    assert(xs_tilde.shape == (9, 11, 3))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (9, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))

def test_train_preprocessor_multivariate_zs_states_sensitive_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    preprocessor = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                          num_actions=2, 
                                          reg_model='lm', 
                                          epochs=10, 
                                          cross_folds=1, 
                                          mode='sensitive')
    xs_tilde, rs_tilde = preprocessor.train_preprocessor(zs=zs, 
                                                         xs=states, 
                                                         actions=actions, 
                                                         rewards=rewards)
    
    assert(xs_tilde.shape == (8, 11, 12))
    assert(np.issubdtype(xs_tilde.dtype, np.floating))
    assert(rs_tilde.shape == (8, 10))
    assert(np.issubdtype(rs_tilde.dtype, np.floating))


def test_preprocess_single_step_univariate_zs_states_single_lm():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p2 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    # test preprocessing the first step
    zs = np.array([[0], [0], [1], [1], [2], [2]])
    ux0 = np.random.normal(0, 1, size=[6, 1])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 3))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 3))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    # test preprocessing subsequent steps
    ux1 = np.random.normal(0, 1, size=[6, 1])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 3))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 3))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 1])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 3))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 3))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

    # test restarting the preprocessor
    ux0 = np.random.normal(0, 1, size=[6, 1])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 3))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 3))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    ux1 = np.random.normal(0, 1, size=[6, 1])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 3))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 3))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 1])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 3))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 3))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

def test_preprocess_single_step_multivariate_zs_states_single_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p2 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    # test preprocessing the first step
    zs = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])
    ux0 = np.random.normal(0, 1, size=[6, 3])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 12))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 12))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    # test preprocessing subsequent steps
    ux1 = np.random.normal(0, 1, size=[6, 3])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 12))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 12))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 3])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 12))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 12))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

    # test restarting the preprocessor
    zs = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])
    ux0 = np.random.normal(0, 1, size=[6, 3])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 12))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 12))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    ux1 = np.random.normal(0, 1, size=[6, 3])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 12))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 12))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 3])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 12))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 12))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

def test_preprocess_single_step_univariate_zs_states_sensitive_lm():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p2 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    # test preprocessing the first step
    zs = np.array([[0], [0], [1], [1], [2], [2]])
    ux0 = np.random.normal(0, 1, size=[6, 1])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 3))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 3))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    # test preprocessing subsequent steps
    ux1 = np.random.normal(0, 1, size=[6, 1])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 3))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 3))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 1])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 3))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 3))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

    # test restarting the preprocessor
    ux0 = np.random.normal(0, 1, size=[6, 1])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 3))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 3))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    ux1 = np.random.normal(0, 1, size=[6, 1])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 3))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 3))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 1])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 3))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 3))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

def test_preprocess_single_step_multivariate_zs_states_sensitive_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p2 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    # test preprocessing the first step
    zs = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])
    ux0 = np.random.normal(0, 1, size=[6, 3])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 12))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 12))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    # test preprocessing subsequent steps
    ux1 = np.random.normal(0, 1, size=[6, 3])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 12))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 12))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 3])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 12))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 12))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

    # test restarting the preprocessor
    zs = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])
    ux0 = np.random.normal(0, 1, size=[6, 3])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_tilde1 = p1.preprocess_single_step(z=zs, xt=x0)
    x0_tilde2 = p2.preprocess_single_step(z=zs, xt=x0)
    assert(x0_tilde1.shape == (6, 12))
    assert(np.issubdtype(x0_tilde1.dtype, np.floating))
    assert(x0_tilde2.shape == (6, 12))
    assert(np.issubdtype(x0_tilde2.dtype, np.floating))

    ux1 = np.random.normal(0, 1, size=[6, 3])
    ur0 = np.random.normal(0, 1, size=[6, 1])
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_tilde1, r0_tilde1 = p1.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0, rtm1=r0)
    x1_tilde2 = p2.preprocess_single_step(z=zs, xt=x1, xtm1=x0, atm1=a0)
    assert(x1_tilde1.shape == (6, 12))
    assert(np.issubdtype(x1_tilde1.dtype, np.floating))
    assert(r0_tilde1.shape == (6,))
    assert(np.issubdtype(r0_tilde1.dtype, np.floating))
    assert(x1_tilde2.shape == (6, 12))
    assert(np.issubdtype(x1_tilde2.dtype, np.floating))

    ux2 = np.random.normal(0, 1, size=[6, 3])
    ur1 = np.random.normal(0, 1, size=[6, 1])
    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_tilde1, r1_tilde1 = p1.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1, rtm1=r1)
    x2_tilde2 = p2.preprocess_single_step(z=zs, xt=x2, xtm1=x1, atm1=a1)
    assert(x2_tilde1.shape == (6, 12))
    assert(np.issubdtype(x2_tilde1.dtype, np.floating))
    assert(r1_tilde1.shape == (6,))
    assert(np.issubdtype(r1_tilde1.dtype, np.floating))
    assert(x2_tilde2.shape == (6, 12))
    assert(np.issubdtype(x2_tilde2.dtype, np.floating))

def test_preprocess_multiple_steps_univariate_zs_states_single_lm():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p2 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    zs_test = np.array([[0], [0], [1], [1], [2], [2]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env, 
                                                                         zs=zs_test, 
                                                                         state_dim=1, 
                                                                         T=10, 
                                                                         policy=agent)
    xs_tilde1, rs_tilde1 = p1.preprocess_multiple_steps(zs=zs_test, 
                                                        xs=states_test, 
                                                        actions=actions_test, 
                                                        rewards=rewards_test)
    xs_tilde2 = p2.preprocess_multiple_steps(zs=zs_test, 
                                             xs=states_test, 
                                             actions=actions_test)
    assert(xs_tilde1.shape == (6, 11, 3))
    assert(np.issubdtype(xs_tilde1.dtype, np.floating))
    assert(rs_tilde1.shape == (6, 10))
    assert(np.issubdtype(rs_tilde1.dtype, np.floating))
    assert(xs_tilde2.shape == (6, 11, 3))
    assert(np.issubdtype(xs_tilde2.dtype, np.floating))
    

def test_preprocess_multiple_steps_multivariate_zs_states_single_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p2 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='single')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    zs_test = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env, 
                                                                         zs=zs_test, 
                                                                         state_dim=3, 
                                                                         T=10, 
                                                                         policy=agent)
    xs_tilde1, rs_tilde1 = p1.preprocess_multiple_steps(zs=zs_test, 
                                                        xs=states_test, 
                                                        actions=actions_test, 
                                                        rewards=rewards_test)
    xs_tilde2 = p2.preprocess_multiple_steps(zs=zs_test, 
                                             xs=states_test, 
                                             actions=actions_test)
    assert(xs_tilde1.shape == (6, 11, 12))
    assert(np.issubdtype(xs_tilde1.dtype, np.floating))
    assert(rs_tilde1.shape == (6, 10))
    assert(np.issubdtype(rs_tilde1.dtype, np.floating))
    assert(xs_tilde2.shape == (6, 11, 12))
    assert(np.issubdtype(xs_tilde2.dtype, np.floating))

def test_preprocess_multiple_steps_univariate_zs_states_sensitive_lm():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p2 = SequentialPreprocessor(z_space=[[0], [1], [2]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    zs_test = np.array([[0], [0], [1], [1], [2], [2]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env, 
                                                                         zs=zs_test, 
                                                                         state_dim=1, 
                                                                         T=10, 
                                                                         policy=agent)
    xs_tilde1, rs_tilde1 = p1.preprocess_multiple_steps(zs=zs_test, 
                                                        xs=states_test, 
                                                        actions=actions_test, 
                                                        rewards=rewards_test)
    xs_tilde2 = p2.preprocess_multiple_steps(zs=zs_test, 
                                             xs=states_test, 
                                             actions=actions_test)
    assert(xs_tilde1.shape == (6, 11, 3))
    assert(np.issubdtype(xs_tilde1.dtype, np.floating))
    assert(rs_tilde1.shape == (6, 10))
    assert(np.issubdtype(rs_tilde1.dtype, np.floating))
    assert(xs_tilde2.shape == (6, 11, 3))
    assert(np.issubdtype(xs_tilde2.dtype, np.floating))

def test_preprocess_multiple_steps_multivariate_zs_states_sensitive_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    p1 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p2 = SequentialPreprocessor(z_space=[[0, 1], [1, 0], [0, 0], [1, 1]], 
                                num_actions=2, 
                                reg_model='lm', 
                                epochs=10, 
                                is_action_onehot=False, 
                                cross_folds=1, 
                                mode='sensitive')
    p1.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    p2.train_preprocessor(zs=zs, 
                          xs=states, 
                          actions=actions, 
                          rewards=rewards)
    
    zs_test = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env, 
                                                                         zs=zs_test, 
                                                                         state_dim=3, 
                                                                         T=10, 
                                                                         policy=agent)
    xs_tilde1, rs_tilde1 = p1.preprocess_multiple_steps(zs=zs_test, 
                                                        xs=states_test, 
                                                        actions=actions_test, 
                                                        rewards=rewards_test)
    xs_tilde2 = p2.preprocess_multiple_steps(zs=zs_test, 
                                             xs=states_test, 
                                             actions=actions_test)
    assert(xs_tilde1.shape == (6, 11, 12))
    assert(np.issubdtype(xs_tilde1.dtype, np.floating))
    assert(rs_tilde1.shape == (6, 10))
    assert(np.issubdtype(rs_tilde1.dtype, np.floating))
    assert(xs_tilde2.shape == (6, 11, 12))
    assert(np.issubdtype(xs_tilde2.dtype, np.floating))



# run the tests
'''test_train_preprocessor_univariate_zs_states_single_lm()
test_train_preprocessor_multivariate_zs_states_single_lm()
test_train_preprocessor_univariate_zs_states_sensitive_lm()
test_train_preprocessor_multivariate_zs_states_sensitive_lm()
test_preprocess_single_step_univariate_zs_states_single_lm()
test_preprocess_single_step_multivariate_zs_states_single_lm()
test_preprocess_single_step_univariate_zs_states_sensitive_lm()
test_preprocess_single_step_multivariate_zs_states_sensitive_lm()
test_preprocess_multiple_steps_univariate_zs_states_single_lm()
test_preprocess_multiple_steps_multivariate_zs_states_single_lm()
test_preprocess_multiple_steps_univariate_zs_states_sensitive_lm()
test_preprocess_multiple_steps_multivariate_zs_states_sensitive_lm()
print('All preprocessor tests with 1 cross folds passed!')'''