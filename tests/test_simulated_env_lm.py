from pycfrl.environment.environment import SimulatedEnvironment, SyntheticEnvironment
from pycfrl.environment.environment import sample_simulated_env_trajectory, sample_trajectory
from pycfrl.environment.environment import estimate_counterfactual_trajectories_from_data
from pycfrl.environment.environment import sample_counterfactual_trajectories
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



# custom errors for states
f_counter_states = 0
def f_errors_states(N: int, state_dim: int) -> np.ndarray:
    global f_counter_states 
    f_counter_states += 1
    return np.random.multivariate_normal(
                mean=np.ones(state_dim),
                cov=np.diag(np.ones(state_dim)), 
                size=N,
            )

# custom errors for rewards
f_counter_rewards = 0
def f_errors_rewards(N: int) -> np.ndarray:
    global f_counter_rewards
    f_counter_rewards += 1
    return np.random.normal(
                loc=1, scale=1, size=N
            )

# custom exogenous U for actions
f_counter_actions = 0
def f_ua(N: int) -> np.ndarray:
    global f_counter_actions
    f_counter_actions += 1
    return np.random.uniform(-1, 1, size=[N])



def test_simulated_env_univariate_zs_states_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env_true, 
                                                     zs=zs_in, 
                                                     state_dim=1, 
                                                     T=10, 
                                                     policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs, states=states, actions=actions, rewards=rewards)
    
    # test taking the first step
    x0, _ = env.reset(z=zs)
    assert(x0.shape == (9, 1))
    assert(np.issubdtype(x0.dtype, np.floating))

    # test taking subsequent steps
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0)
    assert(x1.shape == (9, 1))
    assert(np.issubdtype(x1.dtype, np.floating))
    assert(r0.shape == (9,))
    assert(np.issubdtype(r0.dtype, np.floating))

    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1)
    assert(x2.shape == (9, 1))
    assert(np.issubdtype(x2.dtype, np.floating))
    assert(r1.shape == (9,))
    assert(np.issubdtype(r1.dtype, np.floating))

    # test restarting the environment
    x0, _ = env.reset(z=zs)
    assert(x0.shape == (9, 1))
    assert(np.issubdtype(x0.dtype, np.floating))

    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0)
    assert(x1.shape == (9, 1))
    assert(np.issubdtype(x1.dtype, np.floating))
    assert(r0.shape == (9,))
    assert(np.issubdtype(r0.dtype, np.floating))

    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1)
    assert(x2.shape == (9, 1))
    assert(np.issubdtype(x2.dtype, np.floating))
    assert(r1.shape == (9,))
    assert(np.issubdtype(r1.dtype, np.floating))

def test_simulated_env_multivariate_zs_states_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs, states, actions, rewards = sample_trajectory(env=env_true, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs, states=states, actions=actions, rewards=rewards)
    
    # test taking the first step
    x0, _ = env.reset(z=zs)
    assert(x0.shape == (8, 3))
    assert(np.issubdtype(x0.dtype, np.floating))

    # test taking subsequent steps
    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0)
    assert(x1.shape == (8, 3))
    assert(np.issubdtype(x1.dtype, np.floating))
    assert(r0.shape == (8,))
    assert(np.issubdtype(r0.dtype, np.floating))

    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1)
    assert(x2.shape == (8, 3))
    assert(np.issubdtype(x2.dtype, np.floating))
    assert(r1.shape == (8,))
    assert(np.issubdtype(r1.dtype, np.floating))

    # test restarting the environment
    x0, _ = env.reset(z=zs)
    assert(x0.shape == (8, 3))
    assert(np.issubdtype(x0.dtype, np.floating))

    a0 = agent.act(z=zs, xt=x0)
    x1, r0, _, _ = env.step(action=a0)
    assert(x1.shape == (8, 3))
    assert(np.issubdtype(x1.dtype, np.floating))
    assert(r0.shape == (8,))
    assert(np.issubdtype(r0.dtype, np.floating))

    a1 = agent.act(z=zs, xt=x1)
    x2, r1, _, _ = env.step(action=a1)
    assert(x2.shape == (8, 3))
    assert(np.issubdtype(x2.dtype, np.floating))
    assert(r1.shape == (8,))
    assert(np.issubdtype(r1.dtype, np.floating))

def test_sample_simulated_trajectory_univariate_zs_states_default_errors_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=1, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test sample_simulated_env_trajectory()
    zs_test = np.array([[0], [0], [1], [1], [2], [2]])
    zs, states, actions, rewards = sample_simulated_env_trajectory(env=env, 
                                                                   zs=zs_test, 
                                                                   state_dim=1, 
                                                                   T=10, 
                                                                   policy=agent) 
    assert(np.array_equal(zs, zs_test))
    assert(states.shape == (6, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (6, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (6, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

def test_sample_simulated_trajectory_multivariate_zs_states_default_errors_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=3, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test sample_simulated_env_trajectory()
    zs_test = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [1, 1]])
    zs, states, actions, rewards = sample_simulated_env_trajectory(env=env, 
                                                                   zs=zs_test, 
                                                                   state_dim=3, 
                                                                   T=10, 
                                                                   policy=agent) 
    assert(np.array_equal(zs, zs_test))
    assert(states.shape == (6, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (6, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (6, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

def test_sample_simulated_trajectory_univariate_zs_states_custom_errors_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=1, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    global f_counter_states, f_counter_rewards
    f_counter_states = 0
    f_counter_rewards = 0
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test sample_simulated_env_trajectory()
    zs_test = np.array([[0], [0], [1], [1], [2], [2]])
    zs, states, actions, rewards = sample_simulated_env_trajectory(env=env, 
                                                                   zs=zs_test, 
                                                                   state_dim=1, 
                                                                   T=10, 
                                                                   policy=agent, 
                                                                   f_errors_states=f_errors_states, 
                                                                   f_errors_rewards=f_errors_rewards) 
    assert(f_counter_states == 11)
    assert(f_counter_rewards == 10)
    assert(np.array_equal(zs, zs_test))
    assert(states.shape == (6, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (6, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (6, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

def test_sample_simulated_trajectory_multivariate_zs_states_custom_errors_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=3, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test sample_simulated_env_trajectory()
    global f_counter_states, f_counter_rewards
    f_counter_states = 0
    f_counter_rewards = 0
    zs_test = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [1, 1]])
    zs, states, actions, rewards = sample_simulated_env_trajectory(env=env, 
                                                                   zs=zs_test, 
                                                                   state_dim=3, 
                                                                   T=10, 
                                                                   policy=agent, 
                                                                   f_errors_states=f_errors_states, 
                                                                   f_errors_rewards=f_errors_rewards) 
    assert(f_counter_states == 11)
    assert(f_counter_rewards == 10)
    assert(np.array_equal(zs, zs_test))
    assert(states.shape == (6, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (6, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (6, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

def test_estimate_counterfactual_trajectory_univariate_zs_states_default_ua_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=1, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test estimate_counterfactual_trajectories_from_data()
    zs_test = np.array([[0], [0], [1], [1], [2], [2]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env_true, 
                                                                         zs=zs_test, 
                                                                         state_dim=1, 
                                                                         T=10, 
                                                                         policy=agent)
    trajectories = estimate_counterfactual_trajectories_from_data(env=env, 
                                                                  zs=zs_test, 
                                                                  states=states_test, 
                                                                  actions=actions_test, 
                                                                  policy=agent)
    for z in [0, 1, 2]:
        zs = trajectories[tuple([z])]['Z']
        states = trajectories[tuple([z])]['X']
        actions = trajectories[tuple([z])]['A']
        rewards = trajectories[tuple([z])]['R']
        zs_correct = np.repeat(np.array([z]), repeats=6).reshape(-1, 1)
        assert(np.array_equal(zs, zs_correct))
        assert(states.shape == (6, 11, 1))
        assert(np.issubdtype(states.dtype, np.floating))
        assert(actions.shape == (6, 11))
        assert(np.issubdtype(actions.dtype, np.integer))
        assert(rewards.shape == (6, 10))
        assert(np.issubdtype(rewards.dtype, np.floating))

def test_estimated_counterfactual_trajectory_multivariate_zs_states_default_ua_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=3, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test estimate_counterfactual_trajectories_from_data()
    zs_test = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [1, 1]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env_true, 
                                                                         zs=zs_test, 
                                                                         state_dim=3, 
                                                                         T=10, 
                                                                         policy=agent)
    trajectories = estimate_counterfactual_trajectories_from_data(env=env, 
                                                                  zs=zs_test, 
                                                                  states=states_test, 
                                                                  actions=actions_test, 
                                                                  policy=agent)
    for z in [[0, 1], [1, 0], [0, 0], [1, 1]]:
        zs = trajectories[tuple(z)]['Z']
        states = trajectories[tuple(z)]['X']
        actions = trajectories[tuple(z)]['A']
        rewards = trajectories[tuple(z)]['R']
        zs_correct = np.repeat(np.array([z]), repeats=6, axis=0)
        assert(np.array_equal(zs, zs_correct))
        assert(states.shape == (6, 11, 3))
        assert(np.issubdtype(states.dtype, np.floating))
        assert(actions.shape == (6, 11))
        assert(np.issubdtype(actions.dtype, np.integer))
        assert(rewards.shape == (6, 10))
        assert(np.issubdtype(rewards.dtype, np.floating))

def test_estimate_counterfactual_trajectory_univariate_zs_states_custom_ua_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=1, 
                                    z_coef=1, 
                                    f_x0=f_x0_uni, 
                                    f_xt=f_xt_uni, 
                                    f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=1, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test estimate_counterfactual_trajectories_from_data()
    global f_counter_actions
    f_counter_actions = 0
    zs_test = np.array([[0], [0], [1], [1], [2], [2]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env_true, 
                                                                         zs=zs_test, 
                                                                         state_dim=1, 
                                                                         T=10, 
                                                                         policy=agent)
    trajectories = estimate_counterfactual_trajectories_from_data(env=env, 
                                                                  zs=zs_test, 
                                                                  states=states_test, 
                                                                  actions=actions_test, 
                                                                  policy=agent, 
                                                                  f_ua=f_ua)
    assert(f_counter_actions == 11)
    for z in [0, 1, 2]:
        zs = trajectories[tuple([z])]['Z']
        states = trajectories[tuple([z])]['X']
        actions = trajectories[tuple([z])]['A']
        rewards = trajectories[tuple([z])]['R']
        zs_correct = np.repeat(np.array([z]), repeats=6).reshape(-1, 1)
        assert(np.array_equal(zs, zs_correct))
        assert(states.shape == (6, 11, 1))
        assert(np.issubdtype(states.dtype, np.floating))
        assert(actions.shape == (6, 11))
        assert(np.issubdtype(actions.dtype, np.integer))
        assert(rewards.shape == (6, 10))
        assert(np.issubdtype(rewards.dtype, np.floating))

def test_estimated_counterfactual_trajectory_multivariate_zs_states_custom_ua_lm():
    # generate trajectories from the true underlying environment
    env_true = SyntheticEnvironment(state_dim=3, 
                                    z_coef=1, 
                                    f_x0=f_x0_multi, 
                                    f_xt=f_xt_multi, 
                                    f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)
    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env_true, 
                                                                 zs=zs_in, 
                                                                 state_dim=3, 
                                                                 T=10, 
                                                                 policy=agent)
    
    # initialize and train the simulated environment
    env = SimulatedEnvironment(num_actions=2, epochs=10, state_model_type='lm', reward_model_type='lm')
    env.fit(zs=zs_in, states=states_in, actions=actions_in, rewards=rewards_in)

    # test estimate_counterfactual_trajectories_from_data()
    global f_counter_actions
    f_counter_actions = 0
    zs_test = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [1, 1]])
    zs_test, states_test, actions_test, rewards_test = sample_trajectory(env=env_true, 
                                                                         zs=zs_test, 
                                                                         state_dim=3, 
                                                                         T=10, 
                                                                         policy=agent)
    trajectories = estimate_counterfactual_trajectories_from_data(env=env, 
                                                                  zs=zs_test, 
                                                                  states=states_test, 
                                                                  actions=actions_test, 
                                                                  policy=agent, 
                                                                  f_ua=f_ua)
    assert(f_counter_actions == 11)
    for z in [[0, 1], [1, 0], [0, 0], [1, 1]]:
        zs = trajectories[tuple(z)]['Z']
        states = trajectories[tuple(z)]['X']
        actions = trajectories[tuple(z)]['A']
        rewards = trajectories[tuple(z)]['R']
        zs_correct = np.repeat(np.array([z]), repeats=6, axis=0)
        assert(np.array_equal(zs, zs_correct))
        assert(states.shape == (6, 11, 3))
        assert(np.issubdtype(states.dtype, np.floating))
        assert(actions.shape == (6, 11))
        assert(np.issubdtype(actions.dtype, np.integer))
        assert(rewards.shape == (6, 10))
        assert(np.issubdtype(rewards.dtype, np.floating))



# run the experiments
'''test_simulated_env_univariate_zs_states_lm()
test_simulated_env_multivariate_zs_states_lm()
test_sample_simulated_trajectory_univariate_zs_states_default_errors_lm()
test_sample_simulated_trajectory_multivariate_zs_states_default_errors_lm()
test_sample_simulated_trajectory_univariate_zs_states_custom_errors_lm()
test_sample_simulated_trajectory_multivariate_zs_states_custom_errors_lm()
test_estimate_counterfactual_trajectory_univariate_zs_states_default_ua_lm()
test_estimated_counterfactual_trajectory_multivariate_zs_states_default_ua_lm()
test_estimate_counterfactual_trajectory_univariate_zs_states_custom_ua_lm()
test_estimated_counterfactual_trajectory_multivariate_zs_states_custom_ua_lm()
print('All simulated environment tests passed!')'''