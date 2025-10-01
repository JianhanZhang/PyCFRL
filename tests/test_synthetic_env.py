from pycfrl.environment.environment import SyntheticEnvironment
from pycfrl.environment.environment import sample_trajectory, sample_counterfactual_trajectories
import numpy as np
from examples.baseline_agents import RandomAgent

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




def test_synthetic_env_univariate_zs_states():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    
    # test taking the first step
    ux0 = np.random.normal(0, 1, size=[9, 1])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_correct = zs + ux0 + np.repeat(np.array([0.5]), repeats=9).reshape(-1, 1)
    assert(np.allclose(x0, x0_correct, atol=1e-8))

    # test taking subsequent steps
    ux1 = np.random.normal(0, 1, size=[9, 1])
    ur0 = np.random.normal(0, 1, size=[9, 1])
    a0 = np.random.uniform(0, 1, size=[9])
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_correct = zs + 3 * x0 + 2 * a0.reshape(-1, 1) + ux1 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r0_correct = zs + x0 + a0.reshape(-1, 1) + ur0 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r0_correct = r0_correct.flatten()
    assert(np.allclose(x1, x1_correct, atol=1e-8))
    assert(np.allclose(r0, r0_correct, atol=1e-8))

    ux2 = np.random.normal(0, 1, size=[9, 1])
    ur1 = np.random.normal(0, 1, size=[9, 1])
    a1 = np.random.uniform(0, 1, size=[9])
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_correct = zs + 3 * x1 + 2 * a1.reshape(-1, 1) + ux2 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r1_correct = zs + x1 + a1.reshape(-1, 1) + ur1 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r1_correct = r1_correct.flatten()
    assert(np.allclose(x2, x2_correct, atol=1e-8))
    assert(np.allclose(r1, r1_correct, atol=1e-8))

    # test restarting the environment
    ux0 = np.random.normal(0, 1, size=[9, 1])
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_correct = zs + ux0 + np.repeat(np.array([0.5]), repeats=9).reshape(-1, 1)
    assert(np.allclose(x0, x0_correct, atol=1e-8))

    ux1 = np.random.normal(0, 1, size=[9, 1])
    ur0 = np.random.normal(0, 1, size=[9, 1])
    a0 = np.random.uniform(0, 1, size=[9])
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_correct = zs + 3 * x0 + 2 * a0.reshape(-1, 1) + ux1 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r0_correct = zs + x0 + a0.reshape(-1, 1) + ur0 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r0_correct = r0_correct.flatten()
    assert(np.allclose(x1, x1_correct, atol=1e-8))
    assert(np.allclose(r0, r0_correct, atol=1e-8))

    ux2 = np.random.normal(0, 1, size=[9, 1])
    ur1 = np.random.normal(0, 1, size=[9, 1])
    a1 = np.random.uniform(0, 1, size=[9])
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_correct = zs + 3 * x1 + 2 * a1.reshape(-1, 1) + ux2 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r1_correct = zs + x1 + a1.reshape(-1, 1) + ur1 + np.repeat(np.array([-0.5]), repeats=9).reshape(-1, 1)
    r1_correct = r1_correct.flatten()
    assert(np.allclose(x2, x2_correct, atol=1e-8))
    assert(np.allclose(r1, r1_correct, atol=1e-8))

def test_synthetic_env_multivariate_zs_states():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    
    # test taking the first step
    ux0 = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=8)
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + ux0 + np.repeat(np.array([0.5]), repeats=8).reshape(-1, 1)
    assert(np.allclose(x0, x0_correct, atol=1e-8))

    # test taking subsequent steps
    ux1 = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=8)
    ur0 = np.random.normal(0, 1, size=[8, 1])
    a0 = np.random.uniform(0, 1, size=[8])
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + 3 * np.sum(x0, axis=1).reshape(-1, 1) + 2 * a0.reshape(-1, 1) + ux1 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r0_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + np.sum(x0, axis=1).reshape(-1, 1) + a0.reshape(-1, 1) + ur0 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r0_correct = r0_correct.flatten()
    assert(np.allclose(x1, x1_correct, atol=1e-8))
    assert(np.allclose(r0, r0_correct, atol=1e-8))

    ux2 = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=8)
    ur1 = np.random.normal(0, 1, size=[8, 1])
    a1 = np.random.uniform(0, 1, size=[8])
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + 3 * np.sum(x1, axis=1).reshape(-1, 1) + 2 * a1.reshape(-1, 1) + ux2 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r1_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + np.sum(x1, axis=1).reshape(-1, 1) + a1.reshape(-1, 1) + ur1 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r1_correct = r1_correct.flatten()
    assert(np.allclose(x2, x2_correct, atol=1e-8))
    assert(np.allclose(r1, r1_correct, atol=1e-8))

    # test restarting the environment
    ux0 = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=8)
    x0, _ = env.reset(z=zs, ux0=ux0)
    x0_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + ux0 + np.repeat(np.array([0.5]), repeats=8).reshape(-1, 1)
    assert(np.allclose(x0, x0_correct, atol=1e-8))

    ux1 = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=8)
    ur0 = np.random.normal(0, 1, size=[8, 1])
    a0 = np.random.uniform(0, 1, size=[8])
    x1, r0, _, _ = env.step(action=a0, uxt=ux1, urtm1=ur0)
    x1_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + 3 * np.sum(x0, axis=1).reshape(-1, 1) + 2 * a0.reshape(-1, 1) + ux1 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r0_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + np.sum(x0, axis=1).reshape(-1, 1) + a0.reshape(-1, 1) + ur0 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r0_correct = r0_correct.flatten()
    assert(np.allclose(x1, x1_correct, atol=1e-8))
    assert(np.allclose(r0, r0_correct, atol=1e-8))

    ux2 = np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=8)
    ur1 = np.random.normal(0, 1, size=[8, 1])
    a1 = np.random.uniform(0, 1, size=[8])
    x2, r1, _, _ = env.step(action=a1, uxt=ux2, urtm1=ur1)
    x2_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + 3 * np.sum(x1, axis=1).reshape(-1, 1) + 2 * a1.reshape(-1, 1) + ux2 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r1_correct = zs[:, 0].reshape(-1, 1) + zs[:, 1].reshape(-1, 1) + np.sum(x1, axis=1).reshape(-1, 1) + a1.reshape(-1, 1) + ur1 + np.repeat(np.array([-0.5]), repeats=8).reshape(-1, 1)
    r1_correct = r1_correct.flatten()
    assert(np.allclose(x2, x2_correct, atol=1e-8))
    assert(np.allclose(r1, r1_correct, atol=1e-8))

def test_sample_trajectory_univariate_zs_states():
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
    
    assert(np.array_equal(zs, zs_in))
    assert(states.shape == (9, 11, 1))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (9, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (9, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

def test_sample_trajectory_multivariate_zs_states():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)

    def f_ux_multi(N, state_dim):
        return np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=N)

    zs, states, actions, rewards = sample_trajectory(env=env, 
                                                     zs=zs_in, 
                                                     state_dim=3, 
                                                     T=10, 
                                                     policy=agent, 
                                                     f_ux=f_ux_multi)
    
    assert(np.array_equal(zs, zs_in))
    assert(states.shape == (8, 11, 3))
    assert(np.issubdtype(states.dtype, np.floating))
    assert(actions.shape == (8, 10))
    assert(np.issubdtype(actions.dtype, np.integer))
    assert(rewards.shape == (8, 10))
    assert(np.issubdtype(rewards.dtype, np.floating))

def test_sample_counterfactual_trajectory_univariate_zs_states():
    env = SyntheticEnvironment(state_dim=1, 
                               z_coef=1, 
                               f_x0=f_x0_uni, 
                               f_xt=f_xt_uni, 
                               f_rt=f_rt_uni)
    zs_in = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    z_eval_levels = np.array([[0], [1], [2]])
    agent = RandomAgent(2)

    trajectories = sample_counterfactual_trajectories(env=env, 
                                                      zs=zs_in, 
                                                      z_eval_levels=z_eval_levels, 
                                                      state_dim=1, 
                                                      T=10, 
                                                      policy=agent)
    
    for z in [0, 1, 2]:
        zs = trajectories[tuple([z])]['Z']
        states = trajectories[tuple([z])]['X']
        actions = trajectories[tuple([z])]['A']
        rewards = trajectories[tuple([z])]['R']
        zs_correct = np.repeat(np.array([z]), repeats=9).reshape(-1, 1)
        assert(np.array_equal(zs, zs_correct))
        assert(states.shape == (9, 11, 1))
        assert(np.issubdtype(states.dtype, np.floating))
        assert(actions.shape == (9, 10))
        assert(np.issubdtype(actions.dtype, np.integer))
        assert(rewards.shape == (9, 10))
        assert(np.issubdtype(rewards.dtype, np.floating))


def test_sample_counterfactual_trajectory_multivariate_zs_states():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    z_eval_levels = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    agent = RandomAgent(2)

    def f_ux_multi(N, state_dim):
        return np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=N)
    
    trajectories = sample_counterfactual_trajectories(env=env, 
                                                      zs=zs_in, 
                                                      z_eval_levels=z_eval_levels, 
                                                      state_dim=3, 
                                                      T=10, 
                                                      policy=agent, 
                                                      f_ux=f_ux_multi)
    
    for z in [[0, 1], [1, 0], [0, 0], [1, 1]]:
        zs = trajectories[tuple(z)]['Z']
        states = trajectories[tuple(z)]['X']
        actions = trajectories[tuple(z)]['A']
        rewards = trajectories[tuple(z)]['R']
        zs_correct = np.repeat(np.array([z]), repeats=8, axis=0)
        assert(np.array_equal(zs, zs_correct))
        assert(states.shape == (8, 11, 3))
        assert(np.issubdtype(states.dtype, np.floating))
        assert(actions.shape == (8, 10))
        assert(np.issubdtype(actions.dtype, np.integer))
        assert(rewards.shape == (8, 10))
        assert(np.issubdtype(rewards.dtype, np.floating))



# run the tests
test_synthetic_env_univariate_zs_states()
test_synthetic_env_multivariate_zs_states()
test_sample_trajectory_univariate_zs_states()
test_sample_trajectory_multivariate_zs_states()
test_sample_counterfactual_trajectory_univariate_zs_states()
test_sample_counterfactual_trajectory_multivariate_zs_states()
print('All synthetic environment tests passed!')