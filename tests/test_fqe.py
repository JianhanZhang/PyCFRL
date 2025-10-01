from pycfrl.fqe.fqe import FQE
from pycfrl.environment.environment import SyntheticEnvironment, sample_trajectory
from examples.baseline_agents import RandomAgent
import numpy as np
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



def test_fqe_univariate_zs_states_nn():
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
    
    f = FQE(model_type='nn', num_actions=2, policy=agent, epochs=10)
    f.fit(zs=zs_in, 
          states=states_in, 
          actions=actions_in, 
          rewards=rewards_in, 
          max_iter=10)
    out = f.evaluate(zs=zs_in, states=states_in, actions=actions_in)
    assert(out.shape == (90,))
    assert(np.issubdtype(out.dtype, np.floating))

def test_fqe_multivariate_zs_states_nn():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)

    def f_ux_multi(N, state_dim):
        return np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=N)

    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env, 
                                                                 zs=zs_in, 
                                                                 state_dim=3, 
                                                                 T=10, 
                                                                 policy=agent, 
                                                                 f_ux=f_ux_multi)
    
    f = FQE(model_type='nn', num_actions=2, policy=agent, epochs=10)
    f.fit(zs=zs_in, 
          states=states_in, 
          actions=actions_in, 
          rewards=rewards_in, 
          max_iter=10)
    out = f.evaluate(zs=zs_in, states=states_in, actions=actions_in)
    assert(out.shape == (80,))
    assert(np.issubdtype(out.dtype, np.floating))



@pytest.mark.skip(reason="Skipped because 'lm' is currently not supported.")
def test_fqe_univariate_zs_states_lm():
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
    
    f = FQE(model_type='lm', num_actions=2, policy=agent, epochs=10)
    f.fit(zs=zs_in, 
          states=states_in, 
          actions=actions_in, 
          rewards=rewards_in, 
          max_iter=10)
    out = f.evaluate(zs=zs_in, states=states_in, actions=actions_in)
    assert(out.shape == (90,))
    assert(np.issubdtype(out.dtype, np.floating))

@pytest.mark.skip(reason="Skipped because 'lm' is currently not supported.")
def test_fqe_multivariate_zs_states_lm():
    env = SyntheticEnvironment(state_dim=3, 
                               z_coef=1, 
                               f_x0=f_x0_multi, 
                               f_xt=f_xt_multi, 
                               f_rt=f_rt_multi)
    zs_in = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
    agent = RandomAgent(2)

    def f_ux_multi(N, state_dim):
        return np.random.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]), size=N)

    zs_in, states_in, actions_in, rewards_in = sample_trajectory(env=env, 
                                                                 zs=zs_in, 
                                                                 state_dim=3, 
                                                                 T=10, 
                                                                 policy=agent, 
                                                                 f_ux=f_ux_multi)
    
    f = FQE(model_type='lm', num_actions=2, policy=agent, epochs=10)
    f.fit(zs=zs_in, 
          states=states_in, 
          actions=actions_in, 
          rewards=rewards_in, 
          max_iter=10)
    out = f.evaluate(zs=zs_in, states=states_in, actions=actions_in)
    assert(out.shape == (80,))
    assert(np.issubdtype(out.dtype, np.floating))



# run the tests
test_fqe_univariate_zs_states_nn()
test_fqe_multivariate_zs_states_nn()
#test_fqe_univariate_zs_states_lm()
#test_fqe_multivariate_zs_states_lm()
print('All fqe tests passed!')
