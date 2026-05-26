import gymnasium as gym
import numpy as np
import copy
import torch
from sklearn.preprocessing import OneHotEncoder
from .utils.base_models import NeuralNetRegressor, LinearRegressor
from .utils.custom_errors import InvalidModelError
from typing import Union, Callable, Literal, Dict
from .agents import Agent

def f_x0_default(
        zs: list | np.ndarray, 
        ux0: list | np.ndarray, 
        z_coef: int | float = 1
    ) -> np.ndarray:
    r"""
    Generate the states at time :math:`t = 0` following the default transition rule.

    Args:
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory. It should be a 2D list or array following the Sensitive 
            Attributes Format.
        ux0 (list or np.ndarray): 
            The exogenous variables (:math:`U_{X_0}`) for each 
            individual in the trajectory. It should be a 2D list or array with shape (N, xdim) 
            where N is the total number of individuals in the trajectory and xdim is the 
            number of components of the state vector.
        z_coef (int or float, optional): 
            The strength of impact of the sensitive attribute on the states  
            and rewards. It is the :math:`\delta` in the specification of the default transition 
            rule.
    
    Returns: 
        x0 (np.ndarray): 
            The states at :math:`t = 0` generated following the default 
            transition rule. It is a 2D array following the Single-time States Format.
    """

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

def f_xt_default(
        zs: list | np.ndarray, 
        xtm1: list | np.ndarray, 
        atm1: list | np.ndarray, 
        uxt: list | np.ndarray, 
        z_coef: int | float = 1
    ) -> np.ndarray:
    r"""
    Generate the states at some time :math:`t > 0` following the default transition rule.

    Args:
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory. It should be a 2D list or array following the Sensitive 
            Attributes Format.
        xtm1 (list or np.ndarray): 
            The states of each individual in the trajectory at time 
            :math:`t - 1`. It should be a 2D list or array following the Single-time 
            States Format.
        atm1 (list or np.ndarray): 
            The actions of each individual in the trajectrory at time 
            :math:`t - 1`. It should be a 1D list or array following the Single-time 
            Actions Format.
        uxt (list or np.ndarray): 
            The exogenous variables (:math:`U_{X_t}`) for each 
            individual in the trajectory. It should be a 2D list or array with shape (N, xdim) 
            where N is the total number of individuals in the trajectory and xdim is the 
            number of components of the state vector.
        z_coef (int or float, optional): 
            The strength of impact of the sensitive attribute on the states  
            and rewards. It is the :math:`\delta` in the specification of the default transition 
            rule.
    
    Returns: 
        xt (np.ndarray): 
            The states at time :math:`t` generated following the default 
            transition rule. It is a 2D array following the Single-time States Format.
    """

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

def f_rt_default(
        zs: list | np.ndarray, 
        xt: list | np.ndarray, 
        at: list | np.ndarray, 
        urtm1: list | np.ndarray, 
        z_coef: int | float = 1
    ) -> np.ndarray:
    r"""
    Generate the rewards at some time :math:`t` following the default transition rule.

    Args:
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory. It should be a 2D list or array following the Sensitive 
            Attributes Format.
        xt (list or np.ndarray): 
            The states of each individual in the trajectory at time 
            :math:`t`. It should be a 2D list or array following the Single-time 
            States Format.
        at (list or np.ndarray): 
            The actions of each individual in the trajectrory at time 
            :math:`t`. It should be a 1D list or array following the Single-time 
            Actions Format.
        urt (list or np.ndarray): 
            The exogenous variables (:math:`U_{R_t}`) for each 
            individual in the trajectory. It should be a 2D list or array with shape (N, 1) 
            where N is the total number of individuals in the trajectory.
        z_coef (int or float, optional): 
            The strength of impact of the sensitive attribute on the states  
            and rewards. It is the :math:`\delta` in the specification of the default transition 
            rule.
    
    Returns: 
        rt (np.ndarray): 
            The rewards at time :math:`t` generated following the default 
            transition rule. It is a 2D array following the Single-time Rewards Format.
    """

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



class SyntheticEnvironment(gym.Env):
    r"""
    Implementation of an environment that makes transitions following a pre-specified rule.

    :code:`SyntheticEnvironment` inherits from :code:`gymnasium.Env` and follows an interface similar 
    to :code:`gymnasium.Env`. Users can also specify the transition rule in the constructor of 
    :code:`SyntheticEnvironment`.

    If no transition rule is specified, :code:`SyntheticEnvironment` will use a set of default 
    transition rules (:code:`f_x0_default`, :code:`f_xt_default`, and :code:`f_rt_default`) that 
    assumes the sensitive attribute vector and the state vector are both univariate. More precisely, 
    the default transition rule is 

    .. math::

        X_0 =& -0.3 + 1.0 \delta Z + U_{X_0} \\
        X_t =& -0.3 + 1.0 \delta (Z - 0.5) + 0.5 X_{t-1} + 0.4 (A_{t-1} - 0.5) \\
            &+ 0.3 X_{t-1} (A_{t-1} - 0.5) + 0.3 \delta X_{t-1} (Z - 0.5) 
            + 0.4 \delta (Z - 0.5) (A_{t-1} - 0.5) + U_{X_t} \\
        R_t =& -0.3 + 0.3 X_t + 0.5 \delta Z + 0.5 A_t + 0.2 \delta X_t Z 
            + 0.7 X_t A_t - 1.0 \delta Z A_t

    Currently, :code:`SyntheticEnvironment` assumes the environment is continuing. 
    """

    def __init__(
            self, 
            state_dim: int, 
            z_coef: int | float = 1, 
            f_x0: Callable[[np.ndarray, np.ndarray, Union[int, float]], np.ndarray] = f_x0_default, 
            f_xt: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[int, float]], 
                           np.ndarray] = f_xt_default, 
            f_rt: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[int, float]], 
                           np.ndarray] = f_rt_default
        ) -> None:
        r"""
        Args:
            state_dim (int): 
                The number of components in the state vector.
            z_coef (int or float, optional): 
                The strength of impact of the sensitive attribute on the states  
                and rewards. It is the :math:`\delta` in the specification of the default transition 
                rule.
            f_x0 (Callable, optional): 
                Transition rule for generating the state at time :math:`t = 0`. It 
                should be a function whose argument list, argument names, and return type exactly 
                match those of :code:`f_x0_default`.
            f_xt (Callable, optional): 
                Transition rule for generating the state at time :math:`t > 0`. It 
                should be a function whose argument list, argument names, and return type exactly 
                match those of :code:`f_xt_default`.
            f_rt (Callable, optional): 
                Transition rule for generating the state at time :math:`t > 0`. It 
                should be a function whose argument list, argument names, and return type exactly 
                match those of :code:`f_rt_default`.
        """

        self.state_dim = state_dim
        self.z_coef = z_coef
        self.f_x0 = f_x0
        self.f_xt = f_xt
        self.f_rt = f_rt

        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(state_dim)
        self.atm1 = None
        self.rtm1 = 0
    
    '''def get_z_coef(self):
        return self.z_coef
    
    def set_z_coef(self, new_z_coef):
        self.z_coef = new_z_coef'''
    
    # ux0 needs to be 2D
    def reset(
            self, 
            z: list | np.ndarray, 
            ux0: list | np.ndarray
        ) -> tuple[np.ndarray, None]:
        """
        Reset the environment to an initial state. 

        Users must call :code:`reset()` first before calling :code:`step()`.

        Args:
            z (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the trajectory. It should be a 2D list or array following the Sensitive 
                Attributes Format.
            ux0 (list or np.ndarray): 
                The exogenous variables (:math:`U_{X_0}`) for each 
                individual in the trajectory. It should be a 2D list or array with shape (N, xdim) 
                where N is the total number of individuals in the trajectory and xdim is the 
                number of components of the state vector.

        Returns: 
            observation (np.ndarray): 
                The initial states generated following the pre-specified 
                transition rule. It is a 2D array following the Single-time States Format.
            info (:code:`None`): 
                Exists to be compatible with the interface of :code:`gymnasium.Env`. 
                It is always :code:`None`.
        """

        zs = np.array(z)
        ux0 = np.array(ux0)
        self.N = zs.shape[0] # number of samples/individuals
        self.zs = zs.reshape(self.N, -1)
        #np.random.seed(self.seed)

        # generate initial state
        self.xt = self.f_x0(zs=self.zs, ux0=ux0, z_coef=self.z_coef)

        # form the current observation
        observation = self.xt

        return observation, None
    
    # urtm1 needs to be 2D
    def step(
            self, 
            action: list | np.ndarray, 
            uxt: list | np.ndarray, 
            urtm1: list | np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, Literal[False], Literal[False]]:
        """
        Generate the states at some time :math:`t > 0` following the default transition rule.

        Args:
            action (list or np.ndarray): 
                The actions (:math:`A_{t-1}`) of each individual in the trajectrory. It 
                should be a 1D list or array following the Single-time Actions Format.
            uxt (list or np.ndarray): 
                The exogenous variables (:math:`U_{X_t}`) for each 
                individual in the trajectory. It should be a 2D list or array with shape (N, xdim) 
                where N is the total number of individuals in the trajectory and xdim is the 
                number of components of the state vector.
            urtm1 (list or np.ndarray): 
                The exogenous variables (:math:`U_{R_{t-1}}`) for each 
                individual in the trajectory. It should be a 2D list or array with shape (N, 1) 
                where N is the total number of individuals in the trajectory.
        
        Returns: 
            observation (np.ndarray): 
                The states transitioned to following the pre-specified 
                transition rule (:math:`X_t`). It is a 2D array following the Single-time States 
                sFormat.
            reward (np.ndarray): 
                The rewards generated following the pre-specified transition 
                rule (:math:`R_{t-1}`). It is a 1D array following the Single-time Rewards Format.
            terminated (:code:`False`): 
                Whether the environment reaches a terminal state. It is always 
                :code:`False` because :code:`SyntheticEnvironment` assumes the environment is 
                continuing.
            truncated (:code:`False`): 
                Whether some truncation condition is satisfied. It is always 
                :code:`False` because :code:`SyntheticEnvironment` currenly does not support 
                specifying truncation conditions.
        """
        
        action = np.array(action)
        uxt = np.array(uxt)
        urtm1 = np.array(urtm1)
        self.atm1 = action
        #np.random.seed(self.seed) # ALSO NEED SEED HERE OR NOT?

        # generate next state and reward
        self.rtm1 = self.f_rt(zs=self.zs, xt=self.xt, at=self.atm1, urtm1=urtm1, z_coef=self.z_coef)
        self.xt = self.f_xt(zs=self.zs, xtm1=self.xt, atm1=self.atm1, uxt=uxt, z_coef=self.z_coef)

        # form the current observation and compute the reward
        observation = self.xt
        reward = self.rtm1

        # the synthetic enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated



def f_ux_default(N: int, state_dim: int) -> np.ndarray:
    """
    Generate exogenous variables for the states from a standard normal distribution.

    Args: 
        N (int): 
            The total number of individuals for whom the exogenous variables will 
            be generated.
        state_dim (int): 
            The number of components in the state vector.
    
    Returns: 
        ux (np.ndarray): 
            The generated exogenous variables. It is a (N, :code:`state_dim`) array 
            where each entry is sampled from a standard normal distribution.
    """

    return np.random.normal(0, 1, size=[N, state_dim])

def f_ua_default(N: int) -> np.ndarray:
    """
    Generate exogenous variables for the actions from a uniform distribution between 0 and 1.

    Args: 
        N (int): 
            The total number of individuals for whom the exogenous variables will 
            be generated.
    
    Returns: 
        ua (np.ndarray): 
            The generated exogenous variables. It is a (N, 1) array 
            where each entry is sampled from a uniform distribution between 0 and 1.
    """

    return np.random.uniform(0, 1, size=[N])

def f_ur_default(N: int) -> np.ndarray:
    """
    Generate exogenous variables for the rewards from a standard normal distribution.

    Args: 
        N (int): 
            The total number of individuals for whom the exogenous variables will 
            be generated.
    
    Returns: 
        ur (np.ndarray): 
            The generated exogenous variables. It is a (N, 1) array 
            where each entry is sampled from a standard normal distribution.
    """

    return np.random.normal(0, 1, size=[N, 1])

# REQUIRES: zs should be the same as the zs passed to the environment
def sample_trajectory(
        env: SyntheticEnvironment, 
        zs: list | np.ndarray, 
        state_dim: int, 
        T: int, 
        policy: Agent, 
        f_ux: Callable[[int, int], np.ndarray] = f_ux_default, 
        f_ua: Callable[[int], np.ndarray] = f_ua_default, 
        f_ur: Callable[[int], np.ndarray] = f_ur_default, 
        seed: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a trajectory from some synthetic environment.

    Args: 
        env (SyntheticEnvironment): 
            The environment to sample trajectory from.
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory that is going to be sampled. It should be a 2D list 
            or array following the Sensitive Attributes Format. 
        state_dim (int): 
            The number of components in the state vector.
        T (int): 
            The total number of transitions in the trajectory that is to be sampled.
        policy (Agent): 
            The policy used to generate actions for the trajectory.
        f_ux (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            states. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ux_default`.
        f_ua (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            actions. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ua_default`. 
        f_ur (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            rewards. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ur_default`.
    
    Returns: 
        Z (np.ndarray): 
            The observed sensitive attributes of each individual in the sampled 
            trajectory. It is an array following the Sensitive Attributes Format.
        X (np.ndarray): 
            The sampled state trajectory. It is an array following the 
            Full-trajectoriy States Format.
        A (np.ndarray): 
            The sampled action trajectory. It is an array following the 
            Full-trajectory Actions Format.
        R (np.ndarray): 
            The sampled reward trajectory. It is an array following the 
            Full-trajectory Rewards Format.
    """

    zs = np.array(zs)
    N = zs.shape[0]

    # initialize containers to store the trajectory
    Z = zs.reshape(N, -1)
    X = np.zeros([N, T + 1, state_dim], dtype=float)
    A = np.zeros([N, T], dtype=int)
    R = np.zeros([N, T], dtype=float)
    np.random.seed(seed)

    # SHOULD DELETE RELATED COMPONENTS
    policy_c = copy.deepcopy(policy)
    X_c = np.zeros([N, T + 1, state_dim], dtype=float)
    A_c = np.zeros([N, T], dtype=int)
    R_c = np.zeros([N, T], dtype=float)
    L_c = np.zeros([N, T, 2], dtype=float)

    # generate the initial state
    ux0 = f_ux(N=N, state_dim=state_dim)
    #ux0 = np.random.normal(0, sigma, [N, 1])
    #ux0 = np.ones((N, 1))
    X[:, 0], _ = env.reset(z=Z, ux0=ux0)
    
    # take the first step
    ua0 = f_ua(N=N)
    #ua0 = np.random.uniform(0, 1, size=[N])
    #ua0 = np.zeros(N)
    A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
    A_c[:, 0] = policy_c.act(
            z=Z,
            xt=X_c[:, 0],
            #atm=A_c[:, :0],
            #t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
            #is_return_prob=False,
        ) # SHOULD DELETE LATER
    ur0 = f_ur(N=N)
    ux1 = f_ux(N=N, state_dim=state_dim)
    #ur0 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.random.normal(0, sigma, [N, 1])
    X[:, 1], R[:, 0], _, _ = env.step(action=A[:, 0], uxt=ux1, urtm1=ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = f_ua([N])
        #uat = np.random.uniform(0, 1, size=[N])
        #uat = np.ones(N)
        A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
        A_c[:, t] = policy_c.act(
                z=Z,
                xt=X_c[:, t],
                #atm=A_c[:, :t],
                #t=t,
                xtm1=X_c[:, t - 1],
                atm1=A_c[:, t - 1],
                uat=uat,
                #is_return_prob=False,
            ) # SHOULD DELETE LATER
        urtm1 = f_ur(N=N)
        uxt = f_ux(N=N, state_dim=state_dim)
        #urtm1 = np.random.normal(0, sigma, [N, 1])
        #uxt = np.random.normal(0, sigma, [N, 1])
        X[:, t + 1], R[:, t], _, _ = env.step(action=A[:, t], uxt=uxt, urtm1=urtm1)

    # prepare and return the output
    #out = [Z, X, A, R]
    return Z, X, A, R



# z_eval_levels should have shape (N, zdim)
def sample_counterfactual_trajectories(
        env: SyntheticEnvironment, 
        zs: list | np.ndarray, 
        z_eval_levels: list | np.ndarray, 
        state_dim: int, 
        T: int, 
        policy: Agent, 
        f_ux: Callable[[int, int], np.ndarray] = f_ux_default, 
        f_ua: Callable[[int], np.ndarray] = f_ua_default, 
        f_ur: Callable[[int], np.ndarray] = f_ur_default, 
        seed: int = 1
    ) -> dict[tuple[Union[int, float], ...], dict[str, Union[np.ndarray, SyntheticEnvironment, Agent]]]:
    """
    Sample counterfactual trajectories from some synthetic environment.

    To sample counterfactual trajectories, for every individual, the function simulates 
    transitions under each of the senstive attribute values specified in :code:`z_eval_levels` 
    while keeping the exogenous variables the same for all trajectories of the same 
    individual. 

    The counterfactual trajectories generated by a policy can be used to compute the  
    counterfactual fairness metric of the policy.

    Args: 
        env (SyntheticEnvironment): 
            The environment to sample trajectory from.
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory that is going to be sampled. It should be a 2D list 
            or array following the Sensitive Attributes Format. 
        z_eval_levels (list or np.ndarray): 
            The set of values of sensitive attributes 
            under which counterfactual trajectories will be generated. For every 
            individual, the function generates a counterfactual trajectory for each of 
            the sensitive attribute values specified in this array. It should be a 2D 
            array where each row contains exactly one sensitive attribute value.
        state_dim (int): 
            The number of components in the state vector.
        T (int): 
            The total number of transitions in the trajectory that is to be sampled.
        policy (Agent): 
            The policy used to generate actions for the trajectory.
        f_ux (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            states. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ux_default`.
        f_ua (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            actions. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ua_default`. 
        f_ur (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            rewards. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ur_default`.
        seed (int, optional): 
            The random seed used to generate the trajectories.
    
    Returns: 
        trajectories (dict): 
            The sampled counterfactual trajectories. It is a dictionary where 
            the keys are the sensitive attribute values in :code:`z_eval_levels` (the sensitive 
            attribute values are each converted to a tuple in the keys). The value of each key (the 
            key is denoted :code:`z`) is a dictionary with six keys: `"Z"` (value is an array whose 
            elements are all :code:`z`), `"X"` (value is the state trajectory for each individual 
            under :code:`z`, organized in the Full-trajectory States Format), `"A"` (value is an array 
            of action trajectory for each individual under :code:`z`, organized in the Full-trajectory 
            Actions Format), `"R"` (value is an array of reward trajectory for each individual under 
            :code:`z`, organized in the Full-trajectory Rewards Format), "env_z" (value is a copy of 
            :code:`env` used to generate the trajectories under :code:`z`, with coresponding buffer 
            memories), and "policy_z" (value is a copy of :code:`policy` used to generate the 
            trajectories under :code:`z`, with corresponding buffer memories).
    """

    np.random.seed(seed)
    zs = np.array(zs)
    z_eval_levels = np.array(z_eval_levels)
    N = zs.shape[0]

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(np.array([z_level]), repeats=N, axis=0) # RECENTLY CHANGED
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}

    # generate the initial state
    ux0 = f_ux(N=N, state_dim=state_dim)
    #ux0 = np.random.normal(0, sigma, [N, 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        trajectories[tuple(z_level.flatten())]['X'][:, 0], _ = e.reset(z=np.tile(z_level, (N, 1)), ux0=ux0)
    
    # take the first step
    ua0 = f_ua(N=N)
    #ua0 = np.random.uniform(0, 1, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )
    ur0 = f_ur(N=N)
    ux1 = f_ux(N=N, state_dim=state_dim)
    #ur0 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.random.normal(0, sigma, [N, 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, 0] for i in range(N)]) 
        (
            trajectories[tuple(z_level.flatten())]['X'][:, 1], 
            trajectories[tuple(z_level.flatten())]['R'][:, 0], 
            _, 
            _
        ) = e.step(action=actions, uxt=ux1, urtm1=ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = np.random.uniform(0, 1, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
            trajectories[tuple(z_level)]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    atm1=actions,
                    uat=uat,
                )
        urtm1 = f_ur(N=N)
        uxt = f_ux(N=N, state_dim=state_dim)
        #urtm1 = np.random.normal(0, sigma, [N, 1])
        #uxt = np.random.normal(0, sigma, [N, 1])
        for z_level in z_eval_levels:
            e = trajectories[tuple(z_level.flatten())]['env_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t] for i in range(N)]) 
            (
                trajectories[tuple(z_level.flatten())]['X'][:, t + 1], 
                trajectories[tuple(z_level.flatten())]['R'][:, t], 
                _, 
                _
            ) = e.step(action=actions, uxt=uxt, urtm1=urtm1)

    # return the output
    return trajectories



class SimulatedEnvironment(gym.Env):
    """
    Implementation of an environment that simulates the transition dynamics of real environments.
    
    A :code:`SimulatedEnvironment` learns transition dynamics from data and makes transitions following 
    the learned dynamics. :code:`SimulatedEnvironment` inherits from :code:`gymnasium.Env` and follows 
    an interface similar to :code:`gymnasium.Env`.

    Currently, :code:`SyntheticEnvironment` assumes the environment is continuous.
    """

    def __init__(
        self,
        num_actions: int, 
        state_model_type: Literal["lm", "nn"] = "nn",
        state_model_hidden_dims: list[int] = [32, 32],
        reward_model_type: Literal["lm", "nn"] = "nn",
        reward_model_hidden_dims: list[int] = [32, 32], 
        is_action_onehot: bool = True, 
        epochs: int = 1000,
        batch_size: int = 128,
        learning_rate: int | float = 0.001,
        is_loss_monitored: bool = True,
        is_early_stopping: bool = True,
        test_size: int | float = 0.2,
        patience: int = 10,
        min_delta: int | float = 0.01,
        enforce_min_max: bool = False,
    ) -> None:
        """
        Args: 
            num_actions (int): 
                The total number of legit actions. 
            state_model_type (str, optional): 
                The type of the model used for learning the transition 
                dynamics of the states. Can be "lm" (polynomial regression) or "nn" (neural network).
                *Currently, only 'nn' is supported.*
            state_model_hidden_dims (list[int], optional): 
                The hidden dimensions of the neural network  
                for learning the transition dynamics of the states. This argument is not used if 
                :code:`state_model_type="lm"`.
            reward_model_type (str, optional): 
                The type of the model used for learning the transition 
                dynamics of the rewards. Can be "lm" (polynomial regression) or "nn" (neural network).
                *Currently, only 'nn' is supported.*
            reward_model_hidden_dims (list[int], optional): 
                The hidden dimensions of the neural network  
                for learning the transition dynamics of the rewards. This argument is not used if 
                :code:`reward_model_type="lm"`.  
            is_action_onehot (bool, optional): 
                When set to :code:`True`, the actions will be one-hot encoded internally.
            epochs (int, optional): 
                The number of training epochs for the neural networks. Applies to  
                both the network for states and the network for rewards, if applicable. This argument 
                is not used if both :code:`state_model_type` and :code:`reward_model_type` are set to 
                :code:`"lm"`.   
            batch_size (int, optional): 
                The batch size of the neural networks. Applies to both the network 
                for states and the network for rewards, if applicable. This argument is not used if both 
                :code:`state_model_type` and :code:`reward_model_type` are set to :code:`"lm"`. 
            learning_rate (int or float, optional): 
                The learning rate of the neural networks. Applies to 
                both the network for states and the network for rewards, if applicable. This argument is 
                not used if both :code:`state_model_type` and :code:`reward_model_type` are set to 
                :code:`"lm"`.
            is_loss_monitored (bool, optional):
                When set to :code:`True`, will split the training data into a training set and a 
                validation set, and will monitor the validation loss during training. A warning 
                will be raised if the decrease in the validation loss is greater than :code:`min_delta` for at 
                least one of the final :math:`p` epochs during neural network training, where :math:`p` is specified 
                by the argument :code:`patience`. Applies to both the network for states and the network for rewards, 
                if applicable. This argument is not used if both :code:`state_model_type` and :code:`reward_model_type` 
                are :code:`"lm"`.
            is_early_stopping (bool, optional): 
                When set to :code:`True`, will split the training data into a training set and a 
                validation set, and will enforce early stopping based on the validation loss 
                during neural network training. That is, neural network training will stop early 
                if the decrease in the validation loss is no greater than :code:`min_delta` for :math:`p` consecutive training 
                epochs, where :math:`p` is specified by the argument :code:`patience`. Applies to 
                both the network for states and the network for rewards, if applicable. This argument is not used if 
                both :code:`state_model_type` and :code:`reward_model_type` are :code:`"lm"`.
            test_size (int or float, optional): 
                An :code:`int` or :code:`float` between 0 and 1 (inclusive) that 
                specifies the proportion of the full training data that is used as the validation set for loss 
                monitoring and early stopping. Applies to both the network for states and the network for rewards, 
                if applicable. This argument is not used if both :code:`state_model_type` and :code:`reward_model_type` are 
                :code:`"lm"`, or if both :code:`is_loss_monitored` and :code:`is_early_stopping` are :code:`False`.
            patience (int, optional): 
                The number of consequentive epochs with barely-decreasing validation loss that is needed 
                for loss monitoring and early stopping. Applies to both the network for states and the network for rewards, 
                if applicable. This argument is not used if both :code:`state_model_type` and :code:`reward_model_type` 
                are :code:`"lm"`, or if both :code:`is_loss_monitored` and :code:`is_early_stopping` are :code:`False`.
            min_delta (int for float, optional): 
                The maximum amount of decrease in the validation loss for it to be considered 
                barely-decreasing by the loss monitoring and early stopping mechanisms. Applies to 
                both the network for states and the network for rewards, if applicable. This argument is 
                not used if both :code:`state_model_type` and :code:`reward_model_type` are :code:`"lm"`, or if both 
                :code:`is_loss_monitored` and :code:`is_early_stopping` are :code:`False`.
            enforce_min_max (bool, optional): 
                When set to :code:`True`, each component of the output 
                states will be clipped to the maximum and minimum value of the corresponding 
                component in the training data. Similarly, the output rewards will also be clipped 
                to the maximum and minimum value of the rewards in the training data.
        """
        
        self.action_space = np.array([a for a in range(num_actions)]).reshape(-1, 1)
        self.num_actions = num_actions
        #self.reward_multiplication_factor = reward_multiplication_factor
        #self.state_variance_factor = state_variance_factor
        self.state_variance_factor = 1.0
        #self.z_factor = z_factor
        if state_model_type == 'nn':
            self.trans_model_type = state_model_type
        else:
            raise InvalidModelError("Invalid model type. Only 'nn' is currently supported.")
        self.trans_model_hidden_dims = state_model_hidden_dims
        if reward_model_type == 'nn':
            self.reward_model_type = reward_model_type
        else:
            raise InvalidModelError("Invalid model type. Only 'nn' is currently supported.")
        self.reward_model_hidden_dims = reward_model_hidden_dims
        self.is_action_onehot = is_action_onehot
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_loss_monitored = is_loss_monitored
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.patience = patience
        self.min_delta = min_delta
        self.enforce_min_max = enforce_min_max
        self.is_trained = False
    
    @staticmethod
    def standardize(
            x: np.ndarray, 
            mean: np.ndarray | None = None, 
            std: np.ndarray |  None = None
        ) -> np.ndarray:
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(
            x: np.ndarray, 
            mean: np.ndarray, 
            std: np.ndarray
        ) -> np.ndarray:
        return x * std + mean
    
    def encode_a(self, a: np.ndarray) -> np.ndarray:
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def fit(
            self, 
            zs: list | np.ndarray, 
            states: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray
        ) -> None:
        """
        Fit the transition dynamics of the MDP underlying the training data.

        Internally, the :code:`fit()` function fits two separate models, one for the states and the 
        other for the rewards.

        Args:
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the training data. It should be a list or array following the Sensitive 
                Attributes Format.
            states (list or np.ndarray): 
                The state trajectory used for training. It should be 
                a list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory used for training. It should be 
                a list or array following the Full-trajectory Actions Format.
            rewards (list or np.ndarray): 
                The reward trajectory used for training. It should be 
                a list or array following the Full-trajectory Rewards Format.
        """

        z = np.array(zs)
        xt = np.array(states)
        at = np.array(actions)
        rt = np.array(rewards)
        N, T, state_dim = xt.shape
        self.N = N
        self.T = T
        self.state_dim = state_dim

        # learn the initial state model i.e. t = 0
        # prepare data
        training_y = xt[:, 0, :]
        training_x_cat = zs
        training_x = training_x_cat
        self.initial_y_min = np.min(training_y, axis=0)
        self.initial_y_max = np.max(training_y, axis=0)

        self.initial_model_mean = {}
        self.initial_model_var = {}
        for z_ in np.unique(zs, axis=0):
            idx = np.all(training_x == z_, axis=1).flatten()
            training_y_z = training_y[idx]
            self.initial_model_mean[tuple(z_)] = np.mean(training_y_z, axis=0)
            self.initial_model_var[tuple(z_)] = np.var(training_y_z, axis=0)
        # REMARK: MIGHT NEED TO ADD BACK THE Z_FACTOR LINE SOMETIME LATER.
        #model_mean[1] = model_mean[1] + self.z_factor

        # learn the transition and reward models for t > 0
        # prepare input
        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(self.N, T - 1, -1)
            dim_a = actions.shape[-1]
        else:
            actions = at
            dim_a = 1 # we require actions to be 1-dimensional w/out one-hot encoding
        actions = actions.reshape(-1, dim_a)
        next_states = xt[:, 1:T, :].reshape(N * (T - 1), -1)
        states = xt[:, 0 : (T - 1), :].reshape(N * (T - 1), -1)
        sensitives = np.repeat(z[:, np.newaxis, :], repeats=T - 1, axis=1).reshape(
            N * (T - 1), -1
        )
        rewards = rt.reshape(-1, 1)

        # fit the transition model
        #glogger.info("fitting transition model")
        if self.trans_model_type == "lm":
            self.trans_models = LinearRegressor(featurize_method='polynomial', degree=2)
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED; FIT() HAS ALSO BEEN MODIFIED
            X = np.concatenate([states, sensitives, actions.reshape(-1, dim_a)], axis=1)
            y = next_states
            self.trans_models.fit(
                #states,
                #sensitives,
                #actions.reshape(-1, dim_a),
                #next_states,
                X,
                y, 
            )

        elif self.trans_model_type == "nn":
            # distinct models for different sensitive groups
            self.trans_models = {}
            for z_ in np.unique(zs, axis=0):
                idxs = np.all(sensitives == z_, axis=1).flatten()
                states_z_ = states[idxs]
                actions_z_ = actions[idxs]

                next_states_z_ = next_states[idxs]
                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                self.trans_models[tuple(z_)] = NeuralNetRegressor(
                    in_dim=states_z_.shape[1] + dim_a,
                    out_dim=next_states_z_.shape[1],
                    hidden_dims=self.trans_model_hidden_dims,
                )
                X = np.concatenate([states_z_, actions_z_.reshape(-1, dim_a)], axis=1)
                y = next_states_z_
                self.trans_models[tuple(z_)].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    is_loss_monitored = self.is_loss_monitored,
                    is_early_stopping=self.is_early_stopping,
                    test_size=self.test_size,
                    patience=self.patience,
                    min_delta=self.min_delta,
                )
            var_mean = 0 # CHANGED; NOT SURE ABOUT CORRECTNESS
            for z_ in np.unique(zs, axis=0):
                var_mean += self.trans_models[tuple(z_)].var
            var_mean = var_mean / np.unique(zs, axis=0).shape[0]
            for z_ in np.unique(zs, axis=0):
                self.trans_models[tuple(z_)].var = var_mean * self.state_variance_factor
                self.trans_models[tuple(z_)].mse = self.trans_models[tuple(z_)].var

        # reward model
        # input: x_t (weekly_pain_score, weekly_pain_interference), z (baseline characteristic), a_t
        # output: r_t
        #glogger.info("fitting reward model")
        if self.reward_model_type == "lm":
            self.reward_models = LinearRegressor(featurize_method='polynomial', degree=2)
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED; FIT() HAS ALSO BEEN MODIFIED
            X = np.concatenate([states, sensitives, actions.reshape(-1, dim_a)], axis=1)
            y = rewards.reshape(-1, 1)
            self.reward_models.fit(
                #states,
                #sensitives,
                #actions.reshape(-1, dim_a),
                #rewards,
                X, 
                y, 
            )
        elif self.reward_model_type == "nn":
            self.reward_models = {}
            for z_ in np.unique(zs, axis=0):
                idxs = np.all(sensitives == z_, axis=1).flatten()
                states_z_ = states[idxs]
                actions_z_ = actions[idxs]
                rewards_z_ = rewards[idxs]
                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                self.reward_models[tuple(z_)] = NeuralNetRegressor(
                    in_dim=states_z_.shape[1] + dim_a,
                    out_dim=1,
                    hidden_dims=self.reward_model_hidden_dims,
                )
                X = np.concatenate([states_z_, actions_z_.reshape(-1, dim_a)], axis=1)
                y = rewards_z_.reshape(-1, 1)
                self.reward_models[tuple(z_)].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    is_loss_monitored = self.is_loss_monitored,
                    is_early_stopping=self.is_early_stopping,
                    test_size=self.test_size,
                    patience=self.patience,
                    min_delta=self.min_delta,
                )
            var_mean = 0 # CHANGED; NOT SURE ABOUT CORRECTNESS
            for z_ in np.unique(zs, axis=0):
                var_mean += self.reward_models[tuple(z_)].var
            var_mean = var_mean / np.unique(zs, axis=0).shape[0]
            for z_ in np.unique(zs, axis=0):
                self.reward_models[tuple(z_)].var = var_mean
                self.reward_models[tuple(z_)].mse = self.trans_models[tuple(z_)].var

        # force min and max for states and rewards
        if self.enforce_min_max:
            self.states_max = np.max(states, axis=0)
            self.states_min = np.min(states, axis=0)
            self.rewards_max = np.max(rewards)
            self.rewards_min = np.min(rewards)
            self.next_states_max = np.max(next_states, axis=0)
            self.next_states_min = np.min(next_states, axis=0)
        
        self.is_trained = True
        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(xt.shape[-1])
        self.atm1 = None
        self.rtm1 = 0
    
    def reset(
        self, 
        z: list | np.ndarray, 
        errors_states: np.ndarray | None = None, 
        #enforce_min_max: bool = False, 
        seed: int = 1
        #z_factor=0.0,
    ) -> tuple[np.ndarray, None]:
        """
        Reset the environment to an initial state. 

        Users must call :code:`reset()` first before calling :code:`step()`.

        Args:
            z (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the trajectory. It should be a 2D list or array following the Sensitive 
                Attributes Format.
            errors_states (np.ndarray): 
                The exogenous variables for states :math:`U_{X_0}` 
                for each individual in the trajectory. It should be a 2D list or array with 
                shape (N, xdim) where N is the total number of individuals in the trajectory 
                and xdim is the number of components of the state vector. When set to :code:`None`, 
                the function will generate the exogenous variables following a multivariate 
                standard normal distribution with xdim mutually independent components.
            seed (int, optional): 
                The random seed used for the transition.

        Returns: 
            observation (np.ndarray): 
                The initial states generated following the learned 
                transition dynamics. It is a 2D array following the Single-time States Format.
            info (:code:`None`): 
                Exists to be compatible with the interface of :code:`gymnasium.Env`. 
                It is always :code:`None`.
        """

        zs = np.array(z)
        if errors_states is not None:
            errors_states = np.array(errors_states)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        testing_x_cat = zs
        testing_x = testing_x_cat
        self.zs = zs

        testing_y = np.zeros([testing_x.shape[0], self.state_dim])
        if errors_states is None:
            errors_states = np.random.multivariate_normal(
                mean=np.zeros(self.state_dim), 
                cov=np.diag(np.ones(self.state_dim)), 
                size=testing_x.shape[0]
            )
        for z_ in np.unique(zs, axis=0):
            idx = np.all(testing_x == z_, axis=1).flatten()
            testing_y[idx] = (
                np.repeat(self.initial_model_mean[tuple(z_)][np.newaxis, :], np.sum(idx), axis=0) + errors_states[idx]
            )

        # cliping
        if self.enforce_min_max:
            for i in range(self.state_dim):
                testing_y[:, :, i] = np.clip(
                    testing_y[:, :, i], a_min=self.initial_y_min[i], a_max=self.initial_y_max[i]
                )
        
        self.xt = testing_y
        observation = self.xt

        return observation, None
    
    # helper function
    def _next_state_reward_mean(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            at: list | np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        zs = np.array(z)
        xt = np.array(xt)
        at = np.array(at)
        N = xt.shape[0]
        states = xt.reshape(N, -1)
        actions = at.flatten()
        next_states = np.zeros([N, xt.shape[-1]])
        rewards = np.zeros([N])

        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(N, -1)
            dim_a = actions.shape[-1]
        else:
            dim_a = 1

        # next state
        if self.trans_model_type == "lm":
            #next_states = self.trans_models.predict(states, sensitives, actions)
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED
            X = np.concatenate(
                    [states, zs, actions.reshape(-1, dim_a)], axis=1
                )
            next_states = self.trans_models.predict(X)
            #next_states = self.trans_models.predict(states, zs, actions)

        elif self.trans_model_type == "nn":
            for z_ in np.unique(zs, axis=0):
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                X = np.concatenate(
                    [states_z_, actions_z_.reshape(-1, dim_a)], axis=1
                )

                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                next_states_z_mean = self.trans_models[tuple(z_)].predict(X)
                # MAY NEED TO UNCOMMENT BACK
                #if z_ == 1:
                #if z_ == np.array([1]):
                #    next_states_z_mean = next_states_z_mean + self.z_factor
                next_states_z_ = next_states_z_mean
                next_states[idx] = next_states_z_

        # reward
        if self.reward_model_type == "lm":
            #rewards = self.reward_models.predict(states, sensitives, actions).flatten()
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED
            X = np.concatenate(
                    [states, zs, actions.reshape(-1, dim_a)], axis=1
                )
            rewards = self.reward_models.predict(X).flatten()
            #rewards = self.reward_models.predict(states, zs, actions).flatten()
        else:
            for z_ in np.unique(zs, axis=0):
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                X = np.concatenate(
                    [states_z_, actions_z_.reshape(-1, dim_a)], axis=1
                )
                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                rewards_z_ = self.reward_models[tuple(z_)].predict(X).flatten()
                rewards[idx] = rewards_z_

        if self.enforce_min_max:
            next_states = np.clip(
                next_states, a_min=self.next_states_min, a_max=self.next_states_max
            )
            rewards = np.clip(rewards, a_min=self.rewards_min, a_max=self.rewards_max)

        return next_states, rewards
    
    def step(
            self, 
            action: list | np.ndarray, 
            errors_states: np.ndarray | None = None, 
            errors_rewards: np.ndarray | None = None, 
            seed: int = 1
        ) -> tuple[np.ndarray, np.ndarray, Literal[False], Literal[False]]:
        """
        Generate the states at some time :math:`t > 0` following the default transition rule.

        Args:
            action (list or np.ndarray): 
                The actions of each individual in the trajectrory. It 
                should be a 1D list or array following the Full-trajectory Actions Format.
            errors_states (list or np.ndarray): 
                The exogenous variables for states (:math:`U_{X_t}`) 
                for each individual in the trajectory. It should be a 2D list or array with shape 
                (N, xdim) where N is the total number of individuals in the trajectory and xdim is 
                the number of components of the state vector. When set to :code:`None`, the function 
                will generate the exogenous variables following a multivariate standard normal 
                distribution with xdim mutually independent components.
            errors_rewards (list or np.ndarray): 
                The exogenous variables for rewards (:math:`U_{R_{t-1}}`) 
                for each individual in the trajectory. It should be a 2D list or array with shape 
                (N, 1) where N is the total number of individuals in the trajectory. When set to 
                :code:`None`, the function will generate the exogenous variables following a standard 
                normal distribution.
            seed (int, optional): 
                The random seed used for the transition.
        
        Returns: 
            observation (np.ndarray): 
                The states transitioned to following the pre-specified 
                transition rule. It is a 2D array following the Single-time States Format.
            reward (np.ndarray): 
                The rewards generated following the pre-specified transition 
                rule. It is a 1D array following the Single-time Rewards Format.
            terminated (:code:`False`): 
                Whether the environment reaches a terminal state. It is always 
                :code:`False` because :code:`SimulatedEnvironment` assumes the environment is 
                continuing.
            truncated (:code:`False`): 
                Whether some truncation condition is satisfied. It is always 
                :code:`False` because :code:`SimulatedEnvironment` currenly does not support 
                specifying truncation conditions.
        """

        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        at = np.array(action)
        xt = self.xt
        next_states_mean, rewards_mean = self._next_state_reward_mean(
            z=self.zs, xt=xt, at=at
        )

        if errors_states is not None:
            errors_states = np.array(errors_states)
        if errors_rewards is not None:
            errors_rewards = np.array(errors_rewards)
        
        if errors_states is None:
            errors_states = np.random.multivariate_normal(
                mean=np.zeros(xt.shape[-1]),
                cov=np.diag(np.ones(xt.shape[-1])), 
                size=xt.shape[0],
            )
        if errors_rewards is None:
            errors_rewards = np.random.normal(
                loc=0, scale=1, size=xt.shape[0]
            )

        next_states = next_states_mean + errors_states
        rewards = rewards_mean + errors_rewards

        self.atm1 = action
        self.xt = next_states
        self.rtm1 = rewards

        # form the current observation and compute the reward
        observation = self.xt
        reward = self.rtm1

        # the simulated enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated
    


def f_errors_states_default(N: int, state_dim: int) -> np.ndarray:
    """
    Generate exogenous variables for the states. 

    The exogenous variables are generated from a standard multivariate normal distribution with 
    mutually independent components.

    Args: 
        N (int): 
            The total number of individuals for whom the exogenous variables will 
            be generated.
        state_dim (int): 
            The number of components in the state vector.
    
    Returns: 
        errors_states (np.ndarray): 
            The generated exogenous variables. It is a (N, state_dim) 
            array where each entry is sampled from a standard multivariate normal distribution 
            with mutually independent components.
    """

    return np.random.multivariate_normal(
                mean=np.zeros(state_dim),
                cov=np.diag(np.ones(state_dim)), 
                size=N,
            )

def f_errors_rewards_default(N: int) -> np.ndarray:
    """
    Generate exogenous variables for the rewards from a standard normal distribution.

    Args: 
        N (int): 
            The total number of individuals for whom the exogenous variables will 
            be generated.
    
    Returns: 
        ua (np.ndarray): 
            The generated exogenous variables. It is a (N, 1) array 
            where each entry is sampled from a standard normal distribution.
    """

    return np.random.normal(
                loc=0, scale=1, size=N
            )

# REQUIRES: zs should be the same as the zs passed to the environment
def sample_simulated_env_trajectory(
        env: SimulatedEnvironment, 
        zs: list | np.ndarray, 
        state_dim: int, 
        T: int, 
        policy: Agent, 
        f_ua: Callable[[int], np.ndarray] = f_ua_default, 
        f_errors_states: Callable[[int, int], np.ndarray] = f_errors_states_default, 
        f_errors_rewards: Callable[[int], np.ndarray] = f_errors_rewards_default, 
        seed: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a trajectory from some simulated environment.

    Args: 
        env (SimulatedEnvironment): 
            The environment to sample trajectory from.
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory that is going to be sampled. It should be a 2D list 
            or array following the Sensitive Attributes Format. 
        state_dim (int): 
            The number of components in the state vector.
        T (int): 
            The total number of transitions in the trajectory that is to be sampled.
        policy (Agent): 
            The policy used to generate actions for the trajectory.
        f_ua (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            actions. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ua_default`. 
        f_errors_states (Callable, optional): 
            A rule to generate exogenous variables for each 
            individual's states. It should be a function whose argument list, argument names, 
            and return type exactly match those of :code:`f_errors_states_default`.
        f_errors_rewards (Callable, optional): 
            A rule to generate exogenous variables for each 
            individual's rewards. It should be a function whose argument list, argument names, 
            and return type exactly match those of :code:`f_errors_rewards_default`.
        seed (int, optional): 
            The random seed used to sample the trajectory. 
    
    Returns: 
        Z (np.ndarray): 
            The observed sensitive attributes of each individual in the sampled 
            trajectory. It is an array following the Sensitive Attributes Format.
        X (np.ndarray): 
            The sampled state trajectory. It is an array following the 
            Full-trajectoriy States Format.
        A (np.ndarray): 
            The sampled action trajectory. It is an array following the 
            Full-trajectory Actions Format.
        R (np.ndarray): 
            The sampled reward trajectory. It is an array following the 
            Full-trajectory Rewards Format.
    """
    
    # initialize containers to store the trajectory
    zs = np.array(zs)
    N = zs.shape[0]
    Z = zs.reshape(N, -1)
    X = np.zeros([N, T + 1, state_dim], dtype=float)
    A = np.zeros([N, T], dtype=int)
    R = np.zeros([N, T], dtype=float)
    np.random.seed(seed)

    '''# SHOULD DELETE RELATED COMPONENTS
    policy_c = copy.deepcopy(policy)
    X_c = np.zeros([N, T + 1, state_dim], dtype=float)
    A_c = np.zeros([N, T], dtype=int)
    R_c = np.zeros([N, T], dtype=float)
    L_c = np.zeros([N, T, 2], dtype=float)'''

    # generate the initial state
    errors_states = f_errors_states(N=N, state_dim=state_dim)
    X[:, 0], _ = env.reset(z=Z, errors_states=errors_states)
    
    # take the first step
    ua0 = f_ua(N=N)
    A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
    '''A_c[:, 0] = policy_c.act(
            z=Z,
            xt=X_c[:, 0],
            #atm=A_c[:, :0],
            #t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
            is_return_prob=False,
        ) # SHOULD DELETE LATER'''
    errors_states = f_errors_states(N=N, state_dim=state_dim)
    errors_rewards = f_errors_rewards(N=N)
    X[:, 1], R[:, 0], _, _ = env.step(action=A[:, 0], 
                                      errors_states=errors_states, 
                                      errors_rewards=errors_rewards)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = np.ones(N)
        A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
        '''A_c[:, t] = policy_c.act(
                z=Z,
                xt=X_c[:, t],
                #atm=A_c[:, :t],
                #t=t,
                xtm1=X_c[:, t - 1],
                atm1=A_c[:, t - 1],
                uat=uat,
                is_return_prob=False,
            ) # SHOULD DELETE LATER'''
        #urtm1 = np.random.normal(0, sigma, [N, 1])
        #uxt = np.random.normal(0, sigma, [N, 1])
        #uxt = np.zeros((N, 1))
        #urtm1 = np.zeros((N, 1))
        errors_states = f_errors_states(N=N, state_dim=state_dim)
        errors_rewards = f_errors_rewards(N=N)
        X[:, t + 1], R[:, t], _, _ = env.step(action=A[:, t], 
                                              errors_states=errors_states, 
                                              errors_rewards=errors_rewards)

    # prepare and return the output
    #out = [Z, X, A, R]
    return Z, X, A, R



def estimate_counterfactual_trajectories_from_data(
        env: SimulatedEnvironment, 
        zs: list | np.ndarray, 
        states: list | np.ndarray, 
        actions: list | np.ndarray, 
        policy: Agent, 
        f_ua: Callable[[int], np.ndarray] = f_ua_default, 
        seed: int = 1
    ) -> dict[tuple[Union[int, float], ...], dict[str, Union[np.ndarray, SyntheticEnvironment, Agent]]]:
    """
    Reconstruct the counterfactual trajectories from an observed trajectory.

    For each individual in the input trajectory, :code:`estimate_counterfactual_trajectories_from_data()` 
    reconstructs that individual's counterfactual trajectories under different values of the sensitive 
    attribute. The sensitive attribute values used here are those that appear in :code:`zs`. 

    More precisely, the counterfactual states and rewards are first estimated following the data 
    preprocessing method proposed by Wang et al. (2025), which is referenced below. :code:`policy` is 
    then used to generate the counterfactual action trajectories using the estimated counterfactual 
    states.

    References: 
        .. [2] Wang, J., Shi, C., Piette, J.D., Loftus, J.R., Zeng, D. and Wu, Z. (2025). 
               Counterfactually Fair Reinforcement Learning via Sequential Data 
               Preprocessing. arXiv preprint arXiv:2501.06366.

    The counterfactual trajectories estimated using a policy can be used to compute the  
    counterfactual fairness metric of the policy.
    
    Args: 
        env (SimulatedEnvironment): 
            An environment that simulates the transition dynamics of the 
            MDP underlying :code:`zs`, :code:`states`, :code:`actions`, and :code:`rewards`. 
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual in the 
            trajectory used for estimating the counterfactual trajectories. It should be a list 
            or array following the Sensitive Attributes Format. 
        states (list or np.ndarray): 
            The state trajectory used for estimating the counterfactual 
            trajectories. It should be a list or array following the Full-trajectory States Format.
        actions (list or np.ndarray): 
            The action trajectory used for estimating the counterfactual 
            trajectories. It should be a list or array following the Full-trajectory Actions Format.
        policy (Agent): 
            The policy used to estimate the counterfactual action trajectories. 
        f_ua (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            actions. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ua_default`. 
        seed (int, optional): 
            The seed used to estimate the counterfactual trajectories. 
    
    Returns: 
        trajectories (dict): 
            The sampled counterfactual trajectories. It is a dictionary where 
            the keys are the sensitive attribute values in :code:`z_eval_levels` (the sensitive 
            attribute values are each converted to a tuple in the keys). The value of each key (the 
            key is denoted :code:`z`) is a dictionary with six keys: `"Z"` (value is an array whose 
            elements are all :code:`z`), `"X"` (value is the state trajectory for each individual 
            under :code:`z`, organized in the Full-trajectory States Format), `"A"` (value is an array 
            of action trajectory for each individual under :code:`z`, organized in the Full-trajectory 
            Actions Format), `"R"` (value is an array of reward trajectory for each individual under 
            :code:`z`, organized in the Full-trajectory Rewards Format), "env_z" (value is a copy of 
            :code:`env` used to generate the trajectories under :code:`z`, with coresponding buffer 
            memories), and "policy_z" (value is a copy of :code:`policy` used to generate the 
            trajectories under :code:`z`, with corresponding buffer memories).
    """
    
    zs = np.array(zs)
    N = actions.shape[0]
    T = actions.shape[1]
    state_dim = states.shape[2]
    np.random.seed(seed)
    z_eval_levels = np.unique(zs, axis=0)
    z_eval_levels = np.array(z_eval_levels)

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(np.array([z_level]), repeats=N, axis=0)
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T + 1], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}

    # generate the initial state
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        for z_obs in np.unique(zs, axis=0):
            idx = np.all(zs == z_obs, axis=1).flatten()
            (
                trajectories[tuple(z_level.flatten())]['X'][idx, 0]
            ) = (states[idx, 0] 
                 - e.initial_model_mean[tuple(z_obs.flatten())] 
                 + e.initial_model_mean[tuple(z_level.flatten())])
    
    # take the first step
    ua0 = f_ua(N=N)
    #ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        for z_obs in np.unique(zs, axis=0):
            idx = np.all(zs == z_obs, axis=1).flatten()
            obs_mean, _ = e._next_state_reward_mean(
                        z=zs[idx],
                        xt=states[idx, 0, :],
                        at=actions[idx, 0],
                        )
            cf_mean, _ = e._next_state_reward_mean(
                        z=trajectories[tuple(z_level.flatten())]['Z'][idx],
                        xt=trajectories[tuple(z_level.flatten())]['X'][idx, 0, :],
                        at=actions[idx, 0],
                        )
            (
                trajectories[tuple(z_level.flatten())]['X'][idx, 1]
            ) = (states[idx, 1] - obs_mean + cf_mean)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = np.random.uniform(0, sigma_a, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            trajectories[tuple(z_level.flatten())]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    atm1=actions[:, t - 1],
                    uat=uat,
                )
        for z_level in z_eval_levels:
            e = trajectories[tuple(z_level.flatten())]['env_z']
            for z_obs in np.unique(zs, axis=0):
                idx = np.all(zs == z_obs, axis=1).flatten()
                obs_mean, _ = e._next_state_reward_mean(
                        z=zs[idx],
                        xt=states[idx, t, :],
                        at=actions[idx, t],
                        )
                cf_mean, _ = e._next_state_reward_mean(
                        z=trajectories[tuple(z_level.flatten())]['Z'][idx],
                        xt=trajectories[tuple(z_level.flatten())]['X'][idx, t, :],
                        at=actions[idx, t],
                        )
                (
                    trajectories[tuple(z_level.flatten())]['X'][idx, t + 1]
                ) = (states[idx, t + 1] - obs_mean + cf_mean)
    
    # take the final step
    uaT = f_ua(N=N)
    #ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, T] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, T],
                xtm1=trajectories[tuple(z_level.flatten())]['X'][:, T - 1],
                atm1=actions[:, T - 1],
                uat=uaT,
            )
    
    return trajectories