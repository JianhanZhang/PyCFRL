import numpy as np
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from .environment import sample_trajectory, sample_counterfactual_trajectories
from .environment import SyntheticEnvironment, SimulatedEnvironment
from .environment import estimate_counterfactual_trajectories_from_data
from .fqe import FQE
from .agents import Agent
from typing import Union, Callable, Literal
from .utils.custom_errors import InvalidModelError

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

def evaluate_reward_through_simulation(
        env: SyntheticEnvironment, 
        z_eval_levels: list | np.ndarray, 
        state_dim: int, 
        N: int, 
        T: int, 
        policy: Agent, 
        f_ux: Callable[[int, int], np.ndarray] = f_ux_default, 
        f_ua: Callable[[int], np.ndarray] = f_ua_default, 
        f_ur: Callable[[int], np.ndarray] = f_ur_default, 
        z_probs: list | np.ndarray | None = None, 
        gamma: int | float = 0.9, 
        seed: int = 1
    ) -> np.integer | np.floating: 
    """
    Estimate the value of a policy using simulation in a synthetic environment. 

    The function first simulates a trajectory of a pre-specified length :code:`T` using the policy. 
    Then it computes the cumulative discounted rewards achieved throughout the trajectory. 

    Since the discounted rewards are added across all time steps, it should generally be higher 
    if a larger value is specified for the argument :code:`T`. 

    Args: 
        env (SyntheticEnvironment): 
            The synthetic environment in which the simulation is run.
        z_eval_levels (list or np.ndarray): 
            The values of sensitive attributes used in the simulation. 
            The observed sensitive attributes of the individuals in the simulation will be sampled 
            from this set. 
        state_dim (int): 
            The number of components in the state vector. 
        N (int): 
            The total number of individuals in the trajectory sampled during the simulation.
        T (int): 
            The total number of transitions in the trajectory sampled during the simulation.
        policy (Agent): 
            The policy whose value is to be evaluated. 
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
        z_probs (list or np.ndarray, optional): 
            The probability of an individual taking each of 
            the values in :code:`z_eval_levels`. When set to :code:`None`, a uniform distribution 
            will be used. 
        gamma (int or float, optional): 
            The discount factor used for calculating the discounted 
            cumulative rewards. 
        seed (int, optional): 
            The random seed used to run the simulation.

    Returns: 
        discounted_cumulative_reward (np.integer or np.floating): 
            An estimation of the discounted 
            cumulative reward achieved by the policy throughout the trajectory. 
    """

    z_levels = np.array(z_eval_levels)
    if z_probs is not None:
        z_probs = np.array(z_probs)
    np.random.seed(seed)

    # generate the sensitive attribute for each simulated individual
    #Z = np.random.binomial(1, p=0.5, size=[N, 1])
    Z = np.zeros((N, z_levels.shape[1]))
    if z_probs is None:
        Z_idx = np.random.choice(range(z_levels.shape[0]), size=N, replace=True)
    else:
        Z_idx = np.random.choice(range(z_levels.shape[0]), size=N, p=z_probs, replace=True)
    for i in range(N):
        Z[i] = z_levels[Z_idx[i]]

    # simulate a trajectory and compute the discounted cumulative reward
    _, _, _, rewards = sample_trajectory(env=env, zs=Z, state_dim=state_dim, T=T, 
                                         policy=policy, f_ux=f_ux, f_ua=f_ua, f_ur=f_ur, seed=seed)
    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    
    return discounted_cumulative_reward



# REQUIRES: z_eval_levels must be the one used to generate the counterfactual trajectories
def _compute_cf_metric(
        trajectories: dict[tuple[Union[int, float], ...], dict[str, Union[np.ndarray, SyntheticEnvironment, Agent]]], 
        z_eval_levels: list | np.ndarray
    ) -> np.integer | np.floating:
    z_eval_levels = np.array(z_eval_levels)

    max_cf_metric = 0
    for i in range(len(z_eval_levels)):
        for j in range(i, len(z_eval_levels)):
            ai = trajectories[tuple(z_eval_levels[i].flatten())]['A']
            aj = trajectories[tuple(z_eval_levels[j].flatten())]['A']
            cf_metric = np.mean(np.mean(np.abs(ai != aj), axis=0))
            if cf_metric > max_cf_metric:
                max_cf_metric = cf_metric

    return max_cf_metric



def evaluate_fairness_through_simulation(
        env: SyntheticEnvironment, 
        z_eval_levels: list | np.ndarray, 
        state_dim: int, 
        N: int, 
        T: int, 
        policy: Agent, 
        f_ux: Callable[[int, int], int]=f_ux_default, 
        f_ua: Callable[[int], int]=f_ua_default, 
        f_ur: Callable[[int], int]=f_ur_default, 
        z_probs: list | np.ndarray | None = None, 
        seed: int = 1
    ) -> np.integer | np.floating:
    r"""
    Estimate the counterfactual fairness metric of a policy using simulation in a synthetic environment.

    The function first simulates a set of counterfactual trajectories with a pre-specified length 
    using :code:`sample_counterfactual_trajectories()` in the `environment` module. Then it computes 
    a counterfactual fairness metric using the following formula given in Wang et al. (2025): 

     .. math:: 

        \max_{z', z \in eval(Z)} \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} 
        \mathbb{I} \left( A_t^{Z \leftarrow z'}\left(\bar{U}_t(h_{it})\right) 
        \neq A_t^{Z \leftarrow z}\left(\bar{U}_t(h_{it})\right) \right).
    
    where :math:`eval(Z)` is the set of sensitive attribute values passed in by `z_eval_levels`, 
    :math:`A_t^{Z \leftarrow z'}\left(\bar{U}_t(h_{it})\right)` is the action taken in the 
    counterfactual trajectory under :math:`Z=z'`, and 
    :math:`A_t^{Z \leftarrow z}\left(\bar{U}_t(h_{it})\right)` is the action taken under the 
    counterfactual trajectory under :math:`Z=z`. This metric is bounded between 0 and 1, with 0 
    representing perfect fairness and 1 indicating complete unfairness.

    References: 
        .. [2] Wang, J., Shi, C., Piette, J.D., Loftus, J.R., Zeng, D. and Wu, Z. (2025). 
               Counterfactually Fair Reinforcement Learning via Sequential Data 
               Preprocessing. arXiv preprint arXiv:2501.06366.

    Args: 
        env (SyntheticEnvironment): 
            The synthetic environment in which the simulation is run.
        z_eval_levels (list or np.ndarray): 
            The values of sensitive attributes for which 
            counterfactual trajectories are generated in the simulation. 
            The observed sensitive attributes of the individuals in the simulation will also be 
            sampled from this set. 
        state_dim (int): 
            The number of components in the state vector. 
        N (int): 
            The total number of individuals in the counterfactual trajectories sampled during the 
            simulation.
        T (int): 
            The total number of transitions in the counterfactual trajectories sampled during the 
            simulation.
        policy (Agent): 
            The policy whose fairness is to be evaluated. 
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
        z_probs (list or np.ndarray, optional): 
            The probability of an individual taking each of 
            the values in :code:`z_eval_levels` as the observed sensitive attribute. When set to 
            :code:`None`, a uniform distribution will be used. 
        seed (int, optional): 
            The random seed used to run the simulation.

    Returns: 
        cf_metric (np.integer or np.floating): 
            The counterfactual fairness metric of the policy. 
    """
    
    z_eval_levels = np.array(z_eval_levels)
    if z_probs is not None:
        z_probs = np.array(z_probs)
    np.random.seed(seed)

    # generate the sensitive sttribute for each simulated individual
    zs = np.random.binomial(n=1, p=1/2, size=[N, z_eval_levels.shape[1]])
    zs = np.zeros((N, z_eval_levels.shape[1]))
    if z_probs is None:
        Z_idx = np.random.choice(range(z_eval_levels.shape[0]), size=N, replace=True)
    else:
        Z_idx = np.random.choice(range(z_eval_levels.shape[0]), size=N, p=z_probs, replace=True)
    for i in range(N):
        zs[i] = z_eval_levels[Z_idx[i]]

    # generate the simulated counterfactual trajectories and compute the CF metric
    trajectories = sample_counterfactual_trajectories(env=env, zs=zs, z_eval_levels=z_eval_levels, 
                                                      state_dim=state_dim, T=T, policy=policy, 
                                                      f_ux=f_ux, f_ua=f_ua, f_ur=f_ur, seed=seed)
    cf_metric = _compute_cf_metric(trajectories=trajectories, z_eval_levels=z_eval_levels)

    return cf_metric



def evaluate_fairness_through_model(
        env: SimulatedEnvironment, 
        zs: list | np.ndarray, 
        states: list | np.ndarray, 
        actions: list | np.ndarray, 
        policy: Agent, 
        f_ua: Callable[[int], int] = f_ua_default, 
        seed: int = 1
    ) -> np.integer | np.floating:
    r"""
    Estimate the counterfactual fairness metric of a policy from an offline trajectory.

    The function first estimates a set of counterfactual trajectories from the offline trajectory 
    using :code:`estimate_counterfactual_trajectories_from_data()` in the :code:`environment` module. 
    Then it computes a counterfactual fairness metric using the following formula given in Wang et al. 
    (2025): 

     .. math:: 
        
        \max_{z', z \in eval(Z)} \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} 
        \mathbb{I} \left( A_t^{Z \leftarrow z'}\left(\bar{U}_t(h_{it})\right) 
        \neq A_t^{Z \leftarrow z}\left(\bar{U}_t(h_{it})\right) \right).
    
    where :math:`eval(Z)` is the set of sensitive attribute values passed in by `z_eval_levels`, 
    :math:`A_t^{Z \leftarrow z'}\left(\bar{U}_t(h_{it})\right)` is the action taken in the 
    counterfactual trajectory under :math:`Z=z'`, and 
    :math:`A_t^{Z \leftarrow z}\left(\bar{U}_t(h_{it})\right)` is the action taken under the 
    counterfactual trajectory under :math:`Z=z`. This metric is bounded between 0 and 1, with 0 
    representing perfect fairness and 1 indicating complete unfairness.

    References: 
        .. [2] Wang, J., Shi, C., Piette, J.D., Loftus, J.R., Zeng, D. and Wu, Z. (2025). 
               Counterfactually Fair Reinforcement Learning via Sequential Data 
               Preprocessing. arXiv preprint arXiv:2501.06366.
    
    Args: 
        env (SimulatedEnvironment): 
            An environment that simulates the transition dynamics of the 
            MDP underlying :code:`zs`, :code:`states`, :code:`actions`, and :code:`rewards`. 
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual in the 
            offline trajectory. It should be a list or array following the Sensitive Attributes 
            Format. 
        states (list or np.ndarray): 
            The state trajectory. It should be a list or array following 
            the Full-trajectory States Format.
        actions (list or np.ndarray): 
            The action trajectory. It should be a list or array following 
            the Full-trajectory Actions Format.
        policy (Agent): 
            The policy whose fairness is to be evaluated. 
        f_ua (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            actions. It should be a function whose argument list, argument names, and return 
            type exactly match those of :code:`f_ua_default`. 
        seed (int, optional): 
            The seed used to estimate the counterfactual trajectories. 
    Returns: 
        cf_metric (np.integer or np.floating): 
            The counterfactual fairness metric of the policy. 
    """
    
    zs = np.array(zs)
    z_eval_levels = np.unique(zs, axis=0)
    states = np.array(states)
    actions = np.array(actions)
    np.random.seed(seed)

    # generate the simulated counterfactual trajectories
    trajectories = estimate_counterfactual_trajectories_from_data(env=env, zs=zs, states=states, 
                                                                  actions=actions, policy=policy, 
                                                                  f_ua=f_ua, seed=seed)

    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories=trajectories, z_eval_levels=z_eval_levels)

    return cf_metric



def evaluate_reward_through_fqe(
        zs: list | np.ndarray, 
        states: list | np.ndarray, 
        actions: list | np.ndarray, 
        rewards: list | np.ndarray, 
        policy: Agent, 
        model_type: Literal["lm", "nn"], 
        f_ua: Callable[[int], int] = f_ua_default, 
        hidden_dims: list[int] = [32], 
        learning_rate: int | float = 0.1, 
        epochs: int = 500, 
        gamma: int | float = 0.9, 
        max_iter: int = 200, 
        seed: int = 1, 
        is_loss_monitored: bool = False,
        is_early_stopping_nn: bool = False,
        test_size_nn: int | float = 0.2,
        patience_nn: int = 10,
        min_delta_nn: int | float = 0.005, 
        is_q_monitored: bool = True,
        is_early_stopping_q: bool = True,
        patience_q: int = 5,
        min_delta_q: int | float = 0.005
    ) -> np.integer | np.floating:
    """
    Estimate the value of a policy using fitted Q evaluation (FQE).

    The function takes in a offline trajectory and the policy of interest, which are then used by 
    a FQE algorithm to evaluate the value of the policy.

    Args: 
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual in the 
            offline trajectory. It should be a list or array following the Sensitive Attributes 
            Format. 
        states (list or np.ndarray): 
            The state trajectory. It should be a list or array following 
            the Full-trajectory States Format.
        actions (list or np.ndarray): 
            The action trajectory. It should be a list or array following 
            the Full-trajectory Actions Format.
        rewards (list or np.ndarray): 
            The reward trajectory. It should be a list or array following 
            the Full-trajectory Rewards Format.
        policy (Agent): 
            The policy whose value is to be evaluated.
        model_type (str): 
            The type of the model used for FQE. Can be "lm" (polynomial regression) or 
            "nn" (neural network). *Currently, only 'nn' is supported.*
        f_ua (Callable, optional): 
            A rule to generate exogenous variables for each individual's 
            actions during training. It should be a function whose argument list, argument names, 
            and return type exactly match those of :code:`f_ua_default`. 
        hidden_dims (list[int], optional): 
            The hidden dimensions of the neural network. This 
            argument is not used if :code:`model_type="lm"`. 
        learning_rate (int or float, optional): 
            The learning rate of the neural network. This 
            argument is not used if :code:`model_type="lm"`. 
        epochs (int, optional): 
            The number of training epochs for the neural network. This 
            argument is not used if :code:`model_type="lm"`. 
        gamma (int or float, optional): 
            The discount factor for the cumulative discounted reward 
            in the objective function. 
        max_iter (int, optional): 
            The number of iterations for learning the Q function. 
        seed (int, optional): 
            The random seed used for FQE.
        is_loss_monitored (bool, optional):
                When set to :code:`True`, will split the training data into a training set and a 
                validation set, and will monitor the validation loss when training the neural network 
                approximator of the Q function in each iteration. A warning 
                will be raised if the percent decrease in the validation loss is greater than :code:`min_delta_nn` for at 
                least one of the final :math:`p` epochs during neural network training, where :math:`p` is specified 
                by the argument :code:`patience_nn`. This argument is not used if :code:`model_type="lm"`.
            is_early_stopping_nn (bool, optional): 
                When set to :code:`True`, will split the training data into a training set and a 
                validation set, and will enforce early stopping based on the validation loss 
                when training the neural network approximator of the Q function in each iteration. That is, in each iteration, 
                neural network training will stop early 
                if the percent decrease in the validation loss is no greater than :code:`min_delta_nn` for :math:`p` consecutive training 
                epochs, where :math:`p` is specified by the argument :code:`patience_nn`. This argument is not used if 
                :code:`model_type="lm"`.
            test_size_nn (int or float, optional): 
                An :code:`int` or :code:`float` between 0 and 1 (inclusive) that 
                specifies the proportion of the full training data that is used as the validation set for loss 
                monitoring and early stopping. This argument is not used if :code:`model_type="lm"` or 
                both :code:`is_loss_monitored` and :code:`is_early_stopping` are :code:`False`.
            patience_nn (int, optional): 
                The number of consequentive epochs with barely-decreasing validation loss that is needed 
                for loss monitoring and early stopping. This argument is not used if :code:`model_type="lm"` 
                or both :code:`is_loss_monitored` and :code:`is_early_stopping_nn` are :code:`False`.
            min_delta_nn (int for float, optional): 
                The maximum amount of decrease in the validation loss for it to be considered 
                barely-decreasing by the loss monitoring and early stopping mechanisms. This argument is 
                not used if :code:`model_type="lm"` or both :code:`is_loss_monitored` and 
                :code:`is_early_stopping` are :code:`False`.
            is_q_monitored (bool, optional):
                When set to :code:`True`, will monitor the Q values estimated by the neural network 
                approximator of the Q function in each iteration. A warning 
                will be raised if the percent absolute change in some Q value is greater than :code:`min_delta_q` for at 
                least one of the final :math:`r` epochs during neural network training, where :math:`r` is specified 
                by the argument :code:`patience_q`. This argument is not used if :code:`model_type="lm"`.
            is_early_stopping_q (bool, optional): 
                When set to :code:`True`, will monitor the Q values estimated by the neural network 
                approximator of the Q function in each iteration, and will enforce early stopping based on the estimated Q values 
                when training the neural network approximator of the Q function in each iteration. That is, 
                FQE training will stop early 
                if the percent absolute changes in all the predicted Q values are no greater than :code:`min_delta_q` for :math:`r` consecutive 
                iterations, where :math:`r` is specified by the argument :code:`patience_q`. This argument is not used if 
                :code:`model_type="lm"`.
            patience_q (int, optional): 
                The number of consequentive epochs with barely-changing estimated Q values that is needed 
                for Q value monitoring and early stopping. This argument is not used if :code:`model_type="lm"` 
                or both :code:`is_q_monitored` and :code:`is_early_stopping_q` are :code:`False`.
            min_delta_q (int for float, optional): 
                The maximum amount of change in the validation loss for it to be considered 
                barely-changing by the Q value monitoring and early stopping mechanisms. This argument is 
                not used if :code:`model_type="lm"` or both :code:`is_q_monitored` and 
                :code:`is_early_stopping_q` are :code:`False`.

    Returns: 
        discounted_cumulative_reward (np.integer or np.floating): 
            An estimation of the discounted 
            cumulative reward achieved by the policy throughout the trajectory. 
    """
    
    if model_type != 'nn':
        raise InvalidModelError("Invalid model type. Only 'nn' is currently supported.")
    np.random.seed(seed)
    zs = np.array(zs)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    action_size = len(np.unique(actions.flatten(), axis=0))

    fqe = FQE(model_type=model_type, 
              num_actions=action_size, 
              policy=policy, 
              hidden_dims=hidden_dims, 
              learning_rate=learning_rate, 
              epochs=epochs, 
              gamma=gamma, 
              is_loss_monitored=is_loss_monitored,
              is_early_stopping_nn=is_early_stopping_nn,
              test_size_nn=test_size_nn,
              patience_nn=patience_nn,
              min_delta_nn=min_delta_nn, 
              is_q_monitored=is_q_monitored,
              is_early_stopping_q=is_early_stopping_q,
              patience_q=patience_q,
              min_delta_q=min_delta_q
             )
    fqe.fit(
        states=states, zs=zs, actions=actions, rewards=rewards, 
        max_iter=max_iter, f_ua=f_ua
    )

    return np.mean(fqe.evaluate(zs=zs, states=states, actions=actions, f_ua=f_ua))