Custom Agents
=========================

In addition to :code:`FQI`, users can also define custom decision-making agents and use them 
in PyCFRL. A custom decision-agent often represents a pre-specified decision rule (e.g. making 
decisions randomly) or implements another reinforcement learning algorithm not provided by PyCFRL.

To ensure a custom agent is compatible with PyCFRL, it must inherit from the 
:code:`Agent` class provided by the :code:`agents` module. That is, 

- The custom agent should be a subclass of :code:`Agent`.
- The custom preprocessor should have an :code:`act()` method whose function name, 
  parameter names, parameter data types, parameter default values, and return type are 
  exactly as that defined in the :code:`Agent` class, except that it might have some additional 
  arguments. The input and output lists or arrays should also follow the same 
  :ref:`Trajectory Array format <trajectory_arrays>` or have the same shape as those defined in 
  :code:`Preprocessor`.

For example, though simple, the following :code:`RandomAgent` is a valid custom 
agent that will be compatible with PyCFRL.

.. code-block:: python

    class RandomAgent(Agent):
        def __init__(self, num_action_levels: int):
            self.num_action_levels = num_action_levels
            self.__name__ = 'RandomAgent'

        def act(self, 
                z: list | np.ndarray, 
                xt: list | np.ndarray, 
                xtm1: list | np.ndarray | None = None, 
                atm1: list | np.ndarray | None = None, 
                uat: list | np.ndarray | None = None, 
                is_return_probs: bool = False, 
                verbose: bool = False
                ) -> np.ndarray:
            if verbose:
                print("RandomAgent taking actions...")
            N = z.shape[0]
            if uat is None:
                out = np.zeros(N)
                for i in range(N):
                    out[i] = np.random.randint(self.num_action_levels)
                if is_return_probs:
                    factor = 1 / self.num_action_levels
                    probs = np.ones((N, self.num_action_levels)) * factor
                    return probs
                else:
                    return out
            else:
                action = (uat.flatten() <= 0.5).astype(int)
                if is_return_probs:
                    probs = np.ones((N, self.num_action_levels)) * 0.5
                    return probs
                else:
                    return action


On the other hand, the following agent will not be compatible with PyCFRL 
because its :code:`act()` does not have :code:`uat` in its argument list. 
:code:`uat` should be included in the argument list here to ensure 
compatibility even though it is not used in the function.

.. code-block:: python

    class RandomAgent(Agent):
        def __init__(self, num_action_levels: int):
            self.num_action_levels = num_action_levels
            self.__name__ = 'RandomAgent'

        def act(self, 
                z: list | np.ndarray, 
                xt: list | np.ndarray, 
                xtm1: list | np.ndarray | None = None, 
                atm1: list | np.ndarray | None = None, 
                is_return_probs: bool = False, 
                verbose: bool = False) -> np.ndarray:
            if verbose:
                print("RandomAgent taking actions...")
            N = z.shape[0]
            if uat is None:
                out = np.zeros(N)
                for i in range(N):
                    out[i] = np.random.randint(self.num_action_levels)
                if is_return_probs:
                    factor = 1 / self.num_action_levels
                    probs = np.ones((N, self.num_action_levels)) * factor
                    return probs
                else:
                    return out
            else:
                action = (uat.flatten() <= 0.5).astype(int)
                if is_return_probs:
                    probs = np.ones((N, self.num_action_levels)) * 0.5
                    return probs
                else:
                    return action


If an agent is a valid custom agent, then it can be used wherever 
an :code:`FQI` can be used. For example, we can use 
:code:`evaluate_fairness_through_model()` to calculate its counterfactual 
fairness metric.

.. code-block:: python

    # Suppose e is a SimulatedEnvironment object that has already been trained.
    # Also suppose zs, states, actions is a trajectory from the environment 
    # on which e is trained.
    agent = RandomAgent(p=0.2)
    evaluate_fairness_through_model(env=e, 
                                    zs=zs, 
                                    states=states, 
                                    actions=actions, 
                                    policy=agent)