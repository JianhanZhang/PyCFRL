import numpy as np
import torch
import copy
from scipy.special import expit
from pycfrl.agents.agents import Agent

# A random policy
class RandomAgent(Agent):
    def __init__(self, num_action_levels):
        self.num_action_levels = num_action_levels
        self.p1 = 0.5
        self.__name__ = 'RandomAgent'

    def act(self, z, xt, xtm1=None, atm1=None, uat=None, **kwargs):
        if uat is None:
            #np.random.seed(10)
            N = z.shape[0]
            out = np.zeros(N)
            for i in range(N):
                out[i] = np.random.randint(self.num_action_levels)
            return out
            #return np.random.randint(self.num_action_levels)
        else:
            action = (uat.flatten() <= self.p1).astype(int)
            # logits = np.ones_like(uat) * logit(self.p1)
            #p1 = np.ones_like(uat) * self.p1
            #p0 = 1 - p1
            return action
        


# A binary behavioral policy
class BehaviorAgent(Agent):
    def __init__(self, seed=1) -> None:
        super().__init__()
        np.random.seed(seed)
        # self.eta = np.random.uniform(-0.9, 0.9, size=[4])
        # self.eta = np.array([-0.5, 0.0, 0.5, 0.0])
        self.eta = np.array([-1.39, 0.0, 2.77, 0.0])
        self.name = "behavior"

    def act(self, xt, z, uat, is_return_prob=False, **kwargs):
        n = xt.shape[0]
        M = np.concatenate(
            [np.ones([n, 1]), xt, z, xt * z],
            axis=1,
        )
        ps = expit(M @ self.eta)
        action_behavior = (uat.flatten() <= ps.flatten()).astype(int)
        action_random = (uat.flatten() <= 0.5).astype(int)

        idx_random = np.random.uniform(0, 1, size=[n]) <= 0.0  # epsilon-greedy
        action = action_behavior * (1 - idx_random) + action_random * idx_random
        #action = action_behavior
        if is_return_prob:
            return action, np.vstack([1 - ps, ps]).T
        else:
            return action