import numpy as np
from sklearn.linear_model import LinearRegression
from pycfrl.utils.base_models import NeuralNetRegressor
from pycfrl.preprocessor.preprocessor import Preprocessor
#from utils.utils import timer_func
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

# Preprocessor for the baseline "unaware"
class UnawarenessPreprocessor(Preprocessor):
    def __init__(self, z_space, action_space) -> None:
        self.action_space = action_space
        self.z_space = z_space

    def preprocess(self, xt, **kwargs):
        """Preprocess the data

        Args:
            z (np.ndarray): sensitive variables, dim: (N, zdim) or (zdim,)
            xt (np.ndarray): states, dim: (N, xdim) or (xdim,)
        Returns:
            xt_new (np.ndarray): processed states, dim: (N, xdim) or (xdim,)
        """

        return xt   
    
    def preprocess_single_step(self, z, xt, xtm1=None, atm1=None, rtm1=None, set_seed=True, **kwargs):
        if rtm1 is None:
            return xt
        else:
            return xt, rtm1
        
    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        if rewards is not None:
            return xs, rewards
        else:
            return xs
        
    

# Preprocessor for the baseline "full"
class ConcatenatePreprocessor(Preprocessor):
    def __init__(self, z_space, action_space) -> None:
        self.action_space = action_space
        self.z_space = z_space

    def preprocess(self, z, xt, **kwargs):
        """Preprocess the data

        Args:
            z (np.ndarray): sensitive variables, dim: (N, zdim) or (zdim,)
            xt (np.ndarray): states, dim: (N, xdim) or (xdim,)
        Returns:
            xt_new (np.ndarray): processed states, dim: (N, xdim) or (xdim,)
        """

        if xt.ndim == 1:
            xt = xt[np.newaxis, :]
            z = z[np.newaxis, :]
            xt_new = np.concatenate([xt, z], axis=1)
            return xt_new.flatten()
        elif xt.ndim == 2:
            xt_new = np.concatenate([xt, z], axis=1)
            return xt_new
        
    def preprocess_single_step(self, z, xt, xtm1=None, atm1=None, rtm1=None, set_seed=True, **kwargs):
        xt_new = self.preprocess(z, xt)
        if rtm1 is None:
            return xt_new
        else:
            return xt_new, rtm1
        

    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        # some convenience variables
        N, T, xdim = xs.shape
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim + zs.shape[-1]])
        rs_tilde = np.zeros([N, T - 1])

        # preprocess the initial step
        np.random.seed(0)
        xs_tilde[:, 0, :] = self.preprocess_single_step(zs, xs[:, 0, :])

        # preprocess subsequent steps
        if rewards is not None:
            for t in range (1, T):
                np.random.seed(t)
                xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(zs, 
                                                                                  xs[:, t, :], 
                                                                                  xs[:, t-1, :], 
                                                                                  #actions[:, t-1, :], 
                                                                                  actions[:, t-1], 
                                                                                  rewards[:, t-1], 
                                                                                  set_seed=False
                                                                                  )
            return xs_tilde, rs_tilde                
        else:
            for t in range (1, T):
                np.random.seed(t)
                xs_tilde[:, t, :] = self.preprocess_single_step(zs, 
                                                                xs[:, t, :], 
                                                                xs[:, t-1, :], 
                                                                #actions[:, t-1, :]
                                                                actions[:, t-1], 
                                                                set_seed=False
                                                                )
            return xs_tilde



# Preprocessor for the baseline "oracle"
class SequentialPreprocessorOracle(Preprocessor):
    def __init__(self, env, z_space, action_space) -> None:
        '''self.env = env
        self.xdim = 1
        self.zdim = 1
        self.preprocess_training(xs, zs, actions, rewards)'''
        self.z_space = z_space
        self.action_space = action_space
        self.env = env
        self.__name__ = 'SequentialPreprocessorOracle'

    #@timer_func
    def learn_transition_models(self, **kwargs):
        # learn model at time 0
        model0 = {}
        zs = np.array([[0], [1]])
        for z in np.unique(zs, axis=0):
            model0[tuple(z)] = self.env.f_x0(zs=z.reshape(1, 1), ux0=np.zeros((1, 1)), 
                                             z_coef=self.env.z_coef)

        # learn model for time > 0
        class Model:
            def __init__(self, env) -> None:
                self.env = env

            def predict(self, xtm1, z, atm1):
                xtm1 = xtm1.reshape(-1, 1)
                z = z.reshape(-1, 1)
                atm1 = atm1.reshape(-1, 1)
                uxt = np.zeros_like(xtm1)
                urt = np.zeros_like(uxt)
                xt = self.env.f_xt(z, xtm1, atm1, uxt, z_coef=self.env.z_coef)
                rtm1 = self.env.f_rt(z, xtm1, atm1, urt, z_coef=self.env.z_coef)
                return np.concatenate([xt[:, np.newaxis], rtm1[:, np.newaxis, np.newaxis]], axis=1)

        model = Model(self.env)

        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, actions, states):
        m = model.predict(states, z, actions)
        return m

    def train_preprocessor(self, zs, xs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape
        self.N, self.T, self.xdim = xs.shape
        self.zdim = zs.shape[-1]

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None]
        self.model = [None]

        self.model0[0], self.model[0] = self.learn_transition_models()
        self.reset_buffer(n=N)
        xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
        rs_tilde = np.zeros([N, T - 1])
        for t in range(T):
            if t == 0:
                #for_test = np.array([self.model0[0][tuple(z)] for z in zs])
                epsilon_hat = xs[:, 0, :] - np.array(
                    [self.model0[0][tuple(z)] for z in zs]
                ).reshape(-1, xdim)
                xt_c = {}
                for key, prob in self.marginal_dist_z.items():
                    xt_c_mean = np.broadcast_to(self.model0[0][key], (N, xdim))
                    xt_c[key] = xt_c_mean + epsilon_hat
                xt_tilde = np.hstack(
                    [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                )
                self.buffer = xt_c.copy()
            else:
                m = self._estimate_cf_next_state_reward_mean_tg1(
                    self.model[0], zs, actions[:, t - 1], xs[:, t - 1, :]
                )
                xs_mean = m[:, :-1]
                r_mean = m[:, -1]
                epsilon_hat = xs[:, t, :] - xs_mean.reshape(-1, xdim)
                epsilon_hat_reward = rewards[:, t - 1].reshape(-1, 1) - r_mean
                xt_c, rt_c = {}, {}
                for key, prob in self.marginal_dist_z.items():
                    key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        self.model[0], key_vec, actions[:, t - 1], self.buffer[key]
                    )
                    xs_mean = m[:, :-1]
                    r_mean = m[:, -1]
                    xt_c[key] = xs_mean.reshape(-1, xdim) + epsilon_hat
                    rt_c[key] = r_mean + epsilon_hat_reward
                xt_tilde = np.hstack(
                    [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                )
                rt_tilde = np.sum(
                    [rt_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
                self.buffer = xt_c.copy()
                rs_tilde[:, t - 1] = rt_tilde.reshape(-1)
            xs_tilde[:, t, :] = xt_tilde
        return xs_tilde, rs_tilde

    def preprocess_single_step(self, z, xt, xtm1=None, atm1=None, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)
        xt_tilde = None
        rtm1_tilde = None

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)
            xt_mean = np.zeros_like(xt)

            # estimate epsilon_hat
            for k in range(cross_folds):
                model0 = self.model0[k]

                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    xt_mean[idx_z, :] += model0[tuple(z_)] / cross_folds
            epsilon_hat = xt - xt_mean

            # estimate counterfactual outcome
            xt_c = {}

            for key, prob in self.marginal_dist_z.items():
                xt_c_mean = np.zeros_like(xt)
                for k in range(cross_folds):
                    model0 = self.model0[k]
                    xt_c_mean += np.broadcast_to(model0[key], (N, xdim)) / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
            xt_tilde = np.hstack(
                [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            )
            self.buffer = xt_c.copy()

        else:
            atm1_new = atm1.reshape(-1, 1)
            xt_mean = np.zeros_like(xt)
            rtm1_mean = np.zeros((xt.shape[0]))

            # estimate epsilon_hat
            for k in range(cross_folds):
                model = self.model[k]
                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    z_ = np.repeat(z_.reshape(1, -1), repeats=sum(idx_z), axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, z_, atm1_new[idx_z], xtm1[idx_z]
                    )
                    for_test = m[:, :-1] / cross_folds
                    xt_mean[idx_z, :] += m[:, :-1].reshape(-1, xdim) / cross_folds
                    rtm1_mean[idx_z] += m[:, -1].reshape(-1) / cross_folds

            epsilon_hat = xt - xt_mean
            if rtm1 is not None:
                epsilon_hat_reward = rtm1 - rtm1_mean
            # predict counterfactual outcome
            xt_c = {}
            rtm1_c = {}

            for key, prob in self.marginal_dist_z.items():
                key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                xt_c_mean = np.zeros_like(xt)
                rtm1_c_mean = np.zeros((xt.shape[0]))
                for k in range(cross_folds):
                    model = self.model[k]
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, key_vec, atm1_new, self.buffer[key]
                    )
                    xt_c_mean += m[:, :-1].reshape(-1, xdim) / cross_folds
                    rtm1_c_mean += m[:, -1].reshape(-1) / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
                if rtm1 is not None:
                    rtm1_c[key] = rtm1_c_mean + epsilon_hat_reward
            xt_tilde = np.hstack(
                [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            )
            if rtm1 is not None:
                # print(self.marginal_dist_z.items())
                rtm1_tilde = np.sum(
                    [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
            self.buffer = xt_c.copy()
        if rtm1 is None:
            return xt_tilde
        else:
            return xt_tilde, rtm1_tilde

    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        # some convenience variables
        N, T, xdim = xs.shape

        # check if state dimension of the input is the same as state dimension of the training data
        if xdim != self.xdim:
            raise ValueError('The state dimension of the input does not match that of the training data.')
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
        rs_tilde = np.zeros([N, T - 1])

        # preprocess the initial step
        xs_tilde[:, 0, :] = self.preprocess_single_step(zs, xs[:, 0, :])

        # preprocess subsequent steps
        if rewards is not None:
            for t in range (1, T):
                xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(zs, 
                                                                                  xs[:, t, :], 
                                                                                  xs[:, t-1, :], 
                                                                                  #actions[:, t-1, :], 
                                                                                  actions[:, t-1], 
                                                                                  rewards[:, t-1]
                                                                                  )
            return xs_tilde, rs_tilde                
        else:
            for t in range (1, T):
                xs_tilde[:, t, :] = self.preprocess_single_step(zs, 
                                                                xs[:, t, :], 
                                                                xs[:, t-1, :], 
                                                                #actions[:, t-1, :]
                                                                actions[:, t-1]
                                                                )
            return xs_tilde