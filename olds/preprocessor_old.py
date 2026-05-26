from abc import abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.base_models import NeuralNetRegressor, LinearRegressor
#from utils.utils import timer_func
from utils.custom_errors import InvalidModelError
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from typing import Union, Literal


class Preprocessor:
    """
    Base class for preprocessors.

    Subclasses must implement the :code:`preprocess_single_step` and :code:`preprocess_multiple_steps` 
    methods.
    """

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def preprocess_single_step(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            xtm1: list | np.ndarray | None = None, 
            atm1: list | np.ndarray | None = None, 
            rtm1: list | np.ndarray | None = None, 
            **kwargs
        ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        An abstract prototype of methods that preprocess the states at a single time step.

        Args: 
           zs (list or np.ndarray): 
                The observed sensitive attributes of each individual in the trajectory that 
                is to be preprocessed. It should be a 2D list or array following 
                the Sensitive Attributes Format.
            xt (list or np.ndarray): 
                The states at the current time step of each individual in the trajectory that 
                is to be preprocessed. It should be a 2D list or array following the Single-time 
                States Format.
            xtm1 (list or np.ndarray, optional): 
                The states at the previous time step of each individual  
                in the trajectory that is to be preprocessed. It should be a 2D list or array 
                following the Single-time States Format.
            atm1 (list or np.ndarray, optional): 
                The actions at the previous time step of each individual  
                in the trajectory that is to be preprocessed. It should be a 1D list or array following 
                the Single-time Actions Format. 
            rtm1 (list or np.ndarray, optional): 
                The rewards at the previous time step of each individual in the trajectory that 
                is to be preprocessed. It should be a 2D list or array following the 
                Single-time States Format.

        Returns:
            xt_tilde (np.ndarray): 
                The preprocessed states at the given time step. It should be a 2D array following 
                the Single-time States Format.
            rt_tilde (np.ndarray, optional): 
                The preprocessed rewards at the given time step. It should be a 1D array following 
                the Single-time Rewards Format. :code:`rt_tilde` is not returned if :code:`rtm1=None` 
                in the function input.  
        """
        
        pass

    @abstractmethod
    def preprocess_multiple_steps(
            self, 
            zs: list | np.ndarray, 
            xs: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray | None = None, 
            **kwargs
        ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        r"""
        An abstract prototype of methods that preprocess a whole trajectory.

        Args: 
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual in the trajectory that is 
                to be preprocessed. It should be a 2D list or array following the Sensitive 
                Attributes Format.
            states (list or np.ndarray): 
                The state trajectory that is to be preprocessed. It should be 
                a 3D list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory that is to be preprocessed, often generated using a behavior 
                policy. It should be a 2D list or array following the Full-trajectory Actions 
                Format.
            rewards (list or np.ndarray, optional): 
                The reward trajectory that is to be preprocessed. It should be 
                a 2D list or array following the Full-trajectory Rewards Format.

        Returns:
            xs_tilde (np.ndarray): 
                The preprocessed states trajectory. It should be a 3D array following 
                the Full-trajectory States Format.
            rs_tilde (np.ndarray, optional): 
                The preprocessed reward trajectory. It should be a 2D array following the 
                Full-trajectory Rewards Format. :code:`rs_tilde` is not returned if 
                :code:`rewards=None` in the function input.  
        """

        pass

    @staticmethod
    def standardize(
            x: np.ndarray, 
            mean: np.ndarray | None = None, 
            std: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
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

    def reset_buffer(self, n: int) -> None:
        self.buffer = {}
        for key, prob in self.marginal_dist_z.items():
            self.buffer[key] = np.zeros((n, self.xdim))

    '''def _hash_tuples(self, tuples):
        if tuples.ndim == 1:
            hashing_tuples = "_".join(np.array(tuples).astype("str"))
        else:
            hashing_tuples = np.apply_along_axis(
                lambda x: "_".join(np.array(x).astype("str")), axis=1, arr=tuples
            )
        return hashing_tuples

    def _reverse_hash_tuples(self, hashing_tuples):
        if isinstance(hashing_tuples, str):
            tuples = np.array(hashing_tuples.split("_")).astype("float")
        else:
            tuples = np.apply_along_axis(
                lambda x: np.array(x.split("_")).astype("float"),
                axis=0,
                arr=hashing_tuples,
            )
        return tuples'''


class SequentialPreprocessor(Preprocessor):
    r"""
    Implementation of the sequential data preprocessing method proposed by Wang et al. (2025).

    The preprocessor first learns a model :math:`\mu(s, a, z)` of the transition dynamics of 
    the MDP underlying the input trajectory. Then, at each time step, it uses :math:`\mu` to 
    reconstruct the counterfactual states and concatenates the reconstructed counterfactual 
    states into a new augmented state vector.

    That is, let :math:`z_i` be the observed sensitive attribute. At :math:`t=0` (i.e. the initial 
    time step), for each individual :math:`i` and sensitive attribute level :math:`z`, the 
    preprocessor calculates 

    .. math::
        
        \hat{x}_{i1}^z = x_{i1} - \hat{\mathbb{E}}(X_1|Z=z_i) + \hat{\mathbb{E}}(X_1|Z=z)

    and forms :math:`\tilde{x}_{i1} = [\hat{x}_{i1}^{z^{(1)}}, \dots, \hat{x}_{i1}^{z^{(K)}}]`.

    At :math:`t>0`, for each individual :math:`i` and sensitive attribute level :math:`z`, the 
    preprocessor calculates

    .. math::
        
        [\hat{x}_{it}^z, \hat{r}_{i,t-1}^z] = x_{i1} - \hat{\mu}(x_{i,t-1}, a_{i,t-1}, z_i) 
            + \hat{\mu}(\hat{x}_{i,t-1}^z, a_{i,t-1}, z)

    and forms :math:`\tilde{x}_{it} = [\hat{x}_{it}^{z^{(1)}}, \dots, \hat{x}_{it}^{z^{(K)}}]` 
    and :math:`\tilde{r}_{i,t-1} = \Sigma_{k=1}^K\hat{\mathbb{P}}(Z=z^{(k)})\hat{r}_{i,t-1}^{z^{(K)}}`.

    References: 
        .. [2] Wang, J., Shi, C., Piette, J.D., Loftus, J.R., Zeng, D. and Wu, Z. (2025). 
               Counterfactually Fair Reinforcement Learning via Sequential Data 
               Preprocessing. arXiv preprint arXiv:2501.06366.
    """

    def __init__(
        self,
        z_space: list | np.ndarray,
        #action_space: list | np.ndarray | None = None,
        num_actions: int, 
        cross_folds: int = 1,
        mode: Literal["single", "sensitive"] = "single", 
        reg_model: Literal["lm", "nn"] = "nn",
        hidden_dims: list[int] = [64, 64], 
        epochs: int = 1000,
        learning_rate: int | float = 0.005,
        batch_size: int = 512,
        is_action_onehot: bool = True,
        is_normalized: bool = False,
        is_early_stopping: bool = False,
        test_size: int | float = 0.2,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: int | float = 0.01,
    ) -> None:
        """
        Args: 
            z_space (list or np.ndarray): 
                A 2D list or array of shape (K, zdim) where K is the 
                total number of legit values of the sensitive attribute and zdim is the dimension  
                of the sensitive attribute variable. It contains all legit values of the sensitive 
                attribute. Each legit value should occupy a separate row.
            num_actions (int): 
                The total number of legit actions. 
            cross_folds (int, optional): 
                The number of cross folds used during training. When 
                :code:`cross_folds=k`, the preprocessor will learn :code:`k` models using different 
                subset of the training data, and the final output of :code:`preprocess_single_step` and 
                :code:`preprocess_multiple_steps` will be generally the average of the outputs from each 
                of the :code:`k` models.
            mode (str, optional): 
                Can either be "single" or "sensitive". When :code:`mode="single"`, 
                the preprocessor will learn a single model of the transition dynamics where the 
                sensitive attribute is an input to the model. When :code:`mode="sensitive"`, the 
                preprocessor will learn one transition dynamics model for each level of the sensitive 
                attribute, and transitions under each sensitive attribute :math:`z` will 
                be estimated using the model corresponding to :math:`z`.
            reg_model (str, optional): 
                The type of the model used for learning the transition  
                dynamics. Can be "lm" (polynomial regression) or "nn" (neural network). 
                *Currently, only 'nn' is supported.*
            hidden_dims (list[int], optional): 
                The hidden dimensions of the neural network. This 
                argument is not used if :code:`reg_model="lm"`.
            epochs (int, optional): 
                The number of training epochs for the neural network. This 
                argument is not used if `reg_model="lm"`. 
            learning_rate (int or float, optional): 
                The learning rate of the neural network. This 
                argument is not used if :code:`reg_model="lm"`. 
            batch_size (int, optional): 
                The batch size of the neural network. This argument is 
                not used if :code:`reg_model="lm"`.
            is_action_onehot (bool, optional): 
                When set to :code:`True`, the actions will be one-hot 
                encoded. 
            is_normalized (bool, optional): 
                When set to :code:`True`, the states will be normalized 
                following the formula :code:`x_normalized = (x - mean(x)) / std(x)`.
            is_early_stopping (bool, optional): 
                When set to :code:`True`, will enforce early stopping 
                during neural network training. This argument is not used if :code:`reg_model="lm"`.
            test_size (int or float, optional): 
                An int or float between 0 and 1 (inclusive) that 
                specifies the proportion of the full data that is used as the test set for early 
                stopping. This argument is not used if :code:`reg_model="lm"` or 
                :code:`is_early_stopping=False`.
            early_stopping_patience (int, optional): 
                The number of consequentive epochs with 
                barely-decreasing loss needed for training to be early stopped. This argument is 
                not used if :code:`reg_model="lm"` or :code:`is_early_stopping=False`.
            early_stopping_min_delta (int for float, optional): 
                The minimum amount of decrease 
                in the loss so that the rounded is not considered barely-decreasing by the early 
                stopping mechanism. This argument is not used if :code:`reg_model="lm"` or 
                :code:`is_early_stopping=False`.
        """

        z_space = np.array(z_space)
        '''if action_space is not None:
            action_space = np.array(action_space)
        if (is_action_onehot) and (action_space is None):
            raise ValueError('One hot encoding of actions requires action_space to be not None.')'''

        if reg_model == 'nn':
            self.reg_model = reg_model
        else:
            raise InvalidModelError("Invalid model type. Only 'nn' is currently supported.")
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_action_onehot = is_action_onehot
        self.is_normalized = is_normalized
        #self.action_space = action_space
        self.action_space = np.array([a for a in range(num_actions)]).reshape(-1, 1)
        self.num_actions = num_actions
        self.z_space = z_space
        self.zdim = z_space.shape[-1]
        self.__name__ = 'SequentialPreprocessor'

        # tunnable parameters
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.cross_folds = cross_folds
        self.mode = mode

    def encode_a(self, a: np.ndarray) -> np.ndarray:
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def _learn_initial_model(
            self, 
            xs: np.ndarray, 
            zs: np.ndarray
        ) -> dict[tuple[Union[int, float], ...], np.ndarray]:
        # learn model at time 0
        model0 = {}
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            model0[tuple(z)] = np.mean(xs[idx_z, 0, :], axis=0)
        return model0

    def _learn_transition_models(
            self, 
            xs: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray
        ) -> LinearRegression | NeuralNetRegressor | dict[tuple[int | float, ...], LinearRegressor] | dict[tuple[int | float, ...], NeuralNetRegressor]:
        # learn model after time 0
        N, T, _ = xs.shape
        if self.is_normalized:
            states, self.states_mean, self.states_std = self.standardize(
                xs[:, : (T - 1), :].reshape(-1, self.xdim)
            )
            next_states, self.next_states_mean, self.next_states_std = self.standardize(
                np.concatenate(
                    [
                        xs[:, 1:T, :].reshape(-1, self.xdim),
                        rewards[:, :].reshape(-1, 1),
                    ],
                    axis=1,
                )
            )
        else:
            states = xs[:, : (T - 1), :].reshape(-1, self.xdim)
            next_states = np.concatenate(
                [xs[:, 1:T, :].reshape(-1, self.xdim), rewards[:, :].reshape(-1, 1)],
                axis=1,
            )

        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(N, T - 1, -1)
            self.dim_a = actions.shape[-1]
        else:
            self.dim_a = 1

        if self.reg_model == "lm":
            return self._learn_linear_model(states, next_states, zs, actions, T)
        elif self.reg_model == 'nn':
            return self._learn_neural_model(states, next_states, zs, actions, T)
        else:
            print("Model type is undefined. Please specify either \"lm\" or \"nn\".")
            exit(1)

    def _learn_linear_model(
            self, 
            states: np.ndarray, 
            next_states: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            T: int
        ) -> LinearRegressor | dict[tuple[int | float, ...], LinearRegressor]:
        if self.mode == "single":
            return self._learn_single_linear_model(states, next_states, zs, actions, T)
        elif self.mode == "sensitive":
            return self._learn_sensitive_linear_model(
                states, next_states, zs, actions, T
            )
        else: 
            print("Model mode is undefined. Please specify either \"single\" or \"sensitive\".")
            exit(1)

    def _learn_single_linear_model(
            self, 
            states: np.ndarray, 
            next_states: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            T: int
        ) -> LinearRegressor:
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, self.dim_a),
                states,
            ],
            axis=1,
        )
        Y = next_states
        #return LinearRegression().fit(X, Y) # JITAO'S ORIGINAL
        lr = LinearRegressor(featurize_method='polynomial', degree=2)
        lr.fit(X, Y)
        return lr
    
    def _learn_sensitive_linear_model(
            self, 
            states: np.ndarray, 
            next_states: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            T: int
        ) -> dict[tuple[int | float, ...], LinearRegressor]:
        model = {}
        zs = np.repeat(zs[:, np.newaxis, :], axis=1, repeats=T - 1).reshape(
            -1, self.zdim
        )
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            states_z = states[idx_z].reshape(-1, self.xdim)
            next_states_z = next_states[idx_z].reshape(-1, next_states.shape[-1])
            actions_z = actions[:, : (T - 1)].reshape(-1, self.dim_a)[idx_z]
            X = np.concatenate([actions_z, states_z], axis=1)
            Y = next_states_z

            #model[tuple(z)] = LinearRegression().fit(X, Y)
            lr = LinearRegressor(featurize_method='polynomial', degree=2)
            lr.fit(X, Y)
            model[tuple(z)] = lr
        return model

    def _learn_neural_model(
            self, 
            states: np.ndarray, 
            next_states: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            T: int
        ) -> NeuralNetRegressor | dict[tuple[int | float, ...], NeuralNetRegressor]:
        if self.mode == "single":
            return self._learn_single_neural_model(states, next_states, zs, actions, T)
        elif self.mode == "sensitive":
            return self._learn_sensitive_neural_model(
                states, next_states, zs, actions, T
            )
        else: 
            print("Model mode is undefined. Please specify either \"single\" or \"sensitive\".")
            exit(1)


    def _learn_single_neural_model(
            self, 
            states: np.ndarray, 
            next_states: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            T: int
        ) -> NeuralNetRegressor:
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, self.dim_a),
                states,
            ],
            axis=1,
        )
        Y = next_states
        model = NeuralNetRegressor(
            in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
        )
        model.fit( # there used to be mistakenly calling "model.train()"
            X,
            Y,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            is_early_stopping=self.is_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            test_size=self.test_size,
        )
        return model

    def _learn_sensitive_neural_model(
            self, 
            states: np.ndarray, 
            next_states: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            T: int
        ) -> dict[tuple[int | float, ...], NeuralNetRegressor]:
        model = {}
        zs = np.repeat(zs[:, np.newaxis, :], axis=1, repeats=T - 1).reshape(
            -1, self.zdim
        )
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            states_z = states[idx_z].reshape(-1, self.xdim)
            next_states_z = next_states[idx_z].reshape(-1, next_states.shape[-1])
            actions_z = actions[:, : (T - 1)].reshape(-1, self.dim_a)[idx_z]
            X = np.concatenate([actions_z, states_z], axis=1)
            Y = next_states_z

            model[tuple(z)] = NeuralNetRegressor(
                in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
            )
            model[tuple(z)].fit(
                X,
                Y,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
        return model

    def learn_transition_models(
            self, 
            xs: np.ndarray, 
            zs: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray
        ) -> tuple[dict[tuple[Union[int, float], ...], np.ndarray], 
                        Union[LinearRegression, NeuralNetRegressor]]:
        model0 = self._learn_initial_model(xs, zs)
        model = self._learn_transition_models(xs, zs, actions, rewards)
        return model0, model

    def learn_marginal_dist_z(
            self, 
            zs: np.ndarray
        ) -> dict[tuple[Union[int, float], ...], Union[int, float]]:
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(
            self, 
            model: LinearRegression | NeuralNetRegressor, 
            z: np.ndarray, 
            at: np.ndarray, 
            xt: np.ndarray
        ) -> np.ndarray:
        N = xt.shape[0]
        if self.is_normalized:
            state = self.standardize(xt, self.states_mean, self.states_std)
        else:
            state = xt

        if self.is_action_onehot:
            encoded = self.encode_a(at.reshape(-1, 1))
            at = encoded.reshape(N, -1)
            dim_a = at.shape[-1]
        else:
            dim_a = 1

        if self.mode == "single":
            m = model.predict(
                np.concatenate(
                    [
                        z,
                        at.reshape(-1, dim_a),
                        state,
                    ],
                    axis=1,
                )
            )
        elif self.mode == "sensitive":
            m = np.zeros((xt.shape[0], self.xdim + 1))
            for z_ in self.z_space:
                idx_z = np.all(z == z_, axis=1)
                states_z = state[idx_z]
                actions_z = at[idx_z].reshape(-1, dim_a)
                X = np.concatenate([actions_z, states_z], axis=1)
                if self.reg_model == "nn" or X.shape[0] > 0:
                    m[idx_z] = model[tuple(z_)].predict(X)
        if self.is_normalized:
            m = self.destandardize(m, self.next_states_mean, self.next_states_std)
        return m

    def _process_initial_state(
            self, 
            initial_model: dict[tuple[Union[int, float], ...], np.ndarray], 
            z: np.ndarray, 
            xt: np.ndarray
        ) -> tuple[np.ndarray, dict[tuple[Union[int, float], ...], np.ndarray]]:
        zs = z
        del(z)
        N = xt.shape[0]
        epsilon_hat = xt - np.array([initial_model[tuple(z.flatten())] for z in zs])
        xt_c = {
            key: np.broadcast_to(initial_model[key], (N, self.xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }
        xt_tilde = np.hstack([xt_c[key] for key in self.marginal_dist_z])
        return xt_tilde, xt_c

    def _process_subsequent_states(
        self, 
        transition_model: LinearRegression | NeuralNetRegressor, 
        z: np.ndarray, 
        xt: np.ndarray, 
        xtm1: np.ndarray, 
        atm1: np.ndarray, 
        rtm1: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[tuple[Union[int, float], ...], np.ndarray]]:
        N = xt.shape[0]
        m = self._estimate_cf_next_state_reward_mean_tg1(
            model=transition_model, z=z, at=atm1, xt=xtm1
        )
        xt_mean = m[:, :-1]
        epsilon_hat = xt - xt_mean

        if rtm1 is not None:
            rtm1_mean = m[:, -1]
            epsilon_hat_reward = rtm1 - rtm1_mean

        xt_c = {}
        rtm1_c = {}
        for key, prob in self.marginal_dist_z.items():
            key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
            m = self._estimate_cf_next_state_reward_mean_tg1(
                model=transition_model, z=key_vec, at=atm1, xt=self.buffer[key]
            )
            xt_mean = m[:, :-1]
            xt_c[key] = xt_mean + epsilon_hat
            if rtm1 is not None:
                rtm1_mean = m[:, -1]
                rtm1_c[key] = rtm1_mean + epsilon_hat_reward

        xt_tilde = np.hstack(
            [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
        )
        if rtm1 is not None:
            rtm1_tilde = np.sum(
                [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                axis=0,
            )
        else:
            rtm1_tilde = None

        return xt_tilde, rtm1_tilde, xt_c

    def train_preprocessor(
            self, 
            zs: list | np.ndarray, 
            xs: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Train the sequential preprocessor and preprocess the training trajectory.

        When some :math:`k>1` cross folds are specified, then :math:`k` transition models will be trained, 
        each using all but one of the folds. That is, for each fold in the training trajectory, we train 
        a model using all the other folds, and we preprocess the current fold with this model. 
        The detailed preprocessing procedure can be found 
        `here <https://github.com/JianhanZhang/CFRL/blob/main/examples/real_data_workflow_description.pdf>`_.

        Args: 
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the training data. It should be a 2D list or array following the Sensitive 
                Attributes Format.
            xs (list or np.ndarray): 
                The state trajectory used for training. It should be 
                a 3D list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory used for training, often generated using a behavior 
                policy. It should be a 2D list or array following the Full-trajectory Actions 
                Format.
            rewards (list or np.ndarray): 
                The reward trajectory used for training. It should be 
                a 2D list or array following the Full-trajectory Rewards Format.

        Returns:
            xs_tilde (np.ndarray): 
                The preprocessed states trajectory. It should be a 3D array following 
                the Full-trajectory States Format.
            rs_tilde (np.ndarray): 
                The preprocessed reward trajectory. It should be a 2D array following the 
                Full-trajectory Rewards Format. 
        """

        zs = np.array(zs)
        xs = np.array(xs)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # some convenience variables
        N, T, xdim = xs.shape
        self.N, self.T, self.xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None for _ in range(self.cross_folds)]
        self.model = [None for _ in range(self.cross_folds)]

        if self.cross_folds == 1:
            self.model0[0], self.model[0] = self.learn_transition_models(
                xs=xs, zs=zs, actions=actions, rewards=rewards
            )
            self.reset_buffer(n=N)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for t in range(T):
                if t == 0:
                    xt_tilde, xt_c = self._process_initial_state(
                        initial_model=self.model0[0], xt=xs[:, 0, :], z=zs
                    )
                else:
                    xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                        transition_model=self.model[0],
                        xt=xs[:, t, :],
                        xtm1=xs[:, t - 1, :],
                        z=zs,
                        atm1=actions[:, t - 1],
                        rtm1=rewards[:, t - 1],
                    )
                    rs_tilde[:, t - 1] = rtm1_tilde
                self.buffer = xt_c.copy()
                xs_tilde[:, t, :] = xt_tilde
        else:
            kf = KFold(n_splits=self.cross_folds)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for i, (train_index, test_index) in enumerate(kf.split(xs)):
                xs_train, xs_test = xs[train_index], xs[test_index]
                zs_train, zs_test = zs[train_index], zs[test_index]
                actions_train, actions_test = actions[train_index], actions[test_index]
                rewards_train, rewards_test = rewards[train_index], rewards[test_index]
                self.model0[i], self.model[i] = self.learn_transition_models(
                    xs=xs_train,
                    zs=zs_train,
                    actions=actions_train,
                    rewards=rewards_train,
                )
                n = xs_test.shape[0]
                self.reset_buffer(n=n)
                for t in range(T):
                    if t == 0:
                        xt_tilde_test, xt_c_test = self._process_initial_state(
                            initial_model=self.model0[i], xt=xs_test[:, 0, :], z=zs_test
                        )
                    else:
                        xt_tilde_test, rtm1_tilde_test, xt_c_test = (
                            self._process_subsequent_states(
                                transition_model=self.model[i],
                                xt=xs_test[:, t, :],
                                xtm1=xs_test[:, t - 1, :],
                                z=zs_test,
                                atm1=actions_test[:, t - 1],
                                rtm1=rewards_test[:, t - 1],
                            )
                        )
                        rs_tilde[test_index, t - 1] = rtm1_tilde_test
                    xs_tilde[test_index, t, :] = xt_tilde_test
                    self.buffer = xt_c_test.copy()
        return xs_tilde, rs_tilde

    # SEEMS WE CANNOT PREPROCESS A SINGLE STEP THAT'S NOT CONSECUTIVE? CUZ THEN THE INFO IN THE 
    # BUFFER WOULD BE INCORRECT? (BUFFER STORES THE COUNTERFACTUAL STATES FROM LAST FUNCTION CALL, 
    # IDEALLY IT IS THE COUNTERFACTUAL STATES FROM THE PREVIOUS STEP.)
    def preprocess_single_step(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            xtm1: list | np.ndarray | None = None, 
            atm1: list | np.ndarray | None = None, 
            rtm1: list | np.ndarray | None = None, 
            #**kwargs
        ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        r"""
        Preprocess one single time step of the trajectory.

        When some :math:`k>1` cross folds are specified, the final output will be the avearge of the 
        outputs of each of the :math:`k` transition models.

        Important Note: A :code:`SequentialPreprocessor` object internally stores the preprocessed 
        counterfactual states from the previous function call using a states buffer, and the 
        stored counterfactual states will be used to preprocess the inputs of the current function 
        call. In this case, suppose :code:`preprocess_single_step()` is called on a set of transitions at 
        time :math:`t` in some trajectory. Then, at the next call of :code:`preprocess_single_step()` for 
        this instance of :code:`SequentialPreprocessor`, the transitions passed to the function must be 
        from time :math:`t+1` of the same trajectory to ensure that the buffer works correctly. 
        To preprocess another trajectory, either use another instance of :code:`SequentialPreprocessor`, 
        or pass the initial step of the trajectory to :code:`preprocess_single_step()` with 
        :code:`xtm1=None` and :code:`atm1=None` to reset the buffer.

        In general, unless step-wise preprocessing is necessary, we recommend using 
        :code:`preprocess_multiple_steps()` to preprocess a whole trajectory to avoid unintended bugs.

        Args: 
           zs (list or np.ndarray): 
                The observed sensitive attributes of each individual in the trajectory that 
                is to be preprocessed. It should be a 2D list or array following 
                the Sensitive Attributes Format.
            xt (list or np.ndarray): 
                The states at the current time step of each individual in the trajectory that 
                is to be preprocessed. It should be a 2D list or array following the Single-time 
                States Format.
            xtm1 (list or np.ndarray, optional): 
                The states at the previous time step of each individual  
                in the trajectory that is to be preprocessed. It should be a 2D list or array 
                following the Single-time States Format.
            atm1 (list or np.ndarray, optional): 
                The actions at the previous time step of each individual  
                in the trajectory that is to be preprocessed. It should be a 1D list or array following 
                the Single-time Actions Format. When both :code:`xtm1` and :code:`atm1` are set to 
                :code:`None`, the preprocessor will consider the input to be from the initial time 
                step of a new trajectory, and the internal states buffer will be reset.
            rtm1 (list or np.ndarray, optional): 
                The rewards at the previous time step of each individual in the trajectory that 
                is to be preprocessed. It should be a 2D list or array following the 
                Single-time States Format.

        Returns:
            xt_tilde (np.ndarray): 
                The preprocessed states at the given time step. It should be a 2D array following 
                the Single-time States Format.
            rt_tilde (np.ndarray, optional): 
                The preprocessed rewards at the given time step. It should be a 1D array following 
                the Single-time Rewards Format. :code:`rt_tilde` is not returned if :code:`rtm1=None` 
                in the function input.  
        """

        z = np.array(z)
        xt = np.array(xt)
        if xtm1 is not None:
            xtm1 = np.array(xtm1)
        if atm1 is not None:
            atm1 = np.array(atm1)
        if rtm1 is not None:
            rtm1 = np.array(rtm1)

        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)

        # check if state dimension of the input is the same as state dimension of the training data
        if xdim != self.xdim:
            raise ValueError('The state dimension of the input does not match that of the training data.')

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)
            if self.cross_folds == 1:
                xt_tilde, xt_c = self._process_initial_state(initial_model=self.model0[0], xt=xt, z=z)
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                for k in range(cross_folds):
                    xt_tilde_k, xt_c_k = self._process_initial_state(
                        initial_model=self.model0[k], xt=xt, z=z
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        else:
            if cross_folds == 1:
                xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                    transition_model=self.model[0], xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=rtm1
                )
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                if rtm1 is not None:
                    rtm1_tilde = np.zeros_like(rtm1)
                for k in range(cross_folds):
                    xt_tilde_k, rtm1_tilde_k, xt_c_k = self._process_subsequent_states(
                        transition_model=self.model[k], xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=rtm1
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                    if rtm1 is not None:
                        rtm1_tilde += rtm1_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde
    
    def preprocess_multiple_steps(
            self, 
            zs: list | np.ndarray, 
            xs: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        r"""
        Preprocess a whole trajectory.

        When some :math:`k>1` cross folds are specified, the final output will be the avearge of the 
        outputs of each of the :math:`k` transition models.

        Args: 
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual in the trajectory that is 
                to be preprocessed. It should be a list or array following the Sensitive 
                Attributes Format.
            states (list or np.ndarray): 
                The state trajectory that is to be preprocessed. It should be 
                a list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory that is to be preprocessed, often generated using a behavior 
                policy. It should be a list or array following the Full-trajectory Actions 
                Format.
            rewards (list or np.ndarray, optional): 
                The reward trajectory that is to be preprocessed. It should be 
                a list or array following the Full-trajectory Rewards Format.

        Returns:
            xs_tilde (np.ndarray): 
                The preprocessed states trajectory. It should be a 3D array following 
                the Full-trajectory States Format.
            rs_tilde (np.ndarray, optional): 
                The preprocessed reward trajectory. It should be a 2D array following the 
                Full-trajectory Rewards Format. :code:`rs_tilde` is not returned if :code:`rewards=None` 
                in the function input.  
        """

        zs = np.array(zs)
        xs = np.array(xs)
        actions = np.array(actions)
        if rewards is not None:
            rewards = np.array(rewards)

        # some convenience variables
        N, T, xdim = xs.shape

        # check if state dimension of the input is the same as state dimension of the training data
        if xdim != self.xdim:
            raise ValueError('The state dimension of the input does not match that of the training data.')
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
        rs_tilde = np.zeros([N, T - 1])

        # preprocess the initial step
        xs_tilde[:, 0, :] = self.preprocess_single_step(z=zs, xt=xs[:, 0, :])

        # preprocess subsequent steps
        if rewards is not None:
            for t in range (1, T):
                xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(z=zs, 
                                                                                  xt=xs[:, t, :], 
                                                                                  xtm1=xs[:, t-1, :], 
                                                                                  atm1=actions[:, t-1], 
                                                                                  rtm1=rewards[:, t-1]
                                                                                  )
            return xs_tilde, rs_tilde                
        else:
            for t in range (1, T):
                xs_tilde[:, t, :] = self.preprocess_single_step(z=zs, 
                                                                xt=xs[:, t, :], 
                                                                xtm1=xs[:, t-1, :], 
                                                                atm1=actions[:, t-1]
                                                                )
            return xs_tilde