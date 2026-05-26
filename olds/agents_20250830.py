from abc import abstractmethod
import numpy as np
import torch
import copy
#from utils.utils import glogger
from .utils.base_models import LinearRegressor, NeuralNet, LossConvergenceChecker, DecreasingLossWarning
from .utils.custom_errors import InvalidModelError
from .preprocessor import Preprocessor, SequentialPreprocessor
from sklearn.model_selection import train_test_split
from typing import Literal
from tqdm import tqdm
import warnings

class Agent:
    """
    Base class for reinforcement learning agents.

    Subclasses must implement the :code:`act` method.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            xtm1: list | np.ndarray | None = None, 
            atm1: list | np.ndarray | None = None, 
            uat: list | np.ndarray | None = None, 
            **kwargs
    ) -> np.ndarray:
        """
        An abstract prototype of methods for making decisions using the agent.

        Args: 
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                for whom the decisions are to be made. It should be a 2D list or array following 
                the Sensitive Attributes Format.
            xt (list or np.ndarray): 
                The states at the current time step of each individual for whom the decisions 
                are to be made. It should be a 2D list or array following the Single-time States 
                Format.
            xtm1 (list or np.ndarray, optional): 
                The states at the previous time step of each individual  
                for whom the decisions are to be made. It should be a 2D list or array following the 
                Single-time States Format.
            atm1 (list or np.ndarray, optional): 
                The actions at the previous time step of each individual  
                for whom the decisions are to be made. It should be a 1D list or array following the 
                Single-time Actions Format. When both :code:`xtm1` and :code:`atm1` are set to 
                :code:`None`, the agent will consider the input to be from the initial time step of a 
                new trajectory, and the internal preprocessor be reset if it is an instance of 
                :code:`SequentialPreprocessor`.
            uxt (list or np.ndarray, optional): 
                The exogenous variables for each 
                individual's action. It should be a 2D list or array with shape (N, 1) 
                where N is the total number of individuals.

        Returns: 
            actions (np.ndarray): 
                The decisions made for the individuals. It is a 1D array following the 
                Single-time Actions Format.
        """

        pass



class FQI(Agent):
    """
    Implementation of the fitted Q-iteration (FQI) algorithm. 

    FQI can be used to learn the optimal policy from offline data. 

    In particular, for an :code:`FQI` object, users can specify whether to add a preprocessor 
    internally. If a preprocessor is added internally, then the :code:`FQI` object will preprocess 
    the input data before using the data for training (:code:`train()` method) and decision-making 
    (:code:`act()` method). 

    References:
        .. [1] Riedmiller, M. (2005). Neural fitted q iteration-first experiences with a 
               data efficient neural reinforcement learning method. Machine Learning: ECML 
               2005: 16th European conference on machine learning, Porto, Portugal, October 
               3-7, 2005. preceedings 16', Springer, pp. 317-328.
    """
    
    def __init__(
        self,
        num_actions: int,
        model_type: Literal["nn", "lm"],
        hidden_dims: list[int] = [32],
        preprocessor: Preprocessor | None = None,
        gamma: int | float = 0.9,
        learning_rate: int | float = 0.1,
        epochs: int = 500,
        is_loss_monitored: bool = False,
        is_early_stopping: bool = False,
        test_size: int | float = 0.2,
        patience: int = 10,
        min_delta: int | float = 0.005,
    ) -> None:
        """
        Args: 
            num_actions (int): 
                The total number of legit actions. 
            model_type (str): 
                The type of the model used for learning the Q function. Can 
                be "lm" (polynomial regression) or "nn" (neural network). 
                *Currently, only 'nn' is supported.*
            hidden_dims (list[int], optional): 
                The hidden dimensions of the neural network. This 
                argument is not used if :code:`model_type="lm"`. 
            preprocessor (Preprocessor, optional): 
                A preprocessor used for preprocessing input data 
                before using the data for training or decision-making. The preprocessor must have 
                already been trained if it requires training. When set to :code:`None`, :code:`FQI` 
                will directly use the input data for training or decision-making without 
                preprocessing it.
            gamma (int or float, optional): 
                The discount factor for the cumulative discounted reward 
                in the objective function.
            learning_rate (int or float, optional): 
                The learning rate of the neural network. This 
                argument is not used if :code:`model_type="lm"`. 
            epochs (int, optional): 
                The number of training epochs for the neural network. This 
                argument is not used if :code:`model_type="lm"`. 
            is_loss_monitored (bool, optional):
                When set to :code:`True`, will split the training data into a training set and a 
                validation set, and will monitor the validation loss when training the neural network 
                approximator of the Q function in each iteration. A warning 
                will be raised if the decrease in the validation loss is greater than :code:`min_delta` for at 
                least one of the final :math:`p` epochs during neural network training, where :math:`p` is specified 
                by the argument :code:`patience`. This argument is not used if :code:`model_type="lm"`.
            is_early_stopping (bool, optional): 
                When set to :code:`True`, will split the training data into a training set and a 
                validation set, and will enforce early stopping based on the validation loss 
                when training the neural network approximator of the Q function in each iteration. That is, in each iteration, 
                neural network training will stop early 
                if the decrease in the validation loss is no greater than :code:`min_delta` for :math:`p` consecutive training 
                epochs, where :math:`p` is specified by the argument :code:`patience`. This argument is not used if 
                :code:`model_type="lm"`.
            test_size (int or float, optional): 
                An :code:`int` or :code:`float` between 0 and 1 (inclusive) that 
                specifies the proportion of the full training data that is used as the validation set for loss 
                monitoring and early stopping. This argument is not used if :code:`model_type="lm"` or 
                both :code:`is_loss_monitored` and :code:`is_early_stopping` are :code:`False`.
            patience (int, optional): 
                The number of consequentive epochs with barely-decreasing validation loss that is needed 
                for loss monitoring and early stopping. This argument is not used if :code:`model_type="lm"` 
                or both :code:`is_loss_monitored` and :code:`is_early_stopping` are :code:`False`.
            min_delta (int for float, optional): 
                The maximum amount of decrease in the validation loss for it to be considered 
                barely-decreasing by the loss monitoring and early stopping mechanisms. This argument is 
                not used if :code:`model_type="lm"` or both :code:`is_loss_monitored` and 
                :code:`is_early_stopping` are :code:`False`.
        """
        
        if model_type == 'nn':
            self.model_type = model_type
        else:
            raise InvalidModelError("Invalid model type. Only 'nn' is currently supported.")
        self.action_space = np.array([a for a in range(num_actions)]).reshape(-1, 1)
        self.action_size = num_actions
        self.hidden_dims = hidden_dims
        self.preprocessor = preprocessor
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_loss_monitored = is_loss_monitored
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.patience = patience
        self.min_delta = min_delta
        self.__name__ = 'FQI'
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def _init_model(self, state_dim: int) -> None:
        if self.model_type == "lm":
            self.model = [
                LinearRegressor(
                    featurize_method="polynomial",
                    degree=2,
                    interaction_only=False,
                    is_standarized=False,
                )
                for _ in range(self.action_size)
            ]
        elif self.model_type == "nn":
            self.model = NeuralNet(
                in_dim=state_dim,
                out_dim=self.action_size,
                hidden_dims=self.hidden_dims,
            )

    def _fit_helper(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray, 
            next_states: np.ndarray, 
            state_dim: int, 
            max_iter: int
        ) -> None:
        torch.set_num_threads(1)
        states = states.reshape(-1, state_dim)
        next_states = next_states.reshape(-1, state_dim)
        actions = actions.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        for iteration in tqdm(range(max_iter)):
            if self.model_type == "lm":
                self._fit_lm(states, actions, rewards, next_states, iteration)
            elif self.model_type == "nn":
                self._fit_nn(states, actions, rewards, next_states, iteration, state_dim)

    def _fit_nn(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray, 
            next_states: np.ndarray, 
            iteration: int, 
            state_dim: int
        ) -> None:

        if iteration == 0:
            Y = rewards
        else:
            self.model.eval()
            with torch.no_grad():
                next_q_values = self.model(torch.FloatTensor(next_states)).numpy()
            Y = (
                rewards + self.gamma * np.max(next_q_values, axis=1, keepdims=True)
            ).reshape(-1, 1)

        new_model = NeuralNet(
            in_dim=state_dim,
            out_dim=self.action_size,
            hidden_dims=self.hidden_dims,
        )

        if self.is_loss_monitored or self.is_early_stopping:
            convergence_checker = LossConvergenceChecker(
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min",
            )

            idx_train, idx_test = train_test_split(
                np.arange(states.shape[0]), test_size=self.test_size
            )
            X_train = states[idx_train]
            y_train = Y[idx_train]
            X_test = states[idx_test]
            y_test = Y[idx_test]

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        else:
            X_train, y_train = states, Y

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)

        if self.is_loss_monitored or self.is_early_stopping:
            actions_tensor_train = torch.LongTensor(actions[idx_train].reshape(-1, 1))
            actions_tensor_test = torch.LongTensor(actions[idx_test].reshape(-1, 1))
        else:
            actions_tensor_train = torch.LongTensor(actions.reshape(-1, 1))

        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            new_model.train()
            optimizer.zero_grad()
            y_pred = new_model(X_train).gather(1, actions_tensor_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if self.is_loss_monitored or self.is_early_stopping:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_test).gather(1, actions_tensor_test)
                    val_loss = criterion(val_outputs, y_test)

                val_losses.append(val_loss.item())

                if self.is_early_stopping and convergence_checker(val_loss.item()):
                    break

            if self.is_loss_monitored and epoch == self.epochs - 1 and (not convergence_checker(val_loss.item())):
                warnings.warn('\nThe decrease in the loss is not small enough in at least one of the final ' + str(self.patience) + ' epochs during neural network training', DecreasingLossWarning)
        '''glogger.info(
            f"{iteration}, fqi_nn mse:{loss.item()}, mean_target:{np.mean(Y.detach().numpy())}"
        )'''
        self.model = copy.deepcopy(new_model)

    def _fit_lm(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray, 
            next_states: np.ndarray, 
            iteration: int
        ) -> None:
        if iteration == 0:
            Y = rewards
        else:
            tmp = np.array(
                [model.predict(next_states) for model in self.model]
            ).T.squeeze()
            Y = (rewards + self.gamma * np.max(tmp, axis=1, keepdims=True)).reshape(
                -1, 1
            )

        new_model = [
            LinearRegressor(
                featurize_method="polynomial",
                degree=2,
                interaction_only=False,
                is_standarized=False,
            )
            for _ in range(self.action_size)
        ]

        for a in range(self.action_size):
            idx = actions.flatten() == a
            new_model[a].fit(states[idx], Y[idx])

        mse = np.mean([m.mse for m in new_model])
        #glogger.info(f"{iteration}, fqi_lm mse:{mse}, mean_target:{np.mean(Y)}")
        self.model = copy.deepcopy(new_model)

    def _act_helper(self, states: np.ndarray) -> np.ndarray:
        if self.model_type == "lm":
            q_values = np.array(
                [model.predict(states) for model in self.model]
            ).T.squeeze()
        elif self.model_type == "nn":
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(states)).numpy()
        return np.argmax(q_values, axis=1)
    
    def train(
            self, 
            zs: list | np.ndarray, 
            xs: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray, 
            max_iter: int = 1000, 
            preprocess: bool = True
        ) -> None:  
        """
        Train the FQI agent. 
        
        The observed sensitive attributes :code:`zs` are used only by the internal 
        preprocessor; it is not directly used during policy learning.

        Args: 
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the training data. It should be a list or array following the Sensitive 
                Attributes Format.
            xs (list or np.ndarray): 
                The state trajectory used for training. It should be 
                a list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory used for training. It should be 
                a list or array following the Full-trajectory Actions Format.
            rewards (list or np.ndarray): 
                The reward trajectory used for training. It should be 
                a list or array following the Full-trajectory Rewards Format.
            max_iter (int, optional): 
                The number of iterations for learning the Q function. 
            preprocess (bool, optional): 
                Whether to preprocess the training data before training. 
                When set to :code:`False`, the training data will not be preprocessed even if 
                :code:`preprocessor` is not :code:`None` in the constructor.
        """
        
        zs = np.array(zs)
        xs = np.array(xs)
        actions = np.array(actions)
        rewards = np.array(rewards)

        if self.preprocessor is not None and preprocess:
            states_p, rewards_p = self.preprocessor.preprocess_multiple_steps(
                    zs=zs, xs=xs, actions=actions, rewards=rewards
            )
        else:
            states_p = xs
            rewards_p = rewards
        state_dim = states_p.shape[-1]
        self._init_model(state_dim)

        next_states_p = states_p[:, 1:].copy()
        states_p = states_p[:, :-1].copy()

        self._fit_helper(states_p, actions, rewards_p, next_states_p, state_dim, max_iter)

    # FOR A PARTICULAR SEQUENCE, IT SEEMS WE HAVE TO TAKE ACTIONS SEQUENTIALLY FROM THE BEGINNING 
    # TO THE END, AND WE CANNOT SKIP ANY STEP? OTHERWISE INFO IN BUFFER IS MESSED UP. SO TAKEAWAY
    # IS WE CANNOT PREPROCESS AN ARBITRARY SINGLE STEP USIN THE PREPROCESSOR, AND WE MUST 
    # PREPROCESS A WHOLE TRAJECTORY IN ORDER?
    def act(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            xtm1: list | np.ndarray | None = None, 
            atm1: list | np.ndarray | None = None, 
            uat: list | np.ndarray | None = None, 
            preprocess: bool = True, 
            #**kwargs
        ) -> np.ndarray:
        """
        Make decisions using the `FQI` agent. 

        Important Note when the internal preprocessor is a :code:`SequentialPreprocessor`: A 
        :code:`SequentialPreprocessor` object internally stores the preprocessed counterfactual states 
        from the previous function call using a states buffer, and the stored counterfactual states will 
        be used to preprocess the inputs of the current function call. In this case, suppose :code:`act()` 
        is called on a set of transitions at time :math:`t` in some trajectory. Then, at the next call 
        of :code:`act()` for this instance of `FQI`, the transitions passed to the function must be 
        from time :math:`t+1` of the same trajectory to ensure that the buffer works correctly. 
        To preprocess another trajectory, either use another instance of :code:`FQI`, or pass the 
        initial step of the trajectory to :code:`act()` with :code:`xtm1=None` and :code:`atm1=None` to 
        reset the buffer. 

        Similar issues might also arise when the internal preprocessor is some custom preprocessor 
        that relies on buffers. 

        Args: 
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                for whom the decisions are to be made. It should be a 2D list or array following 
                the Sensitive Attributes Format.
            xt (list or np.ndarray): 
                The states at the current time step of each individual for whom the decisions 
                are to be made. It should be a 2D list or array following the Single-time States 
                Format.
            xtm1 (list or np.ndarray, optional): 
                The states at the previous time step of each individual  
                for whom the decisions are to be made. It should be a 2D list or array following the 
                Single-time States Format.
            atm1 (list or np.ndarray, optional): 
                The actions at the previous time step of each individual  
                for whom the decisions are to be made. It should be a 1D list or array following the 
                Single-time Actions Format. When both :code:`xtm1` and :code:`atm1` are set to 
                :code:`None`, the agent will consider the input to be from the initial time step of 
                a new trajectory, and the internal preprocessor be reset if it is an instance of 
                :code:`SequentialPreprocessor`.
            uxt (list or np.ndarray, optional): 
                The exogenous variables for each 
                individual's action. It should be a 2D list or array with shape (N, 1) 
                where N is the total number of individuals.

        Returns: 
            actions (np.ndarray): 
                The decisions made for the individuals. It is a 1D array following the 
                Single-time Actions Format.
        """
        
        z = np.array(z)
        xt = np.array(xt)
        if xtm1 is not None:
            xtm1 = np.array(xtm1)
        if atm1 is not None:
            atm1 = np.array(atm1)
        if uat is not None:
            uat = np.array(uat)

        if self.preprocessor is not None and preprocess:
            states = self.preprocessor.preprocess_single_step(
                xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None
            )
        else:
            states = xt
        actions = self._act_helper(states)
        return actions