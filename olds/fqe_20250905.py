import numpy as np
import torch
import copy
from .utils.base_models import LinearRegressor, NeuralNet
from .utils.base_models import DecreasingLossWarning, FluctuatingQValueWarning
from .utils.base_models import LossConvergenceChecker, QValueConvergenceChecker
from .utils.custom_errors import InvalidModelError
from .agents import Agent
from sklearn.model_selection import train_test_split
#from my_utils import glogger
from typing import Literal, Callable
from tqdm import tqdm
import warnings



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



class FQE:
    """
    Implementation of the fitted Q evaluation (FQE) algorithm. 

    FQE can be used to estimate the value of a policy using offline data.
    """
    
    def __init__(
            self, 
            num_actions: int, 
            policy: Agent, 
            model_type: Literal["lm", "nn"], 
            hidden_dims: list[int] = [32], 
            learning_rate: int | float = 0.1, 
            epochs: int = 500, 
            gamma: int | float = 0.9, 
            is_loss_monitored: bool = True,
            is_early_stopping_nn: bool = True,
            test_size_nn: int | float = 0.2,
            patience_nn: int = 10,
            min_delta_nn: int | float = 0.005,
            is_q_monitored: bool = True,
            is_early_stopping_q: bool = True,
            patience_q: int = 5,
            min_delta_q: int | float = 0.005
        ) -> None:
        """
        Args: 
            num_actions (int): 
                The total number of legit actions. 
            policy (Agent): 
                The policy to be evaluated. 
            model_type (str): 
                The type of the model used for learning the Q function. Can 
                be "lm" (polynomial regression) or "nn" (neural network). 
                *Currently, only 'nn' is supported.*
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
        """
        
        if model_type == 'nn':
            self.model_type = model_type
        else:
            raise InvalidModelError("Invalid model type. Only 'nn' is currently supported.")
        self.action_size = num_actions
        self.policy = policy
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.is_loss_monitored = is_loss_monitored
        self.is_early_stopping_nn = is_early_stopping_nn
        self.test_size_nn = test_size_nn
        self.patience_nn = patience_nn
        self.min_delta_nn = min_delta_nn
        self.is_q_monitored = is_q_monitored
        self.is_early_stopping_q = is_early_stopping_q
        self.patience_q = patience_q
        self.min_delta_q = min_delta_q

        # sanity check
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def _get_actions(
            self, 
            zs: np.ndarray, 
            states: np.ndarray, 
            actions: np.ndarray, 
            uat: np.ndarray | None = None
        ) -> np.ndarray:
        xs = np.array(states)
        del(states)
        zs = np.array(zs)
        actions = np.array(actions)
        if uat is not None:
            uat = np.array(uat)
        N, T, _ = xs.shape  # xs + 1
        
        p = copy.deepcopy(self.policy) # use a deepcopy to preserve the info in original policy
        actions_taken = np.zeros([N, T])
        for t in range(T): # MIGHT NEED TO CHANGE TO T - 1
            if t == 0:
                actions_taken[:, 0] = p.act(z=zs, 
                                            xt=xs[:, 0], 
                                            xtm1=None, 
                                            atm1=None, 
                                            uat=uat)
            else:
                actions_taken[:, t] = p.act(z=zs, 
                                            xt=xs[:, t], 
                                            xtm1=xs[:, t - 1], 
                                            atm1=actions[:, t - 1], 
                                            uat=uat)

        return actions_taken[:, 1:].reshape(N *(T - 1), -1)

    def fit(
            self, 
            zs: list | np.ndarray, 
            states: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray, 
            max_iter: int = 1000, 
            f_ua: Callable[[int], int] = f_ua_default
        ) -> None:
        """
        Fit the FQE. 
        
        Args:
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the training data. It should be a 2D list or array following the Sensitive 
                Attributes Format.
            states (list or np.ndarray): 
                The state trajectory used for training. It should be 
                a 3D list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory used for training, often generated using a behavior 
                policy. It should be a 2D list or array following the Full-trajectory Actions 
                Format.
            rewards (list or np.ndarray): 
                The reward trajectory used for training. It should be 
                a 2D list or array following the Full-trajectory Rewards Format.
            max_iter (int, optional): 
                The number of iterations for learning the Q function. 
            f_ua (Callable, optional): 
                A rule to generate exogenous variables for each individual's 
                actions during training. It should be a function whose argument list, argument names, 
                and return type exactly match those of :code:`f_ua_default`. 
        """
        
        torch.set_num_threads(1)
        # convenience variables
        xs = np.array(states)
        del(states)
        zs = np.array(zs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        N, T = actions.shape
        sdim = xs.shape[-1] + zs.shape[-1]

        xs_ = copy.deepcopy(xs)
        zs_ = copy.deepcopy(zs)
        actions_ = copy.deepcopy(actions)
        rewards_ = copy.deepcopy(rewards)

        # reshape data
        next_states = np.concatenate(
            [xs_[:, 1:, :], np.repeat(zs_.reshape(-1, 1, zs.shape[-1]), axis=1, repeats=T)], 
            axis=2
        ).reshape(-1, sdim)
        states = np.concatenate(
            [xs_[:, :-1, :], np.repeat(zs_.reshape(-1, 1, zs.shape[-1]), axis=1, repeats=T)],
            axis=2, # TRY TO DEAL WITH DIMENSION PROBLEMS OF ZS AND STATES: COMPLETED!
        ).reshape(-1, sdim)
        actions = actions[:, :].flatten()
        rewards = rewards[:, :].flatten()

        # init model
        if self.model_type == "lm":
            current_model = [None for _ in range(self.action_size)]
            for i in tqdm(range(max_iter)):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    tmp = np.zeros([N * T, self.action_size])
                    for a in range(self.action_size):
                        tmp[:, a] = current_model[a].predict(next_states).flatten()
                    # selected_actions = self.policy.act(
                    #     uat=np.random.uniform(size=states.shape[0]), states=next_states
                    # ).flatten()
                    uat = f_ua(N=N)
                    selected_actions = self._get_actions(zs_, xs_, actions_, uat).flatten()
                    Y = (
                        rewards + self.gamma * tmp[np.arange(tmp.shape[0]), selected_actions.astype(int)]
                    ).reshape(-1, 1)

                # generate input
                new_model = [
                    LinearRegressor(
                        featurize_method="polynomial",
                        degree=2, 
                        interaction_only=False,
                        is_standarized=False,
                    )
                    for i in range(self.action_size)
                ]
                for a in range(self.action_size):
                    idx = actions == a
                    X_a = states[idx]
                    Y_a = Y[idx]
                    new_model[a].fit(X_a, Y_a)
                #glogger.info(
                #    "{}, fqe_lm mse:{}, mean_target:{}".format(
                #        i, np.mean([m.mse for m in new_model]), np.mean(Y)
                #    )
                #)
                current_model = copy.deepcopy(new_model)

        elif self.model_type == "nn":
            #np.random.seed(10) # NEWLY ADDED
            #torch.manual_seed(10) # NEWLY ADDED
            current_model = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims 
            )

            q_checker = QValueConvergenceChecker(patience=self.patience_q, 
                                             min_delta=self.min_delta_q)

            # training loop
            for i in tqdm(range(max_iter)):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    current_model.eval()
                    tmp = (
                        current_model.forward(
                            torch.tensor(next_states, dtype=torch.float32)
                        )
                        .detach()
                        .numpy()
                    )
                    uat = f_ua(N=N)
                    selected_actions = self._get_actions(zs_, xs_, actions_, uat).reshape(
                        -1, 1
                    )

                    Y = (
                        rewards.reshape(-1, 1)
                        + self.gamma * np.take_along_axis(tmp, selected_actions.astype(int), axis=1)
                    ).reshape(-1, 1)

                # generate input
                X = states

                # train model
                #np.random.seed(10) # NEWLY ADDED
                #torch.manual_seed(10) # NEWLY ADDED
                new_model = NeuralNet(
                    in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
                )
                #Y = torch.tensor(Y, dtype=torch.float32)

                if self.is_loss_monitored or self.is_early_stopping_nn:
                    convergence_checker = LossConvergenceChecker(
                        patience=self.patience_nn,
                        min_delta=self.min_delta_nn,
                        mode="min",
                    )

                    idx_train, idx_test = train_test_split(
                        np.arange(states.shape[0]), test_size=self.test_size_nn
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

                if self.is_loss_monitored or self.is_early_stopping_nn:
                    actions_tensor_train = torch.LongTensor(actions[idx_train].reshape(-1, 1))
                    actions_tensor_test = torch.LongTensor(actions[idx_test].reshape(-1, 1))
                else:
                    actions_tensor_train = torch.LongTensor(actions.reshape(-1, 1))

                optimizer = torch.optim.Adam(new_model.parameters(), lr=self.lr)
                criterion = torch.nn.MSELoss()

                train_losses = []
                val_losses = []

                for epoch in range(self.epochs):
                    new_model.train()
                    Y_pred = new_model.forward(X_train).gather(1, actions_tensor_train)
                    optimizer.zero_grad()
                    loss = criterion(Y_pred, y_train)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

                    if self.is_loss_monitored or self.is_early_stopping_nn:
                        new_model.eval()
                        with torch.no_grad():
                            val_outputs = new_model(X_test).gather(1, actions_tensor_test)
                            val_loss = criterion(val_outputs, y_test)

                        val_losses.append(val_loss.item())
                        converged_nn = convergence_checker(val_loss.item())

                        if self.is_early_stopping_nn and converged_nn:
                            break

                        if self.is_loss_monitored and epoch == self.epochs - 1 and (not converged_nn):
                            warnings.warn('\nThe decrease in the loss is not small enough in at least one of the final ' + str(self.patience_nn) + ' epochs during neural network training', DecreasingLossWarning)

                #glogger.info(
                #    "{}, fqe_nn mse:{}, mean_target:{}".format(
                #        i, loss.item(), np.mean(Y.detach().numpy())
                #   )
                #)
                current_model = copy.deepcopy(new_model)

                if self.is_early_stopping_q or self.is_q_monitored:
                    current_model.eval()
                    with torch.no_grad():
                        q = current_model(torch.FloatTensor(states)).numpy()
                    current_model.train()
                    converged_q = q_checker(q=q)
                    
                    if self.is_early_stopping_q and converged_q:
                        break
                    
                    if self.is_q_monitored and i == max_iter - 1 and (not converged_q):
                        warnings.warn('\nThe fluctuation in the q values is not small enough in at least one of the final ' + str(self.patience_q) + ' iterations during FQI training', FluctuatingQValueWarning)

        self.model = copy.deepcopy(current_model)

    def evaluate(
            self, 
            zs: list | np.ndarray, 
            states: list | np.ndarray, 
            actions: list | np.ndarray, 
            f_ua: Callable[[int], int] = f_ua_default
        ) -> np.ndarray:
        """
        Estimate the value of the policy. 

        It uses the FQE algorithm and the input offline trajectory to evaluate the policy of 
        interest.

        Args:
            zs (list or np.ndarray): 
                The observed sensitive attributes of each individual 
                in the offline trajectory used for evaluation. It should be a 2D list or array 
                following the Sensitive Attributes Format.
            states (list or np.ndarray): 
                The state trajectory used for evaluation. It should be 
                a 3D list or array following the Full-trajectory States Format.
            actions (list or np.ndarray): 
                The action trajectory used for evaluation, often generated using a behavior 
                policy. It should be a 2D list or array following the Full-trajectory Actions Format. 
            f_ua (Callable, optional): 
                A rule to generate exogenous variables for each individual's 
                actions during evaluation. It should be a 2D function whose argument list, argument 
                names, and return type exactly match those of `f_ua_default`. 
        
        Returns: 
            Y (np.ndarray): 
                A vector containing multiple estimates of the value of the policy of 
                interest. It is an array with shape (N*T, ) where N is the number of individuals in 
                the input offline trajectory and T is the total number of transitions in the input 
                offline trajectory.
        """
        
        xs = np.array(states)
        del(states)
        zs = np.array(zs)
        actions = np.array(actions)
        sdim = xs.shape[-1] + zs.shape[-1]
        T = actions.shape[1]
        uat = f_ua(N=zs.shape[0])
        actions_taken = self._get_actions(zs, xs, actions, uat)
        xs_ = copy.deepcopy(xs)
        zs_ = copy.deepcopy(zs)

        states = np.concatenate(
            [xs_[:, 1:, :], np.repeat(zs_.reshape(-1, 1, zs.shape[-1]), axis=1, repeats=T)], axis=2
        ).reshape(
            -1, sdim
        )  # use 1: instead of 0: since _get_actions is implemented using next_state

        #np.random.seed(10) # NEWLY ADDED
        #torch.manual_seed(10) # NEWLY ADDED
        if self.model_type == "lm":
            tmp = np.zeros([states.shape[0], self.action_size])
            for a in range(self.action_size):
                tmp[:, a] = (
                    self.model[a].predict(states.reshape(states.shape[0], -1)).flatten()
                )
            return np.take_along_axis(
                tmp, actions_taken.reshape(-1, 1).astype(int), axis=1
            ).flatten()
        elif self.model_type == "nn":
            self.model.eval()
            X = torch.tensor(states, dtype=torch.float32)
            actions_taken = torch.tensor(
                actions_taken.reshape(-1, 1), dtype=torch.int64
            )
            Y = self.model.forward(X).gather(1, actions_taken)
            return Y.detach().numpy().flatten()