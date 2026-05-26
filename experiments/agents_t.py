import numpy as np
import torch
import copy

#from core.utils import glogger
#from core.base_models import NeuralNet
from pycfrl.utils.base_models import NeuralNet
from sklearn.model_selection import KFold


class OfflineFQI_NN:
    def __init__(self, action_size, hidden_dims, device) -> None:
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.device = device

    def fit(
        self,
        states,
        actions,
        rewards,
        max_iter,
        inner_iter=500,
        learning_rate=0.1,
        discount_factor=0.9,
        early_stopping_fqi_loss=1e8,
    ):
        #torch.set_num_threads(1) # ORIGINALLY COMMENTED OUT

        # convenience variables
        N, T = actions.shape
        sdim = states.shape[-1]

        # standarize states
        # self.states_mean = np.mean(states, axis=(0, 1), keepdims=False)
        # self.states_std = np.std(states, axis=(0, 1), keepdims=False)
        # states = (states - self.states_mean) / (self.states_std + 1e-8)

        # # standardize rewards
        # self.rewards_mean = np.mean(rewards, axis=(0, 1), keepdims=False)
        # self.rewards_std = np.std(rewards, axis=(0, 1), keepdims=False)
        # rewards = (rewards - self.rewards_mean) / (self.rewards_std + 1e-8)

        # reshape data
        states_tensor = torch.tensor(
            states[:, :-1, :].reshape(-1, states.shape[-1]), dtype=torch.float32
        ).to(self.device)
        # normalize states_ten
        next_states_tensor = torch.tensor(
            states[:, 1:, :].reshape(-1, states.shape[-1]), dtype=torch.float32
        ).to(self.device)
        actions_tensor = torch.tensor(
            actions.flatten().reshape(-1, 1), dtype=torch.int64
        ).to(self.device)
        rewards_tensor = torch.tensor(rewards.flatten(), dtype=torch.float32).to(
            self.device
        )

        # init model
        current_model = NeuralNet(
            in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
        ).to(self.device)
        old_model = copy.deepcopy(current_model)
        # optimizer = torch.optim.Adam(current_model.parameters(), lr=learning_rate)
        # training loop
        for i in range(max_iter):
            # generate target
            current_model = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
            ).to(self.device)
            optimizer = torch.optim.Adam(current_model.parameters(), lr=learning_rate)
            with torch.no_grad():
                if i == 0:
                    Y = rewards_tensor.unsqueeze(1)
                else:
                    next_q_values = old_model.forward(next_states_tensor)
                    max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
                    Y = (
                        rewards_tensor.unsqueeze(1)
                        + discount_factor * max_next_q_values.detach()
                    )
            current_model.train()
            for _ in range(inner_iter):
                q_values_all_actions = current_model.forward(states_tensor)
                Y_pred = q_values_all_actions.gather(1, actions_tensor)

                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(Y, Y_pred)
                loss.backward()
                optimizer.step()
            old_model = copy.deepcopy(current_model)
            '''glogger.debug(
                "{}, fqi_nn mse:{}, mean_target:{}, lr:{}, ".format(
                    i, loss.item(), np.mean(Y.detach().cpu().numpy()), learning_rate
                )
            )'''
            # if loss.item() > 10:
            if loss.item() > early_stopping_fqi_loss:
                '''glogger.warning(
                    "FQI NN diverged, stop training, loss: {}, iter: {}".format(
                        loss.item(), i
                    )
                )'''
                break

        self.model = copy.deepcopy(current_model)
        return loss.item()

    def act(self, states):
        self.model.eval()
        states = states.reshape(states.shape[0], -1)
        # states = (states - self.states_mean) / (self.states_std + 1e-8)
        x = torch.tensor(states, dtype=torch.float32).to(self.device)
        tmp = self.model.forward(x).detach().cpu().numpy()

        return np.argmax(tmp, axis=1)


class FQI:
    def __init__(
        self,
        preprocessor,
        model_type, 
        num_actions,
        state_size=1,
        #action_size=1,
        hidden_dims=[32],
        name='test',
        device="cpu",
    ):
        self.preprocessor = preprocessor
        self.state_size = state_size
        #self.action_size = action_size
        self.action_size = num_actions
        self.name = name

        self.agent = OfflineFQI_NN(
            action_size=self.action_size, hidden_dims=hidden_dims, device=device
        )

    def train(
        self,
        xs,
        zs,
        actions,
        rewards,
        max_iter,
        inner_iter=500,
        learning_rate=0.1,
        discount_factor=0.9,
    ):
        N, T = xs.shape[:2]

        states_p = np.zeros((N, T, self.state_size))
        rewards_p = np.zeros((N, T - 1))
        '''if hasattr(self.preprocessor, "reset_buffer"):
            self.preprocessor.reset_buffer(N)'''

        # ORIGINAL
        '''for t in range(T):
            xt = xs[:, t]
            if t == 0:
                states_p[:, t] = self.preprocessor.preprocess(
                    xt=xt, xtm1=None, z=z, atm1=None, rtm1=None, t=t
                )
            else:
                xtm1 = xs[:, t - 1]
                atm1 = actions[:, t - 1]
                rtm1 = rewards[:, t - 1]
                states_p[:, t], rewards_p[:, t - 1] = self.preprocessor.preprocess(
                    xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=rtm1, t=t
                )'''
        
        # NEWLY ADDED
        if hasattr(self.preprocessor, "xs_tilde") and hasattr(self.preprocessor, "rs_tilde") and self.preprocessor.xs_tilde is not None and self.preprocessor.rs_tilde is not None:
            states_p = self.preprocessor.xs_tilde
            rewards_p = self.preprocessor.rs_tilde
        else:
            for t in range(T):
                xt = xs[:, t]
                if t == 0:
                    states_p[:, t] = self.preprocessor.preprocess_single_step(
                        xt=xt, xtm1=None, z=zs, atm1=None, rtm1=None
                    )
                else:
                    xtm1 = xs[:, t - 1]
                    atm1 = actions[:, t - 1]
                    rtm1 = rewards[:, t - 1]
                    states_p[:, t], rewards_p[:, t - 1] = self.preprocessor.preprocess_single_step(
                        xt=xt, xtm1=xtm1, z=zs, atm1=atm1, rtm1=rtm1
                    )


        num_retries = 0
        while num_retries < 10:
            self.final_fqi_loss = self.agent.fit(
                states_p,
                actions,
                rewards_p,
                max_iter,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            if self.final_fqi_loss <= 2:
                break
            num_retries += 1
        return self.final_fqi_loss

    def act(self, xt, z, xtm1, atm1, uat=None, is_return_prob=False, **kwargs):
        states = self.preprocessor.preprocess_single_step(
            xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None
        )

        probs = np.zeros((xt.shape[0], self.action_size))
        actions = self.agent.act(states)
        probs[np.arange(len(actions)), actions] = 1.0
        if is_return_prob:
            return probs
        else:
            return actions
