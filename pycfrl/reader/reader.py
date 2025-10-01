import pandas as pd
import numpy as np

def _read_trajectory_helper(
        data: pd.DataFrame, 
        z_labels: list[str], 
        state_labels: list[str], 
        action_label: str, 
        reward_label: str, 
        id_label: str, 
        T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # read some basic information
    ids = pd.unique(data[id_label])
    N = ids.shape[0]
    T = T + 1

    # read trajectory data
    z_labels = np.array(z_labels)
    state_labels = np.array(state_labels)
    zs = np.zeros((N, z_labels.shape[0]))
    xs = np.zeros((N, T, state_labels.shape[0]))
    actions = np.zeros((N, T-1))
    rewards = np.zeros((N, T-1))

    '''for i, id in enumerate(ids):
        subdata = data[data[id_label] == id].iloc[0:T]
        for k in range(z_labels.shape[0]): # populate sensitive attributes
            zs[i][k] = subdata[z_labels[k]].iat[0]
        for k in range(state_labels.shape[0]): # populate states at t = 0
            xs[i][0][k] = subdata[state_labels[k]].iat[0]
        for t in range(1, T):
            actions[i][t-1] = subdata[action_label].iat[t-1] # populate actions
            rewards[i][t-1] = subdata[reward_label].iat[t-1] # populate rewards
            for k in range(state_labels.shape[0]): # populate states
                xs[i][t][k] = subdata[state_labels[k]].iat[t]'''
    for i, id in enumerate(ids):
        subdata = data[data[id_label] == id].iloc[0:T]
        for k in range(z_labels.shape[0]): # populate sensitive attributes
            zs[i][k] = subdata[z_labels[k]].iat[0]
        for k in range(state_labels.shape[0]): # populate states at t = 0
            xs[i][0][k] = subdata[state_labels[k]].iat[0]
        for t in range(1, T):
            actions[i][t-1] = subdata[action_label].iat[t] # populate actions
            rewards[i][t-1] = subdata[reward_label].iat[t] # populate rewards
            for k in range(state_labels.shape[0]): # populate states
                xs[i][t][k] = subdata[state_labels[k]].iat[t]
    
    ids = pd.unique(data[id_label]).reshape(-1, 1)
    return zs, xs, actions, rewards, ids



# T is the number of action steps
def read_trajectory_from_csv(
        path: str, 
        z_labels: list[str], 
        state_labels: list[str], 
        action_label: str, 
        reward_label: str, 
        id_label: str, 
        T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read tabular trajectory data from a :code:`.csv` file into Trajectory Arrays. 

    The :code:`.csv` file must be in the format specified in the "Tabular Trajectory Data" document of 
    the "Inputs and Outputs" section.

    Args: 
        path (str): 
            The path to the :code:`.csv` file.
        z_labels (list[str]): 
            A list of strings representing the labels of columns in the :code:`.csv` file 
            that contain sensitive attribute variables.
        state_labels (list[str]): 
            A list of strings representing the labels of columns in the :code:`.csv` 
            file that contain state variables. 
        action_label (str): 
            The label of the column in the :code:`.csv` file that contains actions. 
        reward_label (str): 
            The label of the column in the :code:`.csv` file that contains rewards. 
        id_label (str): 
            The label of the column in the :code:`.csv` file that contains IDs. 
        T: 
            The total number of transitions to be read for each individual. 
    
    Returns: 
        zs (np.ndarray): 
            The observed sensitive attributes of each individual in the 
            trajectory read from the :code:`.csv` file. It is an array following the Sensitive 
            Attributes Format.
        xs (np.ndarray): 
            The state trajectory read from the :code:`.csv` file. It is an array following 
            the Full-trajectoriy States Format.
        actions (np.ndarray): 
            The action trajectory read from the :code:`.csv` file. It is an array 
            following the Full-trajectory Actions Format.
        rewards (np.ndarray): 
            The reward trajectory read from the :code:`.csv` file. It is an array 
            following the Full-trajectory Rewards Format. 
        ids (np.ndarray): 
            The ids of each individual in the trajectory read from the :code:`.csv` file. 
            It is an array with size (N, 1) where N is the number of individuals in the trajectory. 
            Specifically, :code:`ids[i, 0]` contains the ID of the i-th individual in the trajectory.
    """
    
    # import trajectory data
    data = pd.read_csv(path)

    # read trajectory data
    return _read_trajectory_helper(data, z_labels, state_labels, action_label, reward_label, 
                                        id_label, T)



# T is the number of action steps
def read_trajectory_from_dataframe(
        data: pd.DataFrame, 
        z_labels: list[str], 
        state_labels: list[str], 
        action_label: str, 
        reward_label: str, 
        id_label: str, 
        T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read tabular trajectory data from a `pandas.DataFrame` into Trajectory Arrays. 

    The dataframe must be in the format specified in the "Tabular Trajectory Data" document of 
    the "Inputs and Outputs" section.

    Args: 
        data (pandas.DataFrame): 
            The dataframe to read data from.
        z_labels (list[str]): 
            A list of strings representing the labels of columns in the :code:`.csv` file 
            that contain sensitive attribute variables.
        state_labels (list[str]): 
            A list of strings representing the labels of columns in the :code:`.csv` 
            file that contain state variables. 
        action_label (str): 
            The label of the column in the :code:`.csv` file that contains actions. 
        reward_label (str): 
            The label of the column in the :code:`.csv` file that contains rewards. 
        id_label (str): 
            The label of the column in the :code:`.csv` file that contains IDs. 
        T: 
            The total number of transitions to be read for each individual. 
    
    Returns: 
        zs (np.ndarray): 
            The observed sensitive attributes of each individual in the 
            trajectory read from the :code:`.csv` file. It is an array following the Sensitive 
            Attributes Format.
        xs (np.ndarray): 
            The state trajectory read from the :code:`.csv` file. It is an array following 
            the Full-trajectoriy States Format.
        actions (np.ndarray): 
            The action trajectory read from the :code:`.csv` file. It is an array 
            following the Full-trajectory Actions Format.
        rewards (np.ndarray): 
            The reward trajectory read from the :code:`.csv` file. It is an array 
            following the Full-trajectory Rewards Format. 
        ids (np.ndarray): 
            The IDs of each individual in the trajectory read from the :code:`.csv` file. 
            It is an array with size (N, 1) where N is the number of individuals in the trajectory. 
            Specifically, :code:`ids[i, 0]` contains the ID of the i-th individual in the trajectory.
    """

    # read trajectory data
    return _read_trajectory_helper(data, z_labels, state_labels, action_label, reward_label, 
                                        id_label, T)



def convert_trajectory_to_dataframe(
        zs: list | np.ndarray, 
        states: list | np.ndarray, 
        actions: list | np.ndarray, 
        rewards: list | np.ndarray, 
        ids: list | np.ndarray, 
        z_labels: list[str] | None = None, 
        state_labels: list[str] | None = None, 
        action_label: str | None = None, 
        reward_label: str | None = None, 
        id_label: str | None = None, 
        T_label: str | None = None
    ) -> pd.DataFrame:
    """
    Convert trajectories from Trajectory Arrays into a :code:`pandas.DataFrame`.

    The output dataframe follows the format specified in the "Tabular Trajectory Data" document 
    of the "Inputs and Outputs" section. The output dataframe will also contain a column that 
    records the time steps of each row, where the time steps start from 1.

    Args: 
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory. It should be a list or array following the Sensitive Attributes 
            Format.
        states (list or np.ndarray): 
            The state trajectory. It should be a list or array following 
            the Full-trajectory States Format.
        actions (list or np.ndarray): 
            The action trajectory. It should be a list or array following 
            the Full-trajectory Actions Format.
        rewards (list or np.ndarray): 
            The reward trajectory. It should be a list or array following 
            the Full-trajectory Actions Format. 
        ids (list or np.ndarray): 
            The IDs of each individual in the trajectory. It is an array with 
            size (N, 1) where N is the number of individuals in the trajectory. Specifically, 
            :code:`ids[i, 0]` contains the ID of the i-th individual in the trajectory. 
        z_labels (list[str], optional): 
            A list of strings representing the labels of columns in the  
            output dataframe that should contain sensitive attribute variables. When set to :code:`None`, 
            the function will use default names `z1, z2, ...` That is, the i-th component of the 
            sensitive attribute vector will be named `zi`.
        state_labels (list[str]): 
            A list of strings representing the labels of columns in the output 
            dataframe that should contain state variables. When set to :code:`None`, the function will 
            use default names `state1, state2, ...` That is, the i-th component of the state vector 
            will be named `statei`.
        action_label (str): 
            The label of the column in the output dataframe that should contain actions. 
            When set to :code:`None`, the column that contains actions will be named `action` by default. 
        reward_label (str): 
            The label of the column in the output dataframe that should contain rewards. 
            When set to :code:`None`, the column that contains rewards will be named `reward` by default. 
        id_label (str): 
            The label of the column in output dataframe that should contain IDs.  When set to 
            :code:`None`, the column that contains IDs will be named `ID` by default. 
        T_label (str): 
            The label of the column in output dataframe that should contain time steps.  When set 
            to :code:`None`, the column that contains time steps will be named `timestamp` by default. 
    
    Returns: 
        data (pandas.DataFrame): 
            A dataframe that contains the input trajectories in the format 
            specified in the "Tabular Trajectory Data" document of the "Inputs and Outputs" section.
    """
    
    N = states.shape[0]
    T = states.shape[1]
    zs = np.array(zs)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    ids = np.array(ids).flatten()

    # define labels of the dataframe
    if z_labels is None:
        z_labels = []
        for i in range(zs.shape[-1]):
            z_labels.append('z' + str(i+1)) # default naming: z1, z2, ...
        z_labels = np.array(z_labels)
    else:
        z_labels = np.array(z_labels)

    if state_labels is None:
        state_labels = []
        for i in range(states.shape[-1]):
            state_labels.append('state' + str(i+1)) # default naming: state1, state2, ...
        state_labels = np.array(state_labels)
    else: 
        state_labels = np.array(state_labels)

    if action_label is None:
        action_label = 'action'

    if reward_label is None:
        reward_label = 'reward'

    if id_label is None:
        id_label = 'ID'

    if T_label is None:
        T_label = 'timestamp'

    # reorganize data
    ids_r = np.repeat(ids, repeats=T).reshape(-1, 1)
    times_r = np.tile(np.arange(1, T+1), reps=N).reshape(-1, 1)
    zs_r = np.repeat(zs, repeats = T, axis=0)
    states_r = states.reshape(N*T, states.shape[-1])
    #actions_r = np.concatenate((actions, np.tile(np.array([[np.nan]]), reps=(N, 1))), axis=1)
    actions_r = np.concatenate((np.tile(np.array([[np.nan]]), reps=(N, 1)), actions), axis=1)
    actions_r = actions_r.reshape(N*T, 1)
    #rewards_r = np.concatenate((rewards, np.tile(np.array([[np.nan]]), reps=(N, 1))), axis=1)
    rewards_r = np.concatenate((np.tile(np.array([[np.nan]]), reps=(N, 1)), rewards), axis=1)
    rewards_r = rewards_r.reshape(N*T, 1)
    data_arr = np.concatenate((ids_r, times_r, zs_r, actions_r, rewards_r, states_r), axis=1)

    # create the trajectory dataframe
    labels = np.concatenate((np.array([id_label]), 
                                np.array([T_label]), 
                                z_labels, 
                                np.array([action_label]), 
                                np.array([reward_label]), 
                                state_labels), 
                                axis=0)
    data = pd.DataFrame(data_arr, columns=labels)

    return data



def export_trajectory_to_csv(
        path: str, 
        zs: list | np.ndarray, 
        states: list | np.ndarray, 
        actions: list | np.ndarray, 
        rewards: list | np.ndarray, 
        ids: list | np.ndarray, 
        z_labels: list[str] | None = None, 
        state_labels: list[str] | None = None, 
        action_label: str | None = None, 
        reward_label: str | None = None, 
        id_label: str | None = None, 
        T_label: str | None = None, 
        **to_csv_kwargs
    ) -> None:
    """
    Export trajectories from Trajectory Arrays to a :code:`.csv` file.

    The output :code:`.csv` follows the format specified in the "Tabular Trajectory Data" document 
    of the "Inputs and Outputs" section. The output :code:`.csv` will also contain a column that 
    records the time steps of each row, where the time steps start from 1.

    Args: 
        path: 
            The path that specifies where the trajectory should be exported. 
        zs (list or np.ndarray): 
            The observed sensitive attributes of each individual 
            in the trajectory. It should be a list or array following the Sensitive Attributes 
            Format.
        states (list or np.ndarray): 
            The state trajectory. It should be a list or array following 
            the Full-trajectory States Format.
        actions (list or np.ndarray): 
            The action trajectory. It should be a list or array following 
            the Full-trajectory Actions Format.
        rewards (list or np.ndarray): 
            The reward trajectory. It should be a list or array following 
            the Full-trajectory Actions Format. 
        ids (list or np.ndarray): 
            The IDs of each individual in the trajectory. It is an array with 
            size (N, 1) where N is the number of individuals in the trajectory. Specifically, 
            :code:`ids[i, 0]` contains the ID of the i-th individual in the trajectory. 
        z_labels (list[str], optional): 
            A list of strings representing the labels of columns in the output dataframe that 
            should contain sensitive attribute variables. When set to :code:`None`, the 
            function will use default names `z1, z2, ...` That is, the i-th component of the 
            sensitive attribute vector will be named `zi`.
        state_labels (list[str]): 
            A list of strings representing the labels of columns in the output dataframe 
            that should contain state variables. When set to :code:`None`, the function will use default 
            names `state1, state2, ...` That is, the i-th component of the state vector will be named 
            `statei`.
        action_label (str): 
            The label of the column in the output dataframe that should contain actions. 
            When set to :code:`None`, the column that contains actions will be named `action` by default. 
        reward_label (str): 
            The label of the column in the output dataframe that should contain rewards. 
            When set to :code:`None`, the column that contains rewards will be named `reward` by default. 
        id_label (str): 
            The label of the column in output dataframe that should contain IDs.  When set to 
            :code:`None`, the column that contains IDs will be named `ID` by default. 
        T_label (str): 
            The label of the column in output dataframe that should contain time steps.  When set 
            to :code:`None`, the column that contains time steps will be named `timestamp` by default. 
        **to_csv_kwargs: 
            Additional arguments that specifies details about the expored :code:`.csv` file. 
            These arguments can be any legit arguments to the :code:`pandas.to_csv()` function, and 
            they will serve the same functionalities as they do for the :code:`pandas.to_csv()` function. 
    """

    data = convert_trajectory_to_dataframe(zs, states, actions, rewards, ids, z_labels, state_labels, 
                                           action_label, reward_label, id_label, T_label)
    data.to_csv(path, **to_csv_kwargs)





