from pycfrl.reader.reader import read_trajectory_from_csv, convert_trajectory_to_dataframe
import numpy as np
import pandas as pd

def test_reader_well_ordered_columns():
    zs, states, actions, rewards, ids = read_trajectory_from_csv(
                      path='./tests/sample_inputs/sample_input_1.csv', 
                      z_labels=['gender', 'race'], 
                      state_labels=['state1', 'state2', 'state3'], 
                      action_label='decision', 
                      reward_label='reward', 
                      id_label='id', 
                      T=2)
    
    zs_correct = np.array([[0, 1], 
                           [1, 0], 
                           [1, 3]])
    states_correct = np.array([[[1, 4, 6], [5, 7, 3], [6, 7, 1]], 
                               [[5, 6, 7], [2, 7, 3], [8, 10, 1]], 
                               [[6, 7, 8], [4, 5, 6], [4, 3, 6]]])
    actions_correct = np.array([[2, 2], 
                                [2, 1], 
                                [1, 2]])
    rewards_correct = np.array([[4, 3], 
                                [4, 4], 
                                [2, 9]])
    ids_correct = np.array([[1], [2], [3]])
    assert(np.array_equal(zs, zs_correct))
    assert(np.array_equal(states, states_correct))
    assert(np.array_equal(actions, actions_correct))
    assert(np.array_equal(rewards, rewards_correct))
    assert(np.array_equal(ids, ids_correct))

    df = convert_trajectory_to_dataframe(
                        zs=zs, 
                        states=states, 
                        actions=actions, 
                        rewards=rewards, 
                        ids=ids, 
                        z_labels=['gender', 'race'], 
                        state_labels=['state1', 'state2', 'state3'], 
                        action_label='decision', 
                        reward_label='reward', 
                        id_label='id'
                        )
    df_correct = pd.read_csv('./tests/sample_inputs/sample_input_1.csv')
    assert(np.array_equal(df[['id', 'race', 'gender', 'state1', 'state2', 'state3']].to_numpy(), 
                          df_correct[['id', 'race', 'gender', 'state1', 'state2', 'state3']].to_numpy()))
    assert(np.array_equal(df[['decision', 'reward']].to_numpy()[[1, 2, 4, 5, 7, 8], :], 
                          df_correct[['decision', 'reward']].to_numpy()[[1, 2, 4, 5, 7, 8], :]))

def test_reader_randomly_ordered_columns():
    zs, states, actions, rewards, ids = read_trajectory_from_csv(
                      path='./tests/sample_inputs/sample_input_2.csv', 
                      z_labels=['gender', 'race'], 
                      state_labels=['state1', 'state2', 'state3'], 
                      action_label='action', 
                      reward_label='reward', 
                      id_label='id', 
                      T=2)
    
    zs_correct = np.array([[0, 1], 
                           [1, 0], 
                           [1, 3], 
                           [0, 1], 
                           [1, 1]])
    states_correct = np.array([[[1, 2, 6], [5, 4, 3], [6, 3, 1]], 
                               [[5, 4, 7], [2, 4, 3], [8, 4, 1]], 
                               [[6, 1, 8], [4, 2, 6], [4, 9, 6]], 
                               [[10, 2, 1], [4, 2, 7], [4, 2, 1]], 
                               [[5, 4, 7], [5, 4, 7], [5, 4, 7]]])
    actions_correct = np.array([[2, 2], 
                                [2, 1], 
                                [1, 2], 
                                [1, 2], 
                                [3, 3]])
    rewards_correct = np.array([[7, 7], 
                                [7, 10], 
                                [5, 3], 
                                [8, 3], 
                                [6, 6]])
    ids_correct = np.array([['wolverine'], 
                            ['spartan'], 
                            ['gaucho'], 
                            ['bruin'], 
                            ['husky']])
    assert(np.array_equal(zs, zs_correct))
    assert(np.array_equal(states, states_correct))
    assert(np.array_equal(actions, actions_correct))
    assert(np.array_equal(rewards, rewards_correct))
    assert(np.array_equal(ids, ids_correct))

    df = convert_trajectory_to_dataframe(
                        zs=zs, 
                        states=states, 
                        actions=actions, 
                        rewards=rewards, 
                        ids=ids, 
                        z_labels=['gender', 'race'], 
                        state_labels=['state1', 'state2', 'state3'], 
                        action_label='action', 
                        reward_label='reward', 
                        id_label='id'
                        )
    df_correct = pd.read_csv('./tests/sample_inputs/sample_input_2.csv')
    assert(np.array_equal(df['id'].to_numpy(), 
                          df_correct['id'].to_numpy()[[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]]))
    assert(np.array_equal(df[['race', 'gender', 'state1', 'state2', 'state3']].to_numpy(), 
                          df_correct[['race', 'gender', 'state1', 'state2', 'state3']].to_numpy()[[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]))
    assert(np.array_equal(df[['action', 'reward']].to_numpy()[[1, 2, 4, 5, 7, 8, 10, 11, 13, 14], :], 
                          df_correct[['action', 'reward']].to_numpy()[[1, 2, 5, 6, 9, 10, 12, 13, 15, 16], :]))



# run the tests
test_reader_well_ordered_columns()
test_reader_randomly_ordered_columns()
print('All reader tests passed!')