import unittest
from unittest.mock import patch, MagicMock
from src.q_learning_framework import QLearningFramework

class TestQLearningFramework(unittest.TestCase):

    def setUp(self):
        self.actions = ["action1", "action2", "action3"]
        self.q_learner = QLearningFramework(actions=self.actions, alpha=0.1, gamma=0.9, epsilon=0.1)

    def test_initialization(self):
        self.assertEqual(self.q_learner.alpha, 0.1)
        self.assertEqual(self.q_learner.gamma, 0.9)
        self.assertEqual(self.q_learner.epsilon, 0.1)
        self.assertEqual(self.q_learner.actions, self.actions)
        self.assertEqual(self.q_learner.q_table, {})

    def test_get_q_value(self):
        state = "test_state"
        action = "action1"
        
        # Test unseen state-action pair
        self.assertEqual(self.q_learner._get_q_value(state, action), 0.0)

        # Set a Q-value and test retrieving it
        self.q_learner.q_table[(state, action)] = 0.5
        self.assertEqual(self.q_learner._get_q_value(state, action), 0.5)

    @patch('random.uniform')
    @patch('random.choice')
    def test_choose_action_exploration(self, mock_choice, mock_uniform):
        state = "test_state"
        self.q_learner.epsilon = 1.0 # Always explore
        mock_uniform.return_value = 0.5 # Ensure exploration branch is taken
        mock_choice.return_value = "action2" # Mock random choice

        chosen_action = self.q_learner.choose_action(state)
        self.assertEqual(chosen_action, "action2")
        mock_uniform.assert_called_once_with(0, 1)
        mock_choice.assert_called_once_with(self.actions)

    @patch('random.uniform')
    @patch('random.choice') # This patch is for tie-breaking in exploitation
    def test_choose_action_exploitation(self, mock_choice, mock_uniform):
        state = "test_state"
        self.q_learner.epsilon = 0.0 # Always exploit

        # Test exploitation without a tie
        mock_uniform.return_value = 0.0 # Ensure exploitation branch is taken

        # Set specific Q-values to ensure a deterministic choice
        self.q_learner.q_table = {} # Clear Q-table from previous tests
        self.q_learner.q_table[(state, "action1")] = 0.1
        self.q_learner.q_table[(state, "action2")] = 0.5 # This should be chosen
        self.q_learner.q_table[(state, "action3")] = 0.2

        # In this case, best_actions will be ["action2"], and random.choice(["action2"]) will return "action2"
        mock_choice.return_value = "action2" 

        chosen_action = self.q_learner.choose_action(state)
        self.assertEqual(chosen_action, "action2")
        mock_uniform.assert_called_once_with(0, 1)
        mock_choice.assert_called_once_with(["action2"]) # random.choice should be called with ["action2"]

        # Reset mocks for the tie-breaking test
        mock_uniform.reset_mock()
        mock_choice.reset_mock()

        # Test with a tie
        self.q_learner.q_table[(state, "action2")] = 0.5
        self.q_learner.q_table[(state, "action3")] = 0.5 # Now action2 and action3 are tied
        mock_choice.return_value = "action3" # Mock random choice for tie-breaking
        mock_uniform.return_value = 0.0 # Ensure exploitation branch is taken

        chosen_action = self.q_learner.choose_action(state)
        self.assertEqual(chosen_action, "action3")
        mock_uniform.assert_called_once_with(0, 1)
        # In this case, best_actions will be ["action2", "action3"] or ["action3", "action2"]
        # The order depends on dictionary iteration, but the content is what matters.
        # We can assert that it was called with a list containing both.
        mock_choice.assert_called_once() # We can't assert the exact list due to order, just that it was called

    def test_update_q_value(self):
        state = "state_s"
        action = "action_a"
        reward = 1.0
        next_state = "state_s_prime"

        # Initial Q-value is 0
        self.assertEqual(self.q_learner._get_q_value(state, action), 0.0)

        # Update (Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)])
        # 0 + 0.1 * [1.0 + 0.9 * 0.0 - 0.0] = 0.1
        self.q_learner.update_q_value(state, action, reward, next_state)
        self.assertAlmostEqual(self.q_learner._get_q_value(state, action), 0.1)

        # Set a Q-value for next state to influence update
        self.q_learner.q_table[(next_state, "action1")] = 2.0 # max Q for next_state will be 2.0
        self.q_learner.q_table[(next_state, "action2")] = 1.0
        self.q_learner.q_table[(next_state, "action3")] = 0.5

        # Update again
        # Q(s,a) is currently 0.1
        # 0.1 + 0.1 * [1.0 + 0.9 * 2.0 - 0.1]
        # 0.1 + 0.1 * [1.0 + 1.8 - 0.1]
        # 0.1 + 0.1 * [2.7] = 0.1 + 0.27 = 0.37
        self.q_learner.update_q_value(state, action, reward, next_state)
        self.assertAlmostEqual(self.q_learner._get_q_value(state, action), 0.37)

        # Test update with terminal state (next_state is None)
        self.q_learner = QLearningFramework(actions=self.actions, alpha=0.1, gamma=0.9, epsilon=0.1) # Reset Q-learner
        terminal_state = None
        self.q_learner.update_q_value(state, action, reward, terminal_state)
        # 0 + 0.1 * [1.0 + 0.9 * 0.0 - 0.0] = 0.1
        self.assertAlmostEqual(self.q_learner._get_q_value(state, action), 0.1)

if __name__ == '__main__':
    unittest.main()
