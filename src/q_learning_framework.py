import random
import logging
import sys

# Configure logging for the module.
# Log messages will be displayed in the console (sys.stdout) with a timestamp,
# log level, and the message itself. INFO level messages and above will be shown.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

class QLearningFramework:
    """
    Implements a basic Q-Learning algorithm.
    This framework allows an agent to learn optimal actions in an environment
    by estimating Q-values for state-action pairs.
    """
    def __init__(self, actions: list, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        """
        Initializes the Q-Learning framework with its core parameters.

        Args:
            actions (list): A list of all possible discrete actions the agent can take.
                            States and actions should be hashable (e.g., strings, numbers, tuples).
            alpha (float): The learning rate (0 < alpha <= 1). Determines how much new information
                           overrides old information. A value of 0 makes the agent learn nothing,
                           while a value of 1 makes the agent consider only the most recent information.
            gamma (float): The discount factor (0 <= gamma <= 1). Determines the importance of
                           future rewards. A value of 0 makes the agent "myopic" by only considering
                           current rewards, while a value of 1 makes it strive for a long-term high reward.
            epsilon (float): The exploration rate (0 <= epsilon <= 1). Governs the trade-off between
                             exploration (taking random actions to discover new strategies) and
                             exploitation (taking the best-known action based on current Q-values).
        """
        # Q-table: Stores the Q-values. It's a dictionary where keys are (state, action) tuples
        # and values are the estimated maximum future reward for taking that action in that state.
        self.q_table = {}  # Format: {(state, action): q_value}
        self.actions = actions  # List of all possible actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        logging.info(f"QLearningFramework initialized with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")

    def _get_q_value(self, state, action) -> float:
        """
        Retrieves the Q-value for a given state-action pair from the Q-table.
        If the state-action pair has not been encountered before, it returns 0.0 (initial Q-value).

        Args:
            state: The current state of the environment.
            action: The action taken in that state.

        Returns:
            float: The Q-value for the (state, action) pair.
        """
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state) -> any:
        """
        Chooses an action for the agent to take in the given state using an epsilon-greedy policy.
        With probability epsilon, the agent explores (chooses a random action).
        With probability (1 - epsilon), the agent exploits (chooses the action with the highest Q-value).

        Args:
            state: The current state from which to choose an action.

        Returns:
            any: The chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose a random action from the list of all possible actions.
            action = random.choice(self.actions)
            logging.debug(f"Exploring: chose random action '{action}' for state '{state}'")
        else:
            # Exploitation: Choose the action that currently has the highest Q-value for the current state.
            q_values_for_state = {action: self._get_q_value(state, action) for action in self.actions}
            
            # Find the maximum Q-value among all actions for the current state.
            max_q = max(q_values_for_state.values())
            
            # Identify all actions that yield this maximum Q-value (to handle ties).
            best_actions = [action for action, q_value in q_values_for_state.items() if q_value == max_q]
            
            # If there are multiple actions with the same maximum Q-value, choose one randomly.
            action = random.choice(best_actions) # Randomly break ties
            
            logging.debug(f"Exploiting: chose action '{action}' with Q-value {max_q} for state '{state}'")
        return action

    def update_q_value(self, state, action, reward: float, next_state) -> None:
        """
        Updates the Q-value for a specific state-action pair using the Q-learning update rule.
        This rule incorporates the immediate reward received and the estimated maximum future
        reward from the next state.

        The Q-learning update formula is:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(next_state, a')) - Q(s,a)]

        Args:
            state: The state before taking the action.
            action: The action that was taken.
            reward (float): The immediate reward received after taking the action.
            next_state: The state observed after taking the action. If it's a terminal state,
                        this can be None or a specific indicator.
        """
        current_q = self._get_q_value(state, action)
        
        # Calculate the maximum Q-value for the next state (max(Q(next_state, a'))).
        # This term represents the estimated optimal future value from the next state.
        max_next_q = 0.0
        if next_state: # Only calculate if next_state is not terminal
            # Get all Q-values for actions from the next state.
            q_values_for_next_state = [self._get_q_value(next_state, a) for a in self.actions]
            # Find the maximum among them. If the list is empty (e.g., no known actions for next_state), max_next_q remains 0.0.
            if q_values_for_next_state:
                max_next_q = max(q_values_for_next_state)

        # Apply the Q-learning update rule.
        # This is the core learning step: adjust the current Q-value towards the
        # "target" value (reward + gamma * max_next_q).
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Store the newly calculated Q-value in the Q-table.
        self.q_table[(state, action)] = new_q
        
        logging.debug(f"Updated Q-value for ({state}, {action}): {current_q:.4f} -> {new_q:.4f}. Reward: {reward:.2f}, Next Max Q: {max_next_q:.4f}")

# Example usage block:
# This code runs only when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    # Define a simple set of actions that the Q-learner can choose from.
    possible_actions = ["action_a", "action_b", "action_c"]
    
    # Instantiate the QLearningFramework.
    q_learner = QLearningFramework(actions=possible_actions)

    # Simulate a few steps of interaction to demonstrate Q-value updates.
    current_state = "start_state"
    print(f"Initial Q-table size: {len(q_learner.q_table)}")

    for i in range(10): # Run 10 simulation steps
        print(f"\n--- Step {i+1} ---")
        
        # Agent chooses an action based on its current state and Q-table.
        action = q_learner.choose_action(current_state)
        
        # Simulate interaction with an environment:
        # Determine a reward and the next state based on the chosen action.
        if action == "action_a":
            reward = 1.0
            next_state = "intermediate_state"
        elif action == "action_b":
            reward = -0.5
            next_state = "start_state" # Staying in the same state
        else: # action_c
            reward = 0.2
            next_state = "end_state" # Moving to a different state

        # Update the Q-value for the (current_state, action) pair based on the received reward and next state.
        q_learner.update_q_value(current_state, action, reward, next_state)
        
        # Transition to the next state for the next step of the simulation.
        current_state = next_state
        print(f"Current state: {current_state}, Action: {action}, Reward: {reward:.2f}")
    
    print("\nFinal Q-table (first 10 entries):")
    # Print a snapshot of the learned Q-values.
    for i, (key, value) in enumerate(q_learner.q_table.items()):
        if i >= 10: # Limit output for readability
            break
        print(f"  {key}: {value:.4f}")
    print(f"Final Q-table size: {len(q_learner.q_table)}")