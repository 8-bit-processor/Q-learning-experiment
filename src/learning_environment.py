import random
import logging
import sys
# Import the agent and Q-learning components that the environment will manage.
from src.agents.teacher_agent import TeacherAgent
from src.agents.student_agent import StudentAgent
from src.q_learning_framework import QLearningFramework
# OllamaClient is imported here only for the example usage block, not for the class itself.
from src.ollama_client import OllamaClient 

# Configure logging for the module.
# Log messages will be displayed in the console (sys.stdout) with a timestamp,
# log level, and the message itself. INFO level messages and above will be shown.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

class LearningEnvironment:
    """
    The LearningEnvironment orchestrates the interaction between the TeacherAgent and StudentAgent.
    It manages the simulation flow, applies Q-learning updates, triggers agent evolution,
    and collects performance metrics.
    """
    def __init__(self, teacher_agent: TeacherAgent, student_agent: StudentAgent, q_learner: QLearningFramework):
        """
        Initializes the LearningEnvironment.

        Args:
            teacher_agent (TeacherAgent): An instance of the TeacherAgent.
            student_agent (StudentAgent): An instance of the StudentAgent.
            q_learner (QLearningFramework): An instance of the QLearningFramework for the student.
        """
        self.teacher = teacher_agent
        self.student = student_agent
        self.q_learner = q_learner
        logging.info("LearningEnvironment initialized.")
        # List to store the results of each interaction round for later analysis.
        self.simulation_results = [] 
        self.current_round_number = 0 # Initialize current round number for tracking

    def _get_reward_from_feedback(self, feedback: str) -> float:
        """
        Parses the teacher's natural language feedback to generate a numerical reward signal
        for the student agent's Q-learning process. This is a heuristic and can be improved.

        Args:
            feedback (str): The feedback string provided by the TeacherAgent.

        Returns:
            float: A numerical reward value based on keywords in the feedback.
                   Positive rewards for "correct" and "strengths".
                   Negative rewards for "incorrect" and "improvement".
        """
        feedback_lower = feedback.lower()
        reward = 0.0
        
        # Heuristic rules for assigning reward based on feedback keywords.
        if "correct" in feedback_lower and "incorrect" not in feedback_lower:
            reward += 1.0
        if "incorrect" in feedback_lower:
            reward -= 1.0
        if "strengths" in feedback_lower:
            reward += 0.5
        if "improvement" in feedback_lower: # Acknowledge need for improvement
            reward -= 0.5 
        
        logging.debug(f"Feedback '{feedback[:50]}...' resulted in reward: {reward}")
        return reward

    def run_interaction_round(self, topic: str, difficulty: str = "medium", total_rounds: int = 0) -> dict:
        """
        Executes a single round of interaction between the teacher and student agents.
        This involves the teacher posing a problem, the student responding, the teacher
        evaluating, and the Q-learner updating its values.

        Args:
            topic (str): The learning topic for this round.
            difficulty (str, optional): The difficulty level for the problem. Defaults to "medium".
            total_rounds (int, optional): The total number of rounds in the simulation, for logging. Defaults to 0.

        Returns:
            dict: A dictionary containing details of the round (topic, problem, response, feedback, reward),
                  or an empty dictionary if an critical error occurred during the round.
        """
        logging.info(f"\n--- Starting a new learning round ({self.current_round_number}/{total_rounds}) on: '{topic}' ---")

        # Step 1: Teacher creates a learning challenge.
        logging.info(f"Teacher is creating a learning problem on '{topic}' (difficulty: {difficulty}).")
        problem = self.teacher.generate_learning_material(topic, difficulty)
        if "Error" in problem:
            logging.error("Teacher failed to generate a problem. Skipping this round.")
            return {}

        # Step 2: Student decides how to approach the problem.
        # This is based on its learned 'strategy' from past experiences (Q-learning).
        logging.info(f"Student is deciding its strategy for '{topic}' (current state) using Q-learning.")
        current_state_for_q_learning = topic 
        
        if not isinstance(current_state_for_q_learning, str):
            current_state_for_q_learning = str(current_state_for_q_learning)

        student_conceptual_action = self.q_learner.choose_action(current_state_for_q_learning)
        logging.info(f"Student decided to try the strategy: '{student_conceptual_action}'.")
        
        # Step 3: Student attempts to solve the problem using its chosen strategy.
        logging.info(f"Student is attempting to solve the problem: '{problem[:50]}...'")
        student_response = self.student.solve_problem(problem)
        if "Error" in student_response:
            logging.error("Student failed to generate a response. Skipping this round.")
            return {}
        logging.info(f"Student's response: '{student_response[:50]}...'")

        # Step 4: Teacher evaluates the student's attempt.
        logging.info("Teacher is evaluating the student's response and preparing feedback.")
        teacher_feedback = self.teacher.evaluate_student_response(problem, student_response)
        if "Error" in teacher_feedback:
            logging.error("Teacher failed to evaluate the student's response. Skipping this round.")
            return {}
        logging.info(f"Teacher provides feedback: '{teacher_feedback[:100]}...'")

        # Step 5: Student processes the feedback and reflects on its performance.
        logging.info("Student is processing the teacher's feedback and reflecting on what it learned.")
        self.student.process_feedback(problem, student_response, teacher_feedback)

        # Step 6: The student's internal 'brain' (Q-learner) learns from the experience.
        # A numerical reward is extracted from the teacher's feedback.
        reward = self._get_reward_from_feedback(teacher_feedback)
        logging.info(f"Student received a reward of {reward:.2f} for this interaction round.")
        
        # The Q-table is updated, helping the student understand which strategies lead to better rewards.
        next_state_for_q_learning = topic 
        self.q_learner.update_q_value(current_state_for_q_learning, student_conceptual_action, reward, next_state_for_q_learning)
        logging.info("Student's internal Q-table updated based on this experience.")

        logging.info(f"--- Learning round completed. Reward for this round: {reward:.2f} ---")
        
        round_data = {
            "round_number": self.current_round_number,
            "topic": topic,
            "difficulty": difficulty,
            "problem": problem,
            "student_action": student_conceptual_action,
            "student_response": student_response,
            "teacher_feedback": teacher_feedback,
            "reward": reward
        }
        self.simulation_results.append(round_data) 
        return round_data

    def run_simulation(self, num_rounds: int, topics: list, evolution_interval: int = 5, socketio=None) -> list[dict]:
        """
        Runs a simulation consisting of multiple interaction rounds between the agents.
        Agents evolve periodically based on accumulated performance metrics.

        Args:
            num_rounds (int): The total number of interaction rounds to run.
            topics (list): A mutable list of possible learning topics. This list can be
                           modified by the teacher synthesizing new topics.
            evolution_interval (int): How often (in terms of rounds) to trigger the
                                      `evolve` method for both agents.
            socketio: Optional SocketIO instance for real-time progress updates.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the results
                        from a single interaction round.
        """
        logging.info(f"\n--- Starting a full learning simulation for {num_rounds} rounds ---")
        rewards_history = [] 

        for i in range(num_rounds):
            self.current_round_number = i + 1 # Track current round number
            
            # Send progress update to web client for real-time feedback
            if socketio:
                socketio.emit('simulation_progress', {'current_round': i + 1, 'total_rounds': num_rounds})
            
            # The teacher occasionally synthesizes new topics to keep the learning fresh and adaptive.
            if i > 0 and i % (evolution_interval // 2) == 0: 
                logging.info(f"Teacher is considering student's past performance to synthesize a new topic.")
                student_performance_summary = (
                    f"Student's average reward over last {len(rewards_history)} rounds was "
                    f"{sum(rewards_history)/len(rewards_history):.2f}" if rewards_history else "No recent performance data."
                )
                new_topic = self.teacher.synthesize_new_topic(topics[-1] if topics else "general knowledge", student_performance_summary)
                
                if new_topic and "Error" not in new_topic and new_topic not in topics:
                    topics.append(new_topic);
                    logging.info(f"Teacher successfully synthesized a new topic: '{new_topic}'. This will be added to the learning curriculum.")
                elif "Error" in new_topic:
                    logging.warning("Teacher attempted to synthesize a new topic but encountered an error. Continuing with existing topics.")
                else:
                    logging.info("Teacher considered synthesizing a new topic, but decided to stick with existing topics or generated a duplicate.")

            # Randomly select a topic for the current round from the available topics.
            current_topic = random.choice(topics) 
            
            # Run one full interaction round.
            round_data = self.run_interaction_round(current_topic, "medium", num_rounds)
            if round_data:
                rewards_history.append(round_data["reward"])
            
            # Periodically, both the teacher and student agents get a chance to evolve their strategies.
            if (i + 1) % evolution_interval == 0 and len(rewards_history) > 0:
                avg_reward = sum(rewards_history) / len(rewards_history) 
                logging.info(f"\n--- Evolution Point (End of Round {i+1}) ---")
                logging.info(f"Agents are evolving! Average reward over the last {len(rewards_history)} rounds was: {avg_reward:.2f}.")
                
                # The teacher's teaching strategy evolves based on student performance.
                self.teacher.evolve(avg_reward)
                # The student's learning strategy evolves based on its performance.
                self.student.evolve(avg_reward) 
                
                rewards_history = [] # Reset history for the next evolution interval.
        
        logging.info(f"\n--- Learning Simulation Completed ---")
        
        summary = self.get_summary_statistics()
        logging.info("\n--- Final Simulation Summary ---")
        for key, value in summary.items():
            logging.info(f"- {key}: {value}")
            
        return self.simulation_results
        
    def get_summary_statistics(self) -> dict:
        """
        Calculates and returns summary statistics of the entire simulation based on
        the `simulation_results` collected from all interaction rounds.

        Returns:
            dict: A dictionary containing various statistics such as total rounds,
                  average reward, minimum reward, maximum reward, and unique topics covered.
        """
        if not self.simulation_results:
            # Return default zero values if no simulation data is available.
            return {"Total Rounds": 0, "Average Reward": 0.0, "Min Reward": 0.0, "Max Reward": 0.0, "Unique Topics Covered": []}

        # Extract rewards and topics from the stored simulation results.
        rewards = [res["reward"] for res in self.simulation_results]
        topics_covered = [res["topic"] for res in self.simulation_results]

        # Compute and return the summary statistics.
        return {
            "Total Rounds": len(self.simulation_results),
            "Average Reward": sum(rewards) / len(rewards),
            "Min Reward": min(rewards),
            "Max Reward": max(rewards),
            "Unique Topics Covered": list(set(topics_covered)) # Use set to get unique topics.
        }

# Example usage block:
# This code runs only when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    # Import configuration for Ollama.
    from src.config import OLLAMA_MODEL

    # Initialize the Ollama client.
    ollama_client_instance = OllamaClient(model=OLLAMA_MODEL)

    # Ensure Ollama server is running before starting the simulation example.
    if not ollama_client_instance.is_ollama_running():
        print("Ollama server is not running or model not available. Please ensure it's set up correctly.")
    else:
        # Initialize the Teacher and Student agents.
        teacher = TeacherAgent(ollama_client=ollama_client_instance)
        student = StudentAgent(ollama_client=ollama_client_instance)

        # Define possible actions for the student's Q-learner (conceptual actions).
        possible_student_q_actions = ["answer_concisely", "answer_in_detail", "ask_for_clarification"]
        # Initialize the Q-learning framework for the student.
        q_learner = QLearningFramework(actions=possible_student_q_actions, epsilon=0.5) 

        # Initialize the LearningEnvironment with the agents and Q-learner.
        env = LearningEnvironment(teacher_agent=teacher, student_agent=student, q_learner=q_learner)

        # Define initial topics for the simulation.
        topics = ["Reinforcement Learning", "Neural Networks", "Generative AI"]
        
        # Run a short simulation for demonstration purposes.
        print("\n--- Running Example Interaction Rounds ---")
        # The run_simulation method will now handle running multiple rounds, evolution, and logging summaries.
        env.run_simulation(num_rounds=5, topics=topics, evolution_interval=2)
        
        # The example usage in the main block used to print individual rounds and Q-table manually.
        # Now, `run_simulation` handles most of that, but if specific post-simulation analysis is needed:
        print("\n--- Post-Simulation Q-table Entries (Example) ---")
        # Print a sample of the Q-table to show what was learned.
        for i, (key, value) in enumerate(q_learner.q_table.items()):
            if i >= 10: break # Show first 10 entries for brevity
            print(f"  {key}: {value:.4f}")