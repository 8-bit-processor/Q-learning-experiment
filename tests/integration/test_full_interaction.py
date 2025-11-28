import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to the path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ollama import ResponseError # Import ResponseError for mocking

from src.ollama_client import OllamaClient
from src.agents.teacher_agent import TeacherAgent
from src.agents.student_agent import StudentAgent
from src.q_learning_framework import QLearningFramework
from src.learning_environment import LearningEnvironment

class TestFullInteraction(unittest.TestCase):

    @patch('ollama.Client') # Patch the ollama.Client directly
    def setUp(self, mock_ollama_client_class):
        # Configure the mocked ollama client instance
        self.mock_ollama_client_instance = MagicMock()
        mock_ollama_client_class.return_value = self.mock_ollama_client_instance

        # Mock OllamaClient.is_ollama_running to always return True for integration tests
        with patch.object(OllamaClient, 'is_ollama_running', return_value=True):
            self.ollama_client = OllamaClient(model="test_model")

        # Set up mock responses for ollama_client.generate (no longer directly called by LearningEnvironment)
        # Instead, mock agent methods directly.

        # Mock TeacherAgent and StudentAgent instances
        self.mock_teacher_agent = MagicMock(spec=TeacherAgent)
        self.mock_student_agent = MagicMock(spec=StudentAgent)

        # Configure mock responses for TeacherAgent methods
        self.mock_teacher_agent.generate_learning_material.return_value = "Problem: What is 2+2?"
        self.mock_teacher_agent.evaluate_student_response.return_value = "Correct. Good job."
        self.mock_teacher_agent.synthesize_new_topic.side_effect = ["Advanced Math", "Fractions", "Linear Algebra"]
        self.mock_teacher_agent.evolve.return_value = None # evolve doesn't return anything

        # Configure mock responses for StudentAgent methods
        self.mock_student_agent.solve_problem.return_value = "4"
        self.mock_student_agent.process_feedback.return_value = None # process_feedback doesn't return anything
        self.mock_student_agent.evolve.return_value = None # evolve doesn't return anything
        
        # Mock QLearningFramework methods
        self.q_learner = QLearningFramework(actions=["answer_concisely", "answer_in_detail"], epsilon=0.5)
        self.q_learner.choose_action = MagicMock(return_value="answer_concisely")
        self.q_learner.update_q_value = MagicMock()

        self.env = LearningEnvironment(
            teacher_agent=self.mock_teacher_agent, # Use mock agent
            student_agent=self.mock_student_agent, # Use mock agent
            q_learner=self.q_learner
        )

        self.initial_topics = ["Basic Math"]

    @patch('random.choice')
    def test_full_learning_simulation(self, mock_random_choice):
        num_rounds = 4
        evolution_interval = 2

        # Mock random.choice for topic selection
        mock_random_choice.side_effect = ["Basic Math"] * num_rounds # Only need to mock the topic selections
        
        # self.q_learner.choose_action is already mocked in setUp

        mock_topics = list(self.initial_topics)
        self.env.run_simulation(num_rounds, mock_topics, evolution_interval)

        # Assertions
        # --------------------------------------------------------------------
        # Check calls to agents and q_learner
        self.assertEqual(self.mock_teacher_agent.generate_learning_material.call_count, num_rounds)
        self.assertEqual(self.mock_student_agent.solve_problem.call_count, num_rounds)
        self.assertEqual(self.mock_teacher_agent.evaluate_student_response.call_count, num_rounds)
        self.assertEqual(self.mock_student_agent.process_feedback.call_count, num_rounds)
        self.assertEqual(self.q_learner.choose_action.call_count, num_rounds)
        self.assertEqual(self.q_learner.update_q_value.call_count, num_rounds)
        
        # Evolution calls
        expected_evolve_calls = num_rounds // evolution_interval
        self.assertEqual(self.mock_teacher_agent.evolve.call_count, expected_evolve_calls)
        self.assertEqual(self.mock_student_agent.evolve.call_count, expected_evolve_calls)

        # Topic synthesis calls (at round 1 and 3, which are i=1 and i=3 for num_rounds=4, evolution_interval=2)
        # Condition: i > 0 and i % (evolution_interval // 2) == 0  => i > 0 and i % 1 == 0
        # i=1 (True), i=2 (True), i=3 (True) -> 3 calls
        self.assertEqual(self.mock_teacher_agent.synthesize_new_topic.call_count, 3)

        # Verify Q-table has been updated (simple check)
        # Note: self.q_learner is a real QLearningFramework, not a mock of the instance.
        # But its methods were mocked for call_count. The actual q_table will not be updated as its update_q_value is mocked.
        # This assertion needs to be removed or adapted.
        # For now, commenting it out as its a real instance with mocked methods
        # self.assertGreater(len(self.q_learner.q_table), 0)
        
        # Verify topics list (if new topics were synthesized)
        self.assertIn("Fractions", mock_topics)
        self.assertIn("Advanced Math", mock_topics)

if __name__ == '__main__':
    unittest.main()
