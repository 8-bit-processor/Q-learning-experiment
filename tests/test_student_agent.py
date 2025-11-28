import unittest
from unittest.mock import MagicMock, patch
from src.agents.student_agent import StudentAgent
from src.ollama_client import OllamaClient # Import for type hinting

class TestStudentAgent(unittest.TestCase):

    def setUp(self):
        # Mock the OllamaClient
        self.mock_ollama_client = MagicMock(spec=OllamaClient)
        self.mock_ollama_client.model = "default_ollama_model" # Set a default model attribute

        self.student_model = "test_student_model"
        self.student_agent = StudentAgent(ollama_client=self.mock_ollama_client, student_model=self.student_model)

    def test_initialization(self):
        self.assertEqual(self.student_agent.ollama_client, self.mock_ollama_client)
        self.assertEqual(self.student_agent.student_model, self.student_model)
        
        # Test default model assignment if not provided
        default_student = StudentAgent(ollama_client=self.mock_ollama_client)
        self.assertEqual(default_student.student_model, self.mock_ollama_client.model)

    def test_solve_problem_success(self):
        problem = "What is 2+2?"
        expected_response = "4"
        self.mock_ollama_client.generate_response.return_value = expected_response

        response = self.student_agent.solve_problem(problem)
        self.assertEqual(response, expected_response)
        self.mock_ollama_client.generate_response.assert_called_once()
        args, kwargs = self.mock_ollama_client.generate_response.call_args
        self.assertIn("You are a student. Here is a problem: What is 2+2?", args[0])
        self.assertEqual(kwargs['model'], self.student_model)

    def test_solve_problem_failure(self):
        problem = "What is 2+2?"
        self.mock_ollama_client.generate_response.return_value = None

        response = self.student_agent.solve_problem(problem)
        self.assertEqual(response, "Error: Could not generate student response.")
        self.mock_ollama_client.generate_response.assert_called_once()
    
    @patch('src.agents.student_agent.StudentAgent.reflect_on_feedback')
    @patch('src.agents.student_agent.logging')
    def test_process_feedback(self, mock_logging, mock_reflect_on_feedback):
        problem = "Problem A"
        student_response = "Response B"
        feedback = "Feedback C"

        self.student_agent.process_feedback(problem, student_response, feedback)
        
        mock_logging.info.assert_any_call(f"Student received feedback from the teacher for its response to problem: '{problem[:50]}...'.")
        mock_logging.info.assert_any_call(f"Student's original response: '{student_response[:50]}...'")
        mock_logging.info.assert_any_call(f"Teacher's feedback: '{feedback}'")
        mock_reflect_on_feedback.assert_called_once_with(problem, student_response, feedback)

    def test_reflect_on_feedback_success(self):
        problem = "Problem A"
        student_response = "Response B"
        feedback = "Feedback C"
        expected_reflection = "I learned X and will do Y next time."
        self.mock_ollama_client.generate_response.return_value = expected_reflection

        reflection = self.student_agent.reflect_on_feedback(problem, student_response, feedback)
        self.assertEqual(reflection, expected_reflection)
        self.mock_ollama_client.generate_response.assert_called_once()
        args, kwargs = self.mock_ollama_client.generate_response.call_args
        self.assertIn("You are a student. You attempted to solve the following problem:", args[0])
        self.assertIn(f"Problem: {problem}", args[0])
        self.assertIn(f"Your response was: {student_response}", args[0])
        self.assertIn(f"The teacher provided the following feedback: {feedback}", args[0])
        self.assertEqual(kwargs['model'], self.student_model)

    def test_reflect_on_feedback_failure(self):
        problem = "Problem A"
        student_response = "Response B"
        feedback = "Feedback C"
        self.mock_ollama_client.generate_response.return_value = None

        reflection = self.student_agent.reflect_on_feedback(problem, student_response, feedback)
        self.assertEqual(reflection, "Error: Could not generate reflection.")
        self.mock_ollama_client.generate_response.assert_called_once()

    @patch('src.agents.student_agent.logging')
    def test_evolve(self, mock_logging):
        performance_metric = 0.65
        self.student_agent.evolve(performance_metric)
        mock_logging.info.assert_called_with(f"Student is evolving its learning strategy based on its recent performance (metric: {performance_metric:.2f}).")

if __name__ == '__main__':
    unittest.main()
