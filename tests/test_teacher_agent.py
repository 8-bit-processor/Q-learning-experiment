import unittest
from unittest.mock import MagicMock, patch
from src.agents.teacher_agent import TeacherAgent
from src.ollama_client import OllamaClient # Import for type hinting

class TestTeacherAgent(unittest.TestCase):

    def setUp(self):
        # Mock the OllamaClient
        self.mock_ollama_client = MagicMock(spec=OllamaClient)
        self.mock_ollama_client.model = "default_ollama_model" # Set a default model attribute

        self.teacher_model = "test_teacher_model"
        self.teacher_agent = TeacherAgent(ollama_client=self.mock_ollama_client, teacher_model=self.teacher_model)

    def test_initialization(self):
        self.assertEqual(self.teacher_agent.ollama_client, self.mock_ollama_client)
        self.assertEqual(self.teacher_agent.teacher_model, self.teacher_model)
        
        # Test default model assignment if not provided
        default_teacher = TeacherAgent(ollama_client=self.mock_ollama_client)
        self.assertEqual(default_teacher.teacher_model, self.mock_ollama_client.model)

    def test_generate_learning_material_success(self):
        topic = "Quantum Physics"
        difficulty = "hard"
        expected_material = "Explain wave-particle duality."
        self.mock_ollama_client.generate_response.return_value = expected_material

        material = self.teacher_agent.generate_learning_material(topic, difficulty)
        self.assertEqual(material, expected_material)
        self.mock_ollama_client.generate_response.assert_called_once()
        args, kwargs = self.mock_ollama_client.generate_response.call_args
        self.assertIn("As an expert teacher, create a hard difficulty learning problem on the topic of 'Quantum Physics'.", args[0])
        self.assertEqual(kwargs['model'], self.teacher_model)

    def test_generate_learning_material_failure(self):
        topic = "Quantum Physics"
        difficulty = "hard"
        self.mock_ollama_client.generate_response.return_value = None # Simulate failure

        material = self.teacher_agent.generate_learning_material(topic, difficulty)
        self.assertEqual(material, "Error: Could not generate learning material.")
        self.mock_ollama_client.generate_response.assert_called_once()

    def test_evaluate_student_response_success(self):
        problem = "What is gravity?"
        student_response = "Gravity is a force that attracts any objects with mass."
        expected_feedback = "Correct, but mention spacetime curvature."
        self.mock_ollama_client.generate_response.return_value = expected_feedback

        feedback = self.teacher_agent.evaluate_student_response(problem, student_response)
        self.assertEqual(feedback, expected_feedback)
        self.mock_ollama_client.generate_response.assert_called_once()
        args, kwargs = self.mock_ollama_client.generate_response.call_args
        self.assertIn("Evaluate the following student response to the problem below.", args[0])
        self.assertIn(problem, args[0])
        self.assertIn(student_response, args[0])
        self.assertEqual(kwargs['model'], self.teacher_model)

    def test_evaluate_student_response_failure(self):
        problem = "What is gravity?"
        student_response = "Gravity is a force that attracts any objects with mass."
        self.mock_ollama_client.generate_response.return_value = None

        feedback = self.teacher_agent.evaluate_student_response(problem, student_response)
        self.assertEqual(feedback, "Error: Could not evaluate student response.")
        self.mock_ollama_client.generate_response.assert_called_once()
    
    def test_synthesize_new_topic_success(self):
        current_topic = "Basic Algebra"
        student_summary = "Student performed well on linear equations."
        expected_new_topic = "Quadratic Equations"
        self.mock_ollama_client.generate_response.return_value = expected_new_topic

        new_topic = self.teacher_agent.synthesize_new_topic(current_topic, student_summary)
        self.assertEqual(new_topic, expected_new_topic)
        
        expected_prompt = (f"As an expert educator, you are teaching a student about '{current_topic}'.\n"
                           f"Based on the student's overall performance '{student_summary}', "
                           f"propose a new, related learning topic that would be a logical next step or "
                           f"address a knowledge gap. Provide only the topic name, concisely.")
        
        self.mock_ollama_client.generate_response.assert_called_once_with(expected_prompt, model=self.teacher_model)


    def test_synthesize_new_topic_failure(self):
        current_topic = "Basic Algebra"
        student_summary = "Student performed well on linear equations."
        self.mock_ollama_client.generate_response.return_value = None

        new_topic = self.teacher_agent.synthesize_new_topic(current_topic, student_summary)
        self.assertEqual(new_topic, "Error: Could not synthesize new topic.")
        self.mock_ollama_client.generate_response.assert_called_once()

    @patch('src.agents.teacher_agent.logging')
    def test_evolve(self, mock_logging):
        performance_metric = 0.75
        self.teacher_agent.evolve(performance_metric)
        mock_logging.info.assert_called_with(f"Teacher is evolving its teaching strategy based on student performance (metric: {performance_metric:.2f}).")

if __name__ == '__main__':
    unittest.main()
