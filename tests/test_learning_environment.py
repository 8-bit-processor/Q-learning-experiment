import unittest
from unittest.mock import MagicMock, patch
from src.learning_environment import LearningEnvironment
from src.agents.teacher_agent import TeacherAgent
from src.agents.student_agent import StudentAgent
from src.q_learning_framework import QLearningFramework

class TestLearningEnvironment(unittest.TestCase):

    def setUp(self):
        self.mock_teacher_agent = MagicMock(spec=TeacherAgent)
        self.mock_student_agent = MagicMock(spec=StudentAgent)
        self.mock_q_learner = MagicMock(spec=QLearningFramework)
        self.mock_q_learner.actions = ["action1", "action2"] # QLearner needs some actions

        self.env = LearningEnvironment(
            teacher_agent=self.mock_teacher_agent,
            student_agent=self.mock_student_agent,
            q_learner=self.mock_q_learner
        )

    def test_initialization(self):
        self.assertEqual(self.env.teacher, self.mock_teacher_agent)
        self.assertEqual(self.env.student, self.mock_student_agent)
        self.assertEqual(self.env.q_learner, self.mock_q_learner)

    def test_get_reward_from_feedback(self):
        self.assertEqual(self.env._get_reward_from_feedback("Your answer is correct."), 1.0)
        self.assertEqual(self.env._get_reward_from_feedback("Your answer is incorrect."), -1.0)
        self.assertEqual(self.env._get_reward_from_feedback("Good, but needs improvement."), -0.5) # 0.5 - 0.5
        self.assertEqual(self.env._get_reward_from_feedback("Excellent strengths! No improvements needed."), 0.0) # 1.0 + 0.5
        self.assertEqual(self.env._get_reward_from_feedback("Incorrect, needs lots of improvement."), -1.5) # -1.0 - 0.5
        self.assertEqual(self.env._get_reward_from_feedback("Neutral feedback."), 0.0)

    @patch('random.choice', side_effect=["action1"]) # Mock student's Q-action
    def test_run_interaction_round_success(self, mock_random_choice):
        topic = "Test Topic"
        difficulty = "easy"
        problem = "What is X?"
        student_response = "X is Y."
        teacher_feedback = "Correct, good strengths."
        reward = 1.5 # Based on _get_reward_from_feedback

        self.mock_q_learner.choose_action.return_value = "action1" # Set return value for mock
        self.mock_teacher_agent.generate_learning_material.return_value = problem
        self.mock_student_agent.solve_problem.return_value = student_response
        self.mock_teacher_agent.evaluate_student_response.return_value = teacher_feedback

        result = self.env.run_interaction_round(topic, difficulty)

        self.mock_teacher_agent.generate_learning_material.assert_called_once_with(topic, difficulty)
        self.mock_q_learner.choose_action.assert_called_once_with(topic)
        self.mock_student_agent.solve_problem.assert_called_once_with(problem)
        self.mock_teacher_agent.evaluate_student_response.assert_called_once_with(problem, student_response)
        self.mock_student_agent.process_feedback.assert_called_once_with(problem, student_response, teacher_feedback)
        self.mock_q_learner.update_q_value.assert_called_once_with(topic, "action1", reward, topic)

        self.assertEqual(result["topic"], topic)
        self.assertEqual(result["problem"], problem)
        self.assertEqual(result["student_response"], student_response)
        self.assertEqual(result["teacher_feedback"], teacher_feedback)
        self.assertEqual(result["reward"], reward)
    
    def test_run_interaction_round_teacher_problem_failure(self):
        topic = "Test Topic"
        self.mock_teacher_agent.generate_learning_material.return_value = "Error: blah"

        result = self.env.run_interaction_round(topic)
        self.assertEqual(result, {})
        self.mock_teacher_agent.generate_learning_material.assert_called_once()
        self.mock_student_agent.solve_problem.assert_not_called()

    def test_run_interaction_round_student_response_failure(self):
        topic = "Test Topic"
        problem = "What is X?"
        self.mock_teacher_agent.generate_learning_material.return_value = problem
        self.mock_student_agent.solve_problem.return_value = "Error: blah"

        result = self.env.run_interaction_round(topic)
        self.assertEqual(result, {})
        self.mock_teacher_agent.generate_learning_material.assert_called_once()
        self.mock_student_agent.solve_problem.assert_called_once()
        self.mock_teacher_agent.evaluate_student_response.assert_not_called()

    def test_run_interaction_round_teacher_feedback_failure(self):
        topic = "Test Topic"
        problem = "What is X?"
        student_response = "X is Y."
        self.mock_teacher_agent.generate_learning_material.return_value = problem
        self.mock_student_agent.solve_problem.return_value = student_response
        self.mock_teacher_agent.evaluate_student_response.return_value = "Error: blah"

        result = self.env.run_interaction_round(topic)
        self.assertEqual(result, {})
        self.mock_teacher_agent.generate_learning_material.assert_called_once()
        self.mock_student_agent.solve_problem.assert_called_once()
        self.mock_teacher_agent.evaluate_student_response.assert_called_once()
        self.mock_student_agent.process_feedback.assert_not_called()

    @patch('random.choice', side_effect=["topic1", "topic2", "action1", "action1"]) # Mock topic and student Q-action
    def test_run_simulation(self, mock_random_choice):
        num_rounds = 4
        topics = ["topic1", "topic2"]
        evolution_interval = 2

        self.mock_teacher_agent.generate_learning_material.return_value = "problem"
        self.mock_student_agent.solve_problem.return_value = "response"
        self.mock_teacher_agent.evaluate_student_response.return_value = "Correct." # Reward = 1.0
        self.mock_teacher_agent.synthesize_new_topic.return_value = "new_topic_A"

        self.env.run_simulation(num_rounds, topics, evolution_interval)

        self.assertEqual(self.mock_teacher_agent.generate_learning_material.call_count, num_rounds)
        self.assertEqual(self.mock_student_agent.solve_problem.call_count, num_rounds)
        self.assertEqual(self.mock_teacher_agent.evaluate_student_response.call_count, num_rounds)
        self.assertEqual(self.mock_student_agent.process_feedback.call_count, num_rounds)
        self.assertEqual(self.mock_q_learner.update_q_value.call_count, num_rounds)

        # Evolution should be called num_rounds / evolution_interval times
        expected_evolve_calls = num_rounds // evolution_interval
        self.assertEqual(self.mock_teacher_agent.evolve.call_count, expected_evolve_calls)
        self.assertEqual(self.mock_student_agent.evolve.call_count, expected_evolve_calls)

        # Synthesize new topic should be called num_rounds / (evolution_interval // 2) times
        # (considering it's called on i % (evolution_interval // 2) == 0, excluding round 0)
        # For num_rounds=4, evolution_interval=2, it's called at round 2 and round 4
        # (index 1 and 3) -> 2 calls
        self.assertEqual(self.mock_teacher_agent.synthesize_new_topic.call_count, 3)
        # Check if new topic was added to the list (mocking should handle this internally)
        self.assertIn("new_topic_A", topics) # Original 'topics' list should be modified

if __name__ == '__main__':
    unittest.main()
