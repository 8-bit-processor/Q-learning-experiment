import logging
import sys
from src.ollama_client import OllamaClient


# Configure logging for the module.
# Log messages will be displayed in the console (sys.stdout)
# with a timestamp, log level, and the message itself.
# INFO level messages and above will be shown.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class StudentAgent:
    """
    The StudentAgent is responsible for attempting to solve problems posed by the
    TeacherAgent, processing feedback, and reflecting on its learning.
    It uses an LLM via Ollama to generate responses and reflections.
    """
    def __init__(self, ollama_client: OllamaClient,
                 student_model: str = None):
        """
        Initializes the StudentAgent.

        Args:
            ollama_client (OllamaClient): An instance of OllamaClient to
                                          communicate with the LLM.
            student_model (str, optional): The specific LLM model to be used by
                                           the student. If None, the model
                                           configured in the ollama_client will
                                           be used.
        """
        self.ollama_client = ollama_client
        # If a specific student_model is provided, use it; otherwise,
        # default to the client's model.
        self.student_model = student_model if student_model else ollama_client.model
        logging.info(f"StudentAgent initialized with model: {self.student_model}")

    def solve_problem(self, problem: str) -> str:
        """
        Receives a problem statement and generates a response or solution using
        the LLM.

        Args:
            problem (str): The problem statement provided by the TeacherAgent.

        Returns:
            str: The student's generated answer or solution, or an error message.
        """
        prompt = (f"You are a student. Here is a problem: {problem}\n\n"
                  "Provide a clear and concise answer or solution to the "
                  "problem.")
        
        logging.info(f"Student is thinking about problem: {problem[:50]}...")
        
        response = self.ollama_client.generate_response(prompt, model=self.student_model)
        
        if response:
            logging.info(f"Student responds: '{response[:100]}...'")
            return response
        else:
            logging.error(f"Student failed to generate a response for problem: "
                          f"'{problem[:50]}...'.")  # noqa: E501
            return "Error: Could not generate student response."

    def process_feedback(self, problem: str, student_response: str,
                         feedback: str) -> None:
        """
        Processes feedback received from the TeacherAgent. This method is a
        crucial point for the student's learning and adaptation.
        It also triggers an internal reflection.

        Args:
            problem (str): The original problem the student attempted.
            student_response (str): The student's response to the problem.
            feedback (str): The feedback provided by the teacher.
        """
        logging.info(f"Student received feedback from the teacher for its "
                     f"response to problem: '{problem[:50]}...'.")  # noqa: E501
        logging.info(f"Student's original response: '{student_response[:50]}...'")
        logging.info(f"Teacher's feedback: '{feedback}'")
        
        # Trigger an internal reflection process based on the feedback.
        self.reflect_on_feedback(problem, student_response, feedback)
        logging.info("Student has processed the feedback and initiated internal "
                     "reflection.")  # noqa: E501
        pass

    def reflect_on_feedback(self, problem: str, student_response: str,
                            feedback: str) -> str:
        """
        Generates an internal reflection (a form of synthesized learning data)
        based on the problem, its own response, and the teacher's feedback.
        This helps the student to understand its mistakes and plan for future
        improvements.

        Args:
            problem (str): The original problem statement.
            student_response (str): The student's answer that was evaluated.
            feedback (str): The teacher's feedback on the student's response.

        Returns:
            str: The generated reflection from the LLM, or an error message.
        """
        prompt = (f"You are a student. You attempted to solve the following "
                  f"problem:\nProblem: {problem}\n\n"
                  f"Your response was: {student_response}\n\n"
                  f"The teacher provided the following feedback: {feedback}\n\n"
                  f"Based on this, reflect on what you learned, what you could "
                  f"have done better, and how you will approach similar problems "
                  f"in the future. Be concise.")
        
        logging.info(f"Student is privately reflecting on its performance and "
                     f"the feedback received.")  # noqa: E501

        response = self.ollama_client.generate_response(prompt, model=self.student_model)

        if response:
            logging.info(f"Student's reflection: '{response[:100]}...'")
            return response
        else:
            logging.error(f"Student failed to generate a reflection for problem:"
                          f" '{problem[:50]}...'.")  # noqa: E501
            return "Error: Could not generate reflection."

    def evolve(self, performance_metric: float) -> None:
        """
        Initiates the evolution process for the StudentAgent.
        This method allows the student's learning strategy to adapt over time
        based on its performance.

        Args:
            performance_metric (float): A quantitative measure of the student's
                                        recent performance (e.g., average
                                        reward received).
        """
        logging.info(f"Student is evolving its learning strategy based on its "
                     f"recent performance (metric: {performance_metric:.2f}).")  # noqa: E501
        # Example: if performance is low, maybe it should try to be more
        # detailed or ask more clarifying questions.
        pass


# Example usage block:
# This code runs only when the script is executed directly, not when
# imported as a module.
if __name__ == "__main__":
    # Import necessary modules for the example.
    from src.config import OLLAMA_MODEL

    # Initialize the Ollama client instance using the configured model.
    ollama_client_instance = OllamaClient(model=OLLAMA_MODEL)

    # Proceed only if the Ollama server is running and accessible.
    if ollama_client_instance.is_ollama_running():
        # Instantiate the StudentAgent.
        student = StudentAgent(ollama_client=ollama_client_instance, student_model=OLLAMA_MODEL)

        # --- Test: Solving a Problem ---
        problem_example = "What is the primary function of a neuron?"
        print(f"\n--- Student solving problem: {problem_example} ---")
        student_answer = student.solve_problem(problem_example)
        print(f"Student's Answer:\n{student_answer}")

        # --- Test: Processing and Reflecting on Feedback (simulated) ---
        feedback_example = ("Your answer is correct, but could be more detailed "
                            "about synaptic transmission.")
        student.process_feedback(problem_example, student_answer, feedback_example)
        print(f"Student's reflection (triggered by feedback):\n"
              f"{student.reflect_on_feedback(problem_example, student_answer, feedback_example)}")  # noqa: F541
        
        # --- Test: Evolution ---
        print("\n--- Testing Evolution ---")
        student.evolve(0.90) # Simulate good performance
        student.evolve(0.20) # Simulate poor performance
    else:
        # Inform the user if the Ollama server is not running.
        print("Ollama server is not running or model not available. Please ensure it's set up correctly.")
