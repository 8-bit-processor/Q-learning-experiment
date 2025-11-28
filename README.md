# Q-Learning Teacher-Student Model with Evolving Agents

## Project Overview

This project implements a novel approach to Q-learning where a teacher agent and a student agent
interact, learn, and evolve. Both agents utilize Large Language Models (LLMs) served by Ollama
to enhance their intelligence and reasoning capabilities. The core idea is that these agents
not only improve through Q-learning but also evolve to actively synthesize their own learning
data, reducing the reliance on pre-fed datasets and addressing some limitations inherent to traditional LLMs.

## Key Features

*   **Evolving Teacher Agent:** Generates dynamic learning materials, adapts problem difficulty, and synthesizes new topics based on student performance.
*   **Evolving Student Agent:** Solves problems, processes teacher feedback, and generates internal reflections for self-improvement.
*   **Ollama Integration:** Leverages local Large Language Models (LLMs) via Ollama for both teacher and student agents' cognitive functions (e.g., generating problems, answering questions, providing feedback, reflecting).
*   **Foundational Q-Learning Framework:** Implements a Q-learning mechanism that guides the agents' strategies and behaviors based on reward signals from interactions.
*   **Dynamic Learning Environment:** Orchestrates the interaction rounds between the teacher and student, manages the learning flow, and triggers agent evolution.
*   **Self-Synthesized Learning Data:** Agents actively create their own learning data (e.g., new topics by the teacher, self-reflections by the student) to drive continuous learning.
*   **Evaluation Metrics:** The learning environment collects and reports summary statistics of the simulation, including average reward, min/max reward, and topics covered, to assess learning progress.

## Project Structure

```
.
├── src/
│   ├── agents/
│   │   ├── student_agent.py
│   │   └── teacher_agent.py
│   ├── config.py
│   ├── learning_environment.py
│   ├── ollama_client.py
│   └── q_learning_framework.py
├── tests/
│   ├── integration/
│   │   └── test_full_interaction.py
│   ├── test_learning_environment.py
│   ├── test_ollama_client.py
│   ├── test_q_learning_framework.py
│   ├── test_student_agent.py
│   └── test_teacher_agent.py
├── app.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── readme
└── requirements.txt
```

*   `src/agents/`: Contains the `TeacherAgent` and `StudentAgent` implementations.
*   `src/config.py`: Configuration settings for Ollama (host, model) and other global parameters.
*   `src/learning_environment.py`: The core simulation environment where agents interact.
*   `src/ollama_client.py`: Handles communication with the local Ollama server.
*   `src/q_learning_framework.py`: Implements the Q-learning algorithm.
*   `tests/`: Contains unit and integration tests for all components.
*   `app.py`: The main Flask application entry point for the web interface.
*   `static/`: Contains static web assets like CSS (`static/css/style.css`) and JavaScript (`static/js/script.js`).
*   `templates/`: Contains HTML templates served by Flask (`templates/index.html`).
*   `readme`: This documentation file.
*   `requirements.txt`: Lists Python dependencies.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install and Configure Ollama:**
    *   Download and install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull an LLM model (e.g., `llama2`, `gemma:7b`). For example:
        ```bash
        ollama pull llama2
        ```
    *   Ensure the Ollama server is running. You can check its status via the command line or desktop application.
    *   Verify the `OLLAMA_MODEL` setting in `src/config.py` matches the model you have pulled.

## Usage

To run the web application:

1.  Ensure you have followed the "Setup and Installation" steps.
2.  Start the Flask application from the project root:
    ```bash
    flask --app app run
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000` (or the address displayed in your terminal).
    From there, you can start and monitor the simulation through the web interface.

To run tests:

```bash
python -m unittest discover tests
``````

## Future Work

*   **Advanced Evolution Mechanisms:** Implement more sophisticated evolutionary algorithms for agent strategies and LLM prompt engineering.
*   **Dynamic Difficulty Adjustment:** Enhance the teacher's ability to precisely adapt problem difficulty based on student learning curves.
*   **Complex State and Action Spaces:** Develop richer representations for states and actions, potentially using LLM embeddings or more structured data.
*   **Persistent Learning:** Implement saving and loading of Q-tables and agent states to allow for continuous learning across sessions.
*   **Visualizations:** Add plotting capabilities for rewards, Q-value convergence, and topic exploration.
*   **More Sophisticated Reward Functions:** Move beyond simple keyword-based feedback parsing to more nuanced LLM-based reward generation.



✦ To demonstrate the concept of this project for users who do not understand evolving Markov Decision Processes (MDPs),
  the best approach is to use clear analogies, focus on the intuitive aspects of "evolving" and "self-synthesizing," and
  walk through a simplified, interactive narrative.

  Here's a strategy:

  ---

  Demo Title: An AI Teacher-Student Duo: Learning How to Learn

  Core Concept to Convey:
  The project isn't just about an AI learning facts; it's about an AI teacher and an AI student continuously improving
  how they teach and learn, by generating their own study material and feedback, just like smart humans do.

  Analogy to Use (Human Teacher-Student Dynamic):
  Imagine a dedicated human tutor and a motivated human student.
   * The tutor doesn't follow a rigid textbook. They observe how you learn, invent new practice questions tailored to
     your needs, give personalized feedback, and even decide when you're ready for a new topic. Over time, the tutor
     becomes better at teaching you.
   * The student doesn't just passively receive information. They try to answer questions, pay close attention to the
     tutor's feedback, and internally reflect on why they succeeded or failed. Based on these reflections, the student
     actively adjusts their learning strategy for the next time. They learn how to learn.
   * The "evolution" is how both the tutor and the student dynamically adapt and get smarter at their respective roles
     based on their ongoing interactions.

  Key Aspects to Emphasize:
   1. AI with a Purpose: These aren't just chatbots. They have specific goals: one to teach effectively, the other to
      learn efficiently.
   2. Dynamic & Adaptive: Nothing is rigidly pre-programmed. The "curriculum" and "study methods" change on the fly.
   3. Self-Generated Learning Data: This is crucial. The system creates its own practice and feedback loops, rather than
      relying on a fixed dataset. This makes it very flexible and capable of tackling new, unforeseen learning tasks.
   4. "Learning How to Learn": The system doesn't just get better at knowing things; it gets better at the process of
      acquiring knowledge and improving teaching.

  Proposed Demo Flow (Narrative & Simplified Console Output):

  Step 1: Setting the Stage - Meet the AI Duo
   * Explanation: "In our project, we have two main AI 'characters': a Teacher Agent and a Student Agent. They both use
     advanced AI models (like the ones behind ChatGPT, but running right here on your computer using Ollama) as their
     'brains'. They interact within a Learning Environment."

  Step 2: The First Interaction - Teacher Poses a Problem
   * Explanation: "The Teacher Agent starts by creating a practice problem for the student."
   * Console Output (Simplified):

   1     Teacher Agent (AI Tutor): "Explain the concept of 'Reinforcement Learning' in one sentence."
   * Emphasis: "Notice, the teacher just generated that question. It wasn't pulled from a predefined list."

  Step 3: Student Responds
   * Explanation: "The Student Agent then tries to answer the problem using its AI brain."
   * Console Output (Simplified):
   1     Student Agent (AI Learner): "Reinforcement learning is about an agent learning to make decisions by trying
     actions and receiving rewards."

  Step 4: Teacher Evaluates & Gives Feedback
   * Explanation: "The Teacher Agent evaluates the student's answer and provides feedback."
   * Console Output (Simplified):
   1     Teacher Agent (AI Tutor): "Correct! Your answer concisely defines the core idea. Strengths: Clear. Areas for
     Improvement: Could mention the goal of maximizing cumulative reward."

  Step 5: Student Reflects & Learns (The "Learning How to Learn" Part)
   * Explanation: "This is where it gets interesting! The student doesn't just move on. It internally reflects on the
     feedback. It synthesizes its own learning data by thinking: 'What did I do right? What could I do better? How
     should I approach similar questions next time?'"
   * Console Output (Simulated Internal Monologue):

   1     Student Agent (Internal Reflection): "I correctly identified key components. Next time, I will try to include t
     'cumulative reward' aspect in my definition to be more complete."
   * Emphasis: "This self-reflection is critical. The student is actively adapting its strategy for learning based on
     experience."

  Step 6: Evolution - Teacher Adapts Curriculum
   * Explanation: "Now, imagine many such interactions. If the teacher notices the student is consistently doing well on
     basic definitions, it will evolve its teaching strategy."
   * Console Output (Simplified):

   1     Teacher Agent (AI Tutor): "Okay, you've mastered basic definitions. Let's explore: 'How does the
     exploration-exploitation dilemma manifest in Q-learning?'"
   * Emphasis: "The teacher synthesized a new, more advanced topic – the exploration-exploitation dilemma – because it
     detected the student was ready. It's an adaptive curriculum."

  Step 7: Evolution - Student Adapts Strategy
   * Explanation: "Similarly, if the student were consistently struggling, it might evolve its learning strategy by
     deciding to always 'ask for clarification' before attempting an answer, or to try a 'more detailed answer'
     approach."
   * Visual Analogy (Optional): Show a simple graph where "average score" or "learning efficiency" improves over "rounds
     of interaction."

  Summary (Recap for the User):
  "What you just saw is a simplified demonstration of our AI teacher and student agents. They are unique because:
   * They create their own learning materials and feedback.
   * They constantly adapt and improve their strategies for teaching and learning.
   * This allows them to 'learn how to learn' and teach without being restricted by static data or curricula, making
     them very powerful for dynamic and complex learning tasks."

  ---

  This approach uses a clear narrative, reduces technical jargon, and highlights the "evolving" and "self-synthesizing"
  aspects that make this project innovative.
