import os
import sys
import logging
from threading import Thread
from queue import Queue

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Load environment variables (if any)
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import simulation components
from src.ollama_client import OllamaClient
from src.agents.teacher_agent import TeacherAgent
from src.agents.student_agent import StudentAgent
from src.q_learning_framework import QLearningFramework
from src.learning_environment import LearningEnvironment
from src.config import OLLAMA_MODEL

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key') # Use a strong secret key in production
socketio = SocketIO(app)

# --- Logging Setup ---
# Create a queue to hold log messages
log_queue = Queue()

class QueueHandler(logging.Handler):
    """Custom logging handler to put messages into a queue."""
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        self.queue.put(self.format(record))

# Configure logging: clear existing handlers and add QueueHandler to both root and Flask app loggers
# This ensures all log messages, including those from Flask and custom modules, go to the queue.
queue_handler = QueueHandler(log_queue)
queue_handler.setLevel(logging.INFO) # Set minimum level for messages to be handled

# Clear handlers from root logger
logging.getLogger().handlers = []
logging.getLogger().addHandler(queue_handler)
logging.getLogger().setLevel(logging.INFO) # Set default level for all loggers

# Clear handlers from Flask's default logger and add our queue handler
# This prevents duplicate messages if Flask adds its own console handler.
app.logger.handlers = []
app.logger.addHandler(queue_handler)
app.logger.setLevel(logging.INFO)

# --- Simulation Management ---
simulation_thread = None
simulation_running = False
simulation_progress = {'current_round': 0, 'total_rounds': 0, 'status': 'idle'}

def run_simulation_task(num_rounds, topics, evolution_interval):
    """Task to run the simulation in a separate thread."""
    global simulation_running
    simulation_running = True
    
    # Re-initialize logging for this thread to include console output if desired
    # For now, just rely on the queue handler for web display
    
    try:
        # Initialize Ollama client
        ollama_client_instance = OllamaClient(model=OLLAMA_MODEL)
        if not ollama_client_instance.is_ollama_running():
            logging.error("Ollama server not running. Simulation aborted.")
            socketio.emit('simulation_status', {'status': 'error', 'message': 'Ollama server not running.'})
            simulation_running = False
            return

        teacher = TeacherAgent(ollama_client=ollama_client_instance)
        student = StudentAgent(ollama_client=ollama_client_instance)
        possible_student_q_actions = ["answer_concisely", "answer_in_detail", "ask_for_clarification"]
        q_learner = QLearningFramework(actions=possible_student_q_actions, epsilon=0.5)
        env = LearningEnvironment(teacher_agent=teacher, student_agent=student, q_learner=q_learner)

        logging.info("Simulation initialized. Starting rounds...")
        socketio.emit('simulation_status', {'status': 'running', 'message': 'Simulation started.'})

        simulation_results = env.run_simulation(num_rounds, topics, evolution_interval, socketio=socketio)
        
        logging.info("Simulation completed.")
        socketio.emit('simulation_status', {'status': 'completed', 'message': 'Simulation finished successfully.'})
        socketio.emit('simulation_results', {'summary': env.get_summary_statistics(), 'rounds': simulation_results})

    except Exception as e:
        logging.error(f"Simulation encountered an error: {e}", exc_info=True)
        socketio.emit('simulation_status', {'status': 'error', 'message': f'Simulation error: {e}'})
    finally:
        simulation_running = False

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon.ico file."""
    return send_from_directory(os.path.join(app.root_path, 'static', 'img'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@socketio.on('start_simulation')
def start_simulation(data):
    """Handle request to start the simulation."""
    global simulation_thread, simulation_running
    if simulation_running:
        emit('simulation_status', {'status': 'info', 'message': 'Simulation already running.'})
        return

    num_rounds = int(data.get('num_rounds', 10))
    topics_str = data.get('topics', "Reinforcement Learning,Neural Networks,Generative AI,Computer Vision")
    topics = [t.strip() for t in topics_str.split(',')]
    evolution_interval = int(data.get('evolution_interval', 5))

    emit('simulation_status', {'status': 'info', 'message': 'Starting simulation...'})
    simulation_thread = Thread(target=run_simulation_task, args=(num_rounds, topics, evolution_interval))
    simulation_thread.daemon = True # Allow main program to exit even if thread is running
    simulation_thread.start()

@socketio.on('connect')
def test_connect():
    """Handle new client connections."""
    emit('my response', {'data': 'Connected'})
    emit('simulation_status', {'status': simulation_progress['status'], 
                               'current_round': simulation_progress['current_round'],
                               'total_rounds': simulation_progress['total_rounds']})
    # Immediately send any buffered logs
    while not log_queue.empty():
        emit('log_message', {'message': log_queue.get()})

@socketio.on('disconnect')
def test_disconnect():
    """Handle client disconnections."""
    print('Client disconnected', request.sid)

# Thread to continuously send log messages from queue to clients
def log_emitter_task():
    while True:
        message = log_queue.get()
        if message is None: # Sentinel value to stop thread
            break
        socketio.emit('log_message', {'message': message})

log_emitter_thread = Thread(target=log_emitter_task)
log_emitter_thread.daemon = True
log_emitter_thread.start()

# --- Run the App ---
if __name__ == '__main__':
    # Use Flask's own reloader, but ensure SocketIO handles threads correctly
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5000)

