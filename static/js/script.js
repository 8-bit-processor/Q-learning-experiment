document.addEventListener('DOMContentLoaded', (event) => {
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    const logDisplay = document.getElementById('log_display');
    const startSimulationBtn = document.getElementById('start_simulation');
    const numRoundsInput = document.getElementById('num_rounds');
    const topicsInput = document.getElementById('topics');
    const evolutionIntervalInput = document.getElementById('evolution_interval');

    // Function to append log messages
    function appendLog(message, level = 'INFO') {
        const p = document.createElement('p');
        p.className = `log-message ${level}`; // Add level class for styling
        p.textContent = message;
        logDisplay.appendChild(p);
        logDisplay.scrollTop = logDisplay.scrollHeight; // Auto-scroll to bottom
    }

    // Socket.IO event handlers
    socket.on('connect', function() {
        appendLog('Connected to the server.', 'INFO');
        socket.emit('my event', {data: 'I\'m connected!'});
    });

    socket.on('log_message', function(data) {
        // Parse message to extract level, assuming format: "YYYY-MM-DD HH:MM:SS,ms - LEVEL - Message"
        const parts = data.message.split(' - ');
        let level = 'INFO'; // Default level
        if (parts.length > 1) {
            const levelPart = parts[1].trim();
            if (['INFO', 'WARNING', 'ERROR', 'DEBUG'].includes(levelPart)) {
                level = levelPart;
            }
        }
        appendLog(data.message, level);
    });

    socket.on('simulation_status', function(data) {
        appendLog('Simulation Status: ' + data.message, data.status.toUpperCase());
        if (data.status === 'running') {
            startSimulationBtn.disabled = true;
            startSimulationBtn.textContent = 'Simulation Running...';
        } else if (data.status === 'completed' || data.status === 'error') {
            startSimulationBtn.disabled = false;
            startSimulationBtn.textContent = 'Start Simulation';
        }
    });

    socket.on('simulation_results', function(data) {
        appendLog('\n--- Final Simulation Results ---', 'INFO');
        for (const key in data.summary) {
            appendLog(`${key}: ${JSON.stringify(data.summary[key])}`, 'INFO');
        }
        // You might want to display round data or further analysis here
    });

    // Event listener for the Start Simulation button
    startSimulationBtn.addEventListener('click', function() {
        startSimulationBtn.disabled = true;
        startSimulationBtn.textContent = 'Starting...';
        logDisplay.innerHTML = ''; // Clear previous logs

        const num_rounds = numRoundsInput.value;
        const topics = topicsInput.value;
        const evolution_interval = evolutionIntervalInput.value;

        socket.emit('start_simulation', {
            num_rounds: num_rounds,
            topics: topics,
            evolution_interval: evolution_interval
        });
    });
});
