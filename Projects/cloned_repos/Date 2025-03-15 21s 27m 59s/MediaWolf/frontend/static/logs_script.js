

document.addEventListener('DOMContentLoaded', () => {
    function fetchLogs() {
        socket.emit("refresh_logs");
    }

    const refreshButton = document.getElementById('refreshLogs');
    refreshButton.addEventListener('click', fetchLogs);

    socket.on("refreshed_logs", function (logs) {
        const logContent = document.getElementById('logContent');
        logContent.innerText = logs;
    });
});