const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});

function renderTasks(tasks) {
    const tableBody = document.getElementById('tasks-table');
    tableBody.innerHTML = '';

    tasks.forEach(task => {
        const template = document.getElementById('task-template');
        const newRow = template.content.cloneNode(true);

        newRow.querySelector('.task-name').textContent = task.name;
        newRow.querySelector('.task-cron').textContent = task.cron;
        newRow.querySelector('.task-status').textContent = task.status;

        const row = newRow.querySelector('tr');
        row.id = `task-row-${task.id}`;

        newRow.querySelector('.btn-primary').addEventListener('click', () => manualStart(task.id));
        newRow.querySelector('.btn-warning').addEventListener('click', () => pauseTask(task.id));
        newRow.querySelector('.btn-danger').addEventListener('click', () => stopTask(task.id));
        newRow.querySelector('.btn-secondary').addEventListener('click', () => cancelTask(task.id));
        newRow.querySelector('.btn-info').addEventListener('click', () => disableTask(task.id));

        tableBody.appendChild(newRow);
    });
}

function manualStart(taskId) {
    socket.emit("task_manual_start", taskId);
}

function pauseTask(taskId) {
    socket.emit("task_pause", taskId);
}

function stopTask(taskId) {
    socket.emit("task_stop", taskId);
}

function cancelTask(taskId) {
    socket.emit("task_cancel", taskId);
}

function disableTask(taskId) {
    socket.emit("task_disable", taskId);
}

socket.on('load_task_data', (tasks) => {
    renderTasks(tasks);
});

socket.on('update_task', (task) => {
    const row = document.querySelector(`#task-row-${task.id}`);

    if (row) {
        row.querySelector('.task-name').textContent = task.name;
        row.querySelector('.task-cron').textContent = task.cron;
        row.querySelector('.task-status').textContent = task.status;
    }
});

function initializeTasksPage() {
    socket.emit("request_tasks")
}