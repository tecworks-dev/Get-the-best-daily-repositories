from flask import Blueprint, render_template
from flask_socketio import SocketIO
from services.config_services import LOG_FILE_NAME

logs_bp = Blueprint("logs", __name__)


class LogsAPI:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.setup_routes()
        self.setup_socket_events()

    def setup_routes(self):
        """Define Flask routes."""

        @logs_bp.route("/logs")
        def serve_logs_page():
            try:
                with open(LOG_FILE_NAME, "r") as log_file:
                    logs = log_file.read()

            except Exception as e:
                logs = f"Error loading logs: {str(e)}"

            return render_template("logs.html", logs=logs)

    def setup_socket_events(self):
        """Handle Socket.IO events."""

        @self.socketio.on("refresh_logs")
        def handle_refresh_logs():
            """Reads the latest logs and sends them to the frontend."""
            try:
                with open(LOG_FILE_NAME, "r") as log_file:
                    logs = log_file.read()

            except Exception as e:
                logs = f"Error loading logs: {str(e)}"

            self.socketio.emit("refreshed_logs", logs)

    def get_blueprint(self):
        return logs_bp
