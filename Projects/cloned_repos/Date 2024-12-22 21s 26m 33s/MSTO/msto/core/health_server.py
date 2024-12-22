"""Health check server module for exposing health check endpoints."""

import json
import logging
import threading
import http.server
from typing import Optional

from msto.core.health import HealthCheckHandler

logger = logging.getLogger(__name__)

class HealthServer(http.server.BaseHTTPRequestHandler):
    """HTTP server for health check endpoints."""

    health_handler: Optional[HealthCheckHandler] = None

    def __init__(self, *args, **kwargs):
        """Initialize the health check server."""
        self.protocol_version = "HTTP/1.1"
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Override to use our JSON logger."""
        logger.debug(json.dumps({
            "level": "DEBUG",
            "message": "Health check request",
            "method": self.command,
            "path": self.path,
            "status": args[1]
        }))

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            if not self.health_handler:
                self.send_error(503, "Health handler not initialized")
                return

            status = self.health_handler.get_status()
            status_code = 200 if status.status == "healthy" else 503

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            response = status.to_json()
            self.wfile.write(response.encode())

            logger.debug(json.dumps({
                "level": "DEBUG",
                "message": "Health check response",
                "status_code": status_code,
                "health_status": json.loads(response)
            }))
        else:
            self.send_error(404, "Not Found")

    def do_HEAD(self):
        """Handle HEAD requests for load balancers."""
        if self.path == "/health":
            if not self.health_handler:
                self.send_error(503, "Health handler not initialized")
                return

            status = self.health_handler.get_status()
            status_code = 200 if status.status == "healthy" else 503

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

def start_health_server(health_handler: HealthCheckHandler, port: int = 8080) -> http.server.HTTPServer:
    """
    Start the health check server in a separate thread.

    Args:
        health_handler: The health check handler instance
        port: Port number to listen on

    Returns:
        The started server instance
    """
    try:
        # Set the health handler for the server class
        HealthServer.health_handler = health_handler

        # Create and start the server
        server = http.server.HTTPServer(("", port), HealthServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        logger.info(json.dumps({
            "level": "INFO",
            "message": "Health check server started",
            "port": port,
            "thread": thread.name
        }))

        return server
    except Exception as e:
        logger.error(json.dumps({
            "level": "ERROR",
            "message": "Failed to start health check server",
            "error": str(e),
            "port": port
        }))
        raise 