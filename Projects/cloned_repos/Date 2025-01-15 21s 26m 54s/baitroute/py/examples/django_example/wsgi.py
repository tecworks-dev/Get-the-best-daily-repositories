import os
from django.core.wsgi import get_wsgi_application
from baitroute import Alert
from baitroute.django_integration import BaitRouteMiddleware
from django.http import HttpRequest

# Set up alert handler
def handle_bait_hit(alert: Alert, request: HttpRequest = None):
    print(f"🚨 Bait hit detected!")
    print(f"Path: {alert.path}")
    print(f"Method: {alert.method}")
    print(f"Remote Address: {alert.remote_addr}")
    print(f"Headers: {alert.headers}")
    if alert.body:
        print(f"Body: {alert.body}")
    print("---")

# Set the alert handler before initializing the application
BaitRouteMiddleware.default_alert_handler = handle_bait_hit

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'examples.django_example.settings')

# Get the WSGI application
application = get_wsgi_application() 