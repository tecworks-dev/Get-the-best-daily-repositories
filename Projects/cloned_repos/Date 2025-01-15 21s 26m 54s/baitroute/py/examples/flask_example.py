from flask import Flask
from baitroute import BaitRoute, Alert
from baitroute.flask_integration import register_with_flask

app = Flask(__name__)

# Create a baitroute instance with rules from the rules directory
# You can also specify specific rules to load instead of all:
# baitroute = BaitRoute("../../rules", selected_rules=["exposures/aws-credentials", "exposures/circleci-ssh-config"])
baitroute = BaitRoute("../../rules")

# Set up alert handler
# This is a simple console logging handler, but you can implement more sophisticated handlers:
# - Send alerts to Sentry:
#   def handle_bait_hit(alert):
#       sentry_sdk.capture_message(
#           f"Bait endpoint hit: {alert.path}",
#           extras={
#               "remote_addr": alert.remote_addr,
#               "headers": alert.headers,
#               "body": alert.body
#           }
#       )
#
# - Send to Exabeam:
#   def handle_bait_hit(alert):
#       exabeam_client.send_event({
#           "eventType": "BAIT_HIT",
#           "sourceAddress": alert.remote_addr,
#           "targetAsset": alert.path,
#           "headers": alert.headers,
#           "body": alert.body
#       })
#
# - Send to Splunk:
#   def handle_bait_hit(alert):
#       splunk_client.send(
#           json.dumps({
#               "event": "bait_hit",
#               "remote_addr": alert.remote_addr,
#               "path": alert.path,
#               "headers": alert.headers,
#               "body": alert.body
#           })
#       )
def handle_bait_hit(alert: Alert):
    print(f"🚨 Bait endpoint hit detected!")
    print(f"Path: {alert.path}")
    print(f"Method: {alert.method}")
    print(f"Remote Address: {alert.remote_addr}")
    print(f"Headers: {alert.headers}")
    if alert.body:
        print(f"Body: {alert.body}")
    print("---")

baitroute.on_bait_hit(handle_bait_hit)

# Register bait endpoints
register_with_flask(app, baitroute)

# Your normal routes
@app.route('/')
def home():
    return 'Welcome to the real application!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8087) 