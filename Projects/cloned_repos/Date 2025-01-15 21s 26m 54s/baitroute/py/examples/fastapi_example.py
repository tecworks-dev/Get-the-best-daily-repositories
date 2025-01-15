from fastapi import FastAPI
from baitroute import BaitRoute, Alert
from baitroute.fastapi_integration import register_with_fastapi
import uvicorn

app = FastAPI()

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
#               "source_ip": alert.source_ip,
#               "rule": alert.rule_name,
#               "request_data": alert.request_data
#           }
#       )
#
# - Send to Exabeam:
#   def handle_bait_hit(alert):
#       exabeam_client.send_event({
#           "eventTime": alert.timestamp.isoformat(),
#           "eventType": "BAIT_HIT",
#           "sourceAddress": alert.source_ip,
#           "targetAsset": alert.path,
#           "additionalData": alert.request_data
#       })
#
# - Send to Splunk:
#   def handle_bait_hit(alert):
#       splunk_client.send(
#           json.dumps({
#               "time": alert.timestamp.timestamp(),
#               "event": "bait_hit",
#               "src_ip": alert.source_ip,
#               "path": alert.path,
#               "data": alert.request_data
#           })
#       )
def handle_bait_hit(alert: Alert):
    print(f"🚨 Bait hit detected!")
    print(f"Path: {alert.path}")
    print(f"Method: {alert.method}")
    print(f"Remote Address: {alert.remote_addr}")
    print(f"Headers: {alert.headers}")
    if alert.body:
        print(f"Body: {alert.body}")
    print("---")

baitroute.on_bait_hit(handle_bait_hit)

# Register baitroute endpoints
register_with_fastapi(app, baitroute)

# Your normal routes
@app.get("/")
async def home():
    return {"message": "Welcome to the real application!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087) 