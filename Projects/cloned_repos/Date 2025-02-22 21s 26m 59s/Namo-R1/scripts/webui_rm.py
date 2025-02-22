import os
import open_webui

# Show where the Open WebUI package is installed
print("Open WebUI is installed at:", open_webui.__file__)

# Construct a potential path to webui.db (commonly located in 'data/webui.db')
db_path = os.path.join(os.path.dirname(open_webui.__file__), "data", "webui.db")
print("Potential path to webui.db:", db_path)

# Check if webui.db exists at that path
if os.path.exists(db_path):
    print("webui.db found at:", db_path)
else:
    print("webui.db not found at:", db_path)