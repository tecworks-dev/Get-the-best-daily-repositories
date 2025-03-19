# Unity MCP Package

A Unity package that enables seamless communication between Unity and Large Language Models (LLMs) like Claude Desktop via the **Model Context Protocol (MCP)**. This server acts as a bridge, allowing Unity to send commands to and receive responses from MCP-compliant tools, empowering developers to automate workflows, manipulate assets, and control the Unity Editor programmatically.

Welcome to the initial release of this open-source project! Whether you're looking to integrate LLMs into your Unity workflow or contribute to an exciting new tool, we're thrilled to have you here.

## Overview

The Unity MCP Server provides a bidirectional communication channel between Unity (via C#) and a Python server, enabling:

- **Asset Management**: Create, import, and manipulate Unity assets programmatically.
- **Scene Control**: Manage scenes, objects, and their properties.
- **Material Editing**: Modify materials and their properties.
- **Script Integration**: View, create, and update Unity scripts.
- **Editor Automation**: Control Unity Editor functions like undo, redo, play, and build.

This project is perfect for developers who want to leverage LLMs to enhance their Unity projects or automate repetitive tasks.

## Installation

### Prerequisites

- Unity 2020.3 LTS or newer
- Python 3.7 or newer
- uv package manager

**If you're on Mac, please install uv as**

```bash
brew install uv
```

**On Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

and then add to your PATH:

```bash
set Path=%USERPROFILE%\.local\bin;%Path%
```

**On Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Otherwise, installation instructions are on their website: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

**⚠️ Do not proceed before installing UV**

### Unity Package Installation

1. **Add the Unity Package**

   - Open Unity Package Manager (`Window > Package Manager`)
   - Click the `+` button and select `Add package from git URL`
   - Enter: `https://github.com/justinpbarnett/unity-mcp.git`

2. **Set Up Python Environment**
   - Navigate to the Python directory in your project:
     - If installed as a package: `Library/PackageCache/com.justinpbarnett.unity-mcp/Python`
     - If installed locally: `Assets/unity-mcp/Python`
   - Install dependencies:
     ```bash
     uv venv
     uv pip install -e .
     ```

### Claude Desktop Integration

1. Open the Unity MCP window (`Window > Unity MCP`)
2. Click the "Configure Claude" button
3. Follow the on-screen instructions to set up the integration

Alternatively, manually configure Claude Desktop:

1. Go to Claude > Settings > Developer > Edit Config
2. Edit `claude_desktop_config.json` to include:

```json
{
  "mcpServers": {
    "unityMCP": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/unity-mcp/Python",
        "run",
        "server.py"
      ]
    }
  }
}
```

Replace `/path/to/your/unity-mcp/Python` with the actual path to the Unity MCP Python directory.

### Cursor Integration

1. Open the Unity MCP window (`Window > Unity MCP`)
2. Click the "Configure Cursor" button
3. Follow the on-screen instructions to set up the integration

Alternatively, go to Cursor Settings > MCP and paste this as a command:

```bash
uv --directory "/path/to/your/unity-mcp/Python" run server.py
```

Replace `/path/to/your/unity-mcp/Python` with the actual path to the Unity MCP Python directory.

**⚠️ Only run one instance of the MCP server (either on Cursor or Claude Desktop), not both**

4. **Start Claude Desktop or Cursor**
   - Launch your preferred tool
   - The Unity MCP Server will automatically connect

## Configuration

To connect the MCP Server to tools like Claude Desktop or Cursor:

1. **Open the Unity MCP Window**  
   In Unity, go to `Window > Unity MCP` to open the editor window.

2. **Configure Your Tools**

   - In the Unity MCP window, you'll see buttons to configure **Claude Desktop** or **Cursor**.
   - Click the appropriate button and follow the on-screen instructions to set up the integration.

3. **Verify Server Status**
   - Check the server status in the Unity MCP window. It will display:
     - **Unity Bridge**: Should show "Running" when active.
     - **Python Server**: Should show "Connected" (green) when successfully linked.

## Manual Configuration for MCP Clients

If you prefer to manually configure your MCP client (like Claude Desktop or Cursor), you can create the configuration file yourself:

1. **Locate the Configuration Directory**

   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. **Create the Configuration File**
   Create a JSON file with the following structure:

   ```json
   {
     "mcpServers": {
       "unityMCP": {
         "command": "uv",
         "args": [
           "--directory",
           "/path/to/your/unity-mcp/Python",
           "run",
           "server.py"
         ]
       }
     }
   }
   ```

3. **Find the Correct Python Path**

   - If installed as a package: Look in `Library/PackageCache/com.justinpbarnett.unity-mcp/Python`
   - If installed locally: Look in `Assets/unity-mcp/Python`

4. **Verify Configuration**
   - Ensure the Python path points to the correct directory containing `server.py`
   - Make sure the `uv` command is available in your system PATH
   - Test the connection using the Unity MCP window

## Usage

Once configured, you can use the MCP Server to interact with LLMs directly from Unity or Python. Here are a couple of examples:

### Creating a Cube in the Scene

```python
# Send a command to create a cube at position (0, 0, 0)
create_primitive(primitive_type="Cube", position=[0, 0, 0])
```

### Changing a Material's Color

```python
# Set a material's color to red (RGBA: 1, 0, 0, 1)
set_material_color(material_name="MyMaterial", color=[1, 0, 0, 1])
```

Explore more commands in the [HOW_TO_ADD_A_TOOL.md](HOW_TO_ADD_A_TOOL.md) file for detailed examples and instructions on extending functionality.

## Features

- **Bidirectional Communication**: Seamlessly send and receive data between Unity and LLMs.
- **Asset Management**: Import assets, instantiate prefabs, and create new prefabs programmatically.
- **Scene Control**: Open, save, and modify scenes, plus create and manipulate game objects.
- **Material Editing**: Apply and modify materials with ease.
- **Script Integration**: Create, view, and update C# scripts within Unity.
- **Editor Automation**: Automate Unity Editor tasks like building projects or entering play mode.

## Contributing

We'd love your help to make the Unity MCP Server even better! Here's how to contribute:

1. **Fork the Repository**  
   Fork [github.com/justinpbarnett/unity-mcp](https://github.com/justinpbarnett/unity-mcp) to your GitHub account.

2. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**  
   Implement your feature or fix, following the project's coding standards (see [HOW_TO_ADD_A_TOOL.md](HOW_TO_ADD_A_TOOL.md) for guidance).

4. **Commit and Push**  
   Use clear, descriptive commit messages:

   ```bash
   git commit -m "Add feature: your feature description"
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request**  
   Open a pull request to the `master` branch. Include a description of your changes and any relevant details.

For more details, check out [CONTRIBUTING.md](CONTRIBUTING.md) (to be created).

## License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it as you see fit. See the full license [here](https://github.com/justinpbarnett/unity-mcp/blob/master/LICENSE).

## Troubleshooting

Encountering issues? Here are some common fixes:

- **Unity Bridge Not Running**  
  Ensure the Unity Editor is open and the MCP window is active. Restart Unity if needed.

- **Python Server Not Connected**

  - Verify the Python server is running (`python server.py` in the `Python` directory).
  - Check `config.json` for correct port settings (default: `unity_port: 6400`, `mcp_port: 6500`).
  - Ensure `uv` and dependencies are installed correctly.

- **Configuration Issues with Claude Desktop or Cursor**  
  Confirm the paths and settings in the configuration dialog match your tool's installation.

For additional help, check the [issue tracker](https://github.com/justinpbarnett/unity-mcp/issues) or file a new issue.

## Contact

Have questions or want to chat about the project? Reach out!

- **X**: [@justinpbarnett](https://x.com/justinpbarnett)
- **GitHub**: [justinpbarnett](https://github.com/justinpbarnett)
- **Discord**: Join our community (link coming soon!).

## Acknowledgments

A huge thanks to everyone who's supported this project's initial release. Special shoutout to Unity Technologies for inspiring tools that push creative boundaries, and to the open-source community for making projects like this possible.

Happy coding, and enjoy integrating LLMs with Unity!
