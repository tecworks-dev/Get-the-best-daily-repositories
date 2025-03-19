using UnityEngine;
using UnityEditor;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System;
using Newtonsoft.Json;
using System.Net.Sockets;
using System.Threading.Tasks;
using System.Text;
using System.Collections.Generic;

public class DefaultServerConfig : ServerConfig
{
    public new string unityHost = "localhost";
    public new int unityPort = 6400;
    public new int mcpPort = 6500;
    public new float connectionTimeout = 15.0f;
    public new int bufferSize = 32768;
    public new string logLevel = "INFO";
    public new string logFormat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s";
    public new int maxRetries = 3;
    public new float retryDelay = 1.0f;

}

[Serializable]
public class MCPConfig
{
    [JsonProperty("mcpServers")]
    public MCPConfigServers mcpServers;
}

[Serializable]
public class MCPConfigServers
{
    [JsonProperty("unityMCP")]
    public MCPConfigServer unityMCP;
}

[Serializable]
public class MCPConfigServer
{
    [JsonProperty("command")]
    public string command;

    [JsonProperty("args")]
    public string[] args;
}

[Serializable]
public class ServerConfig
{
    [JsonProperty("unity_host")]
    public string unityHost = "localhost";

    [JsonProperty("unity_port")]
    public int unityPort;

    [JsonProperty("mcp_port")]
    public int mcpPort;

    [JsonProperty("connection_timeout")]
    public float connectionTimeout;

    [JsonProperty("buffer_size")]
    public int bufferSize;

    [JsonProperty("log_level")]
    public string logLevel;

    [JsonProperty("log_format")]
    public string logFormat;

    [JsonProperty("max_retries")]
    public int maxRetries;

    [JsonProperty("retry_delay")]
    public float retryDelay;
}

public class MCPEditorWindow : EditorWindow
{
    private bool isUnityBridgeRunning = false;
    private Vector2 scrollPosition;
    private string claudeConfigStatus = "Not configured";
    private string pythonServerStatus = "Not Connected";
    private Color pythonServerStatusColor = Color.red;
    private const int unityPort = 6400;  // Hardcoded Unity port
    private const int mcpPort = 6500;    // Hardcoded MCP port
    private const float CONNECTION_CHECK_INTERVAL = 2f; // Check every 2 seconds
    private float lastCheckTime = 0f;

    [MenuItem("Window/Unity MCP")]
    public static void ShowWindow()
    {
        GetWindow<MCPEditorWindow>("MCP Editor");
    }

    private void OnEnable()
    {
        // Check initial states
        isUnityBridgeRunning = UnityMCPBridge.IsRunning;
        CheckPythonServerConnection();
    }

    private void Update()
    {
        // Check Python server connection periodically
        if (Time.realtimeSinceStartup - lastCheckTime >= CONNECTION_CHECK_INTERVAL)
        {
            CheckPythonServerConnection();
            lastCheckTime = Time.realtimeSinceStartup;
        }
    }

    private async void CheckPythonServerConnection()
    {
        try
        {
            using (var client = new TcpClient())
            {
                // Try to connect with a short timeout
                var connectTask = client.ConnectAsync("localhost", unityPort);
                if (await Task.WhenAny(connectTask, Task.Delay(1000)) == connectTask)
                {
                    // Try to send a ping message to verify connection is alive
                    try
                    {
                        NetworkStream stream = client.GetStream();
                        byte[] pingMessage = Encoding.UTF8.GetBytes("ping");
                        await stream.WriteAsync(pingMessage, 0, pingMessage.Length);

                        // Wait for response with timeout
                        byte[] buffer = new byte[1024];
                        var readTask = stream.ReadAsync(buffer, 0, buffer.Length);
                        if (await Task.WhenAny(readTask, Task.Delay(1000)) == readTask)
                        {
                            // Connection successful and responsive
                            pythonServerStatus = "Connected";
                            pythonServerStatusColor = Color.green;
                            UnityEngine.Debug.Log($"Python server connected successfully on port {unityPort}");
                        }
                        else
                        {
                            // No response received
                            pythonServerStatus = "No Response";
                            pythonServerStatusColor = Color.yellow;
                            UnityEngine.Debug.LogWarning($"Python server not responding on port {unityPort}");
                        }
                    }
                    catch (Exception e)
                    {
                        // Connection established but communication failed
                        pythonServerStatus = "Communication Error";
                        pythonServerStatusColor = Color.yellow;
                        UnityEngine.Debug.LogWarning($"Error communicating with Python server: {e.Message}");
                    }
                }
                else
                {
                    // Connection failed
                    pythonServerStatus = "Not Connected";
                    pythonServerStatusColor = Color.red;
                    UnityEngine.Debug.LogWarning($"Python server is not running or not accessible on port {unityPort}");
                }
                client.Close();
            }
        }
        catch (Exception e)
        {
            pythonServerStatus = "Connection Error";
            pythonServerStatusColor = Color.red;
            UnityEngine.Debug.LogError($"Error checking Python server connection: {e.Message}");
        }
    }

    private void OnGUI()
    {
        scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);

        EditorGUILayout.Space(10);
        EditorGUILayout.LabelField("MCP Editor", EditorStyles.boldLabel);
        EditorGUILayout.Space(10);

        // Python Server Status Section
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Python Server Status", EditorStyles.boldLabel);

        // Status bar
        var statusRect = EditorGUILayout.BeginHorizontal();
        EditorGUI.DrawRect(new Rect(statusRect.x, statusRect.y, 10, 20), pythonServerStatusColor);
        EditorGUILayout.LabelField(pythonServerStatus);
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.LabelField($"Unity Port: {unityPort}");
        EditorGUILayout.LabelField($"MCP Port: {mcpPort}");
        EditorGUILayout.HelpBox("Start the Python server using command line: 'uv run server.py' in the Python directory", MessageType.Info);
        EditorGUILayout.EndVertical();

        EditorGUILayout.Space(10);

        // Unity Bridge Section
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Unity MCP Bridge", EditorStyles.boldLabel);
        EditorGUILayout.LabelField($"Status: {(isUnityBridgeRunning ? "Running" : "Stopped")}");
        EditorGUILayout.LabelField($"Port: {unityPort}");

        if (GUILayout.Button(isUnityBridgeRunning ? "Stop Bridge" : "Start Bridge"))
        {
            ToggleUnityBridge();
        }
        EditorGUILayout.EndVertical();

        EditorGUILayout.Space(10);

        // Claude Desktop Configuration Section
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Claude Desktop Configuration", EditorStyles.boldLabel);
        EditorGUILayout.LabelField($"Status: {claudeConfigStatus}");

        if (GUILayout.Button("Configure Claude Desktop"))
        {
            ConfigureClaudeDesktop();
        }
        EditorGUILayout.EndVertical();

        EditorGUILayout.EndScrollView();
    }

    private void ToggleUnityBridge()
    {
        if (isUnityBridgeRunning)
        {
            UnityMCPBridge.Stop();
        }
        else
        {
            UnityMCPBridge.Start();
        }
        isUnityBridgeRunning = !isUnityBridgeRunning;
    }

    private void ConfigureClaudeDesktop()
    {
        try
        {
            // Determine the config file path based on OS
            string configPath;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                configPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "Claude",
                    "claude_desktop_config.json"
                );
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                configPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    "Library",
                    "Application Support",
                    "Claude",
                    "claude_desktop_config.json"
                );
            }
            else
            {
                claudeConfigStatus = "Unsupported OS";
                return;
            }

            // Create directory if it doesn't exist
            Directory.CreateDirectory(Path.GetDirectoryName(configPath));

            // Find the server.py file location
            string serverPath = null;
            string pythonDir = null;

            // List of possible locations to search
            var possiblePaths = new List<string>
            {
                // Search in Assets folder - Manual installation
                Path.GetFullPath(Path.Combine(Application.dataPath, "unity-mcp", "Python", "server.py")),
                Path.GetFullPath(Path.Combine(Application.dataPath, "Packages", "com.justinpbarnett.unity-mcp", "Python", "server.py")),
                
                // Search in package cache - Package manager installation
                Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Library", "PackageCache", "com.justinpbarnett.unity-mcp@*", "Python", "server.py")),
                
                // Search in package manager packages - Git installation
                Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.justinpbarnett.unity-mcp", "Python", "server.py"))
            };

            UnityEngine.Debug.Log("Searching for server.py in the following locations:");

            // First try with explicit paths
            foreach (var path in possiblePaths)
            {
                // Skip wildcard paths for now
                if (path.Contains("*")) continue;

                UnityEngine.Debug.Log($"Checking: {path}");
                if (File.Exists(path))
                {
                    serverPath = path;
                    pythonDir = Path.GetDirectoryName(serverPath);
                    UnityEngine.Debug.Log($"Found server.py at: {serverPath}");
                    break;
                }
            }

            // If not found, try with wildcard paths (package cache with version)
            if (serverPath == null)
            {
                foreach (var path in possiblePaths)
                {
                    if (!path.Contains("*")) continue;

                    string directoryPath = Path.GetDirectoryName(path);
                    string searchPattern = Path.GetFileName(Path.GetDirectoryName(path));
                    string parentDir = Path.GetDirectoryName(directoryPath);

                    if (Directory.Exists(parentDir))
                    {
                        var matchingDirs = Directory.GetDirectories(parentDir, searchPattern);
                        UnityEngine.Debug.Log($"Searching in: {parentDir} for pattern: {searchPattern}, found {matchingDirs.Length} matches");

                        foreach (var dir in matchingDirs)
                        {
                            string candidatePath = Path.Combine(dir, "Python", "server.py");
                            UnityEngine.Debug.Log($"Checking: {candidatePath}");

                            if (File.Exists(candidatePath))
                            {
                                serverPath = candidatePath;
                                pythonDir = Path.GetDirectoryName(serverPath);
                                UnityEngine.Debug.Log($"Found server.py at: {serverPath}");
                                break;
                            }
                        }

                        if (serverPath != null) break;
                    }
                }
            }

            if (serverPath == null || !File.Exists(serverPath))
            {
                ShowManualConfigurationInstructions(configPath);
                return;
            }

            UnityEngine.Debug.Log($"Using server.py at: {serverPath}");
            UnityEngine.Debug.Log($"Python directory: {pythonDir}");

            // Load existing configuration if it exists
            dynamic existingConfig = null;
            if (File.Exists(configPath))
            {
                try
                {
                    string existingJson = File.ReadAllText(configPath);
                    existingConfig = JsonConvert.DeserializeObject(existingJson);
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.LogWarning($"Failed to parse existing Claude config: {ex.Message}. Creating new config.");
                }
            }

            // If no existing config or parsing failed, create a new one
            if (existingConfig == null)
            {
                existingConfig = new
                {
                    mcpServers = new Dictionary<string, object>()
                };
            }

            // Create the Unity MCP server configuration
            var unityMCPConfig = new MCPConfigServer
            {
                command = "uv",
                args = new[]
                {
                    "--directory",
                    pythonDir,
                    "run",
                    "server.py"
                }
            };
            // Add or update the Unity MCP configuration while preserving the rest
            var mcpServers = existingConfig.mcpServers as Newtonsoft.Json.Linq.JObject 
                ?? new Newtonsoft.Json.Linq.JObject();
            
            mcpServers["unityMCP"] = Newtonsoft.Json.Linq.JToken.FromObject(unityMCPConfig);
            existingConfig.mcpServers = mcpServers;
            // Serialize and write to file with proper formatting
            var jsonSettings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented
            };
            string jsonConfig = JsonConvert.SerializeObject(existingConfig, jsonSettings);
            File.WriteAllText(configPath, jsonConfig);

            claudeConfigStatus = "Configured successfully";
            UnityEngine.Debug.Log($"Claude Desktop configuration saved to: {configPath}");
            UnityEngine.Debug.Log($"Configuration contents:\n{jsonConfig}");
        }
        catch (Exception e)
        {
            // Determine the config file path based on OS for error message
            string configPath = "";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                configPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "Claude",
                    "claude_desktop_config.json"
                );
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                configPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    "Library",
                    "Application Support",
                    "Claude",
                    "claude_desktop_config.json"
                );
            }

            ShowManualConfigurationInstructions(configPath);
            UnityEngine.Debug.LogError($"Failed to configure Claude Desktop: {e.Message}\n{e.StackTrace}");
        }
    }

    private void ShowManualConfigurationInstructions(string configPath)
    {
        claudeConfigStatus = "Error: Manual configuration required";

        // Get the Python directory path using Package Manager API
        string pythonDir = FindPackagePythonDirectory();

        // Create the manual configuration message
        var jsonConfig = new MCPConfig
        {
            mcpServers = new MCPConfigServers
            {
                unityMCP = new MCPConfigServer
                {
                    command = "uv",
                    args = new[]
                    {
                        "--directory",
                        pythonDir,
                        "run",
                        "server.py"
                    }
                }
            }
        };

        var jsonSettings = new JsonSerializerSettings
        {
            Formatting = Formatting.Indented
        };
        string manualConfigJson = JsonConvert.SerializeObject(jsonConfig, jsonSettings);

        // Show a dedicated configuration window instead of console logs
        ManualConfigWindow.ShowWindow(configPath, manualConfigJson);
    }

    private string FindPackagePythonDirectory()
    {
        string pythonDir = "/path/to/your/unity-mcp/Python";

        try
        {
            // Try to find the package using Package Manager API
            var request = UnityEditor.PackageManager.Client.List();
            while (!request.IsCompleted) { } // Wait for the request to complete

            if (request.Status == UnityEditor.PackageManager.StatusCode.Success)
            {
                foreach (var package in request.Result)
                {
                    UnityEngine.Debug.Log($"Package: {package.name}, Path: {package.resolvedPath}");

                    if (package.name == "com.justinpbarnett.unity-mcp")
                    {
                        string packagePath = package.resolvedPath;
                        string potentialPythonDir = Path.Combine(packagePath, "Python");

                        if (Directory.Exists(potentialPythonDir) &&
                            File.Exists(Path.Combine(potentialPythonDir, "server.py")))
                        {
                            UnityEngine.Debug.Log($"Found package Python directory at: {potentialPythonDir}");
                            return potentialPythonDir;
                        }
                    }
                }
            }
            else if (request.Error != null)
            {
                UnityEngine.Debug.LogError("Failed to list packages: " + request.Error.message);
            }

            // If not found via Package Manager, try manual approaches
            // First check for local installation
            string[] possibleDirs = {
                Path.GetFullPath(Path.Combine(Application.dataPath, "unity-mcp", "Python"))
            };

            foreach (var dir in possibleDirs)
            {
                UnityEngine.Debug.Log($"Checking local directory: {dir}");
                if (Directory.Exists(dir) && File.Exists(Path.Combine(dir, "server.py")))
                {
                    UnityEngine.Debug.Log($"Found local Python directory at: {dir}");
                    return dir;
                }
            }

            // If still not found, return the placeholder path
            UnityEngine.Debug.LogWarning("Could not find Python directory, using placeholder path");
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"Error finding package path: {e.Message}");
        }

        return pythonDir;
    }
}

// Editor window to display manual configuration instructions
public class ManualConfigWindow : EditorWindow
{
    private string configPath;
    private string configJson;
    private Vector2 scrollPos;
    private bool pathCopied = false;
    private bool jsonCopied = false;
    private float copyFeedbackTimer = 0;

    public static void ShowWindow(string configPath, string configJson)
    {
        var window = GetWindow<ManualConfigWindow>("Manual Configuration");
        window.configPath = configPath;
        window.configJson = configJson;
        window.minSize = new Vector2(500, 400);
        window.Show();
    }

    private void OnGUI()
    {
        scrollPos = EditorGUILayout.BeginScrollView(scrollPos);

        // Header
        EditorGUILayout.Space(10);
        EditorGUILayout.LabelField("Claude Desktop Manual Configuration", EditorStyles.boldLabel);
        EditorGUILayout.Space(10);

        // Instructions
        EditorGUILayout.LabelField("The automatic configuration failed. Please follow these steps:", EditorStyles.boldLabel);
        EditorGUILayout.Space(5);

        EditorGUILayout.LabelField("1. Open Claude Desktop and go to Settings > Developer > Edit Config", EditorStyles.wordWrappedLabel);
        EditorGUILayout.LabelField("2. Create or edit the configuration file at:", EditorStyles.wordWrappedLabel);

        // Config path section with copy button
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.SelectableLabel(configPath, EditorStyles.textField, GUILayout.Height(EditorGUIUtility.singleLineHeight));

        if (GUILayout.Button("Copy Path", GUILayout.Width(80)))
        {
            EditorGUIUtility.systemCopyBuffer = configPath;
            pathCopied = true;
            copyFeedbackTimer = 2f;
        }

        EditorGUILayout.EndHorizontal();

        if (pathCopied)
        {
            EditorGUILayout.LabelField("Path copied to clipboard!", EditorStyles.miniLabel);
        }

        EditorGUILayout.Space(10);

        // JSON configuration
        EditorGUILayout.LabelField("3. Paste the following JSON configuration:", EditorStyles.wordWrappedLabel);
        EditorGUILayout.Space(5);

        EditorGUILayout.LabelField("Make sure to replace the Python path if necessary:", EditorStyles.wordWrappedLabel);
        EditorGUILayout.Space(5);

        // JSON text area with copy button
        GUIStyle textAreaStyle = new GUIStyle(EditorStyles.textArea)
        {
            wordWrap = true,
            richText = true
        };

        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.SelectableLabel(configJson, textAreaStyle, GUILayout.MinHeight(200));
        EditorGUILayout.EndHorizontal();

        if (GUILayout.Button("Copy JSON Configuration"))
        {
            EditorGUIUtility.systemCopyBuffer = configJson;
            jsonCopied = true;
            copyFeedbackTimer = 2f;
        }

        if (jsonCopied)
        {
            EditorGUILayout.LabelField("JSON copied to clipboard!", EditorStyles.miniLabel);
        }

        EditorGUILayout.Space(10);

        // Additional note
        EditorGUILayout.HelpBox("After configuring, restart Claude Desktop to apply the changes.", MessageType.Info);

        EditorGUILayout.EndScrollView();
    }

    private void Update()
    {
        // Handle the feedback message timer
        if (copyFeedbackTimer > 0)
        {
            copyFeedbackTimer -= Time.deltaTime;
            if (copyFeedbackTimer <= 0)
            {
                pathCopied = false;
                jsonCopied = false;
                Repaint();
            }
        }
    }
}