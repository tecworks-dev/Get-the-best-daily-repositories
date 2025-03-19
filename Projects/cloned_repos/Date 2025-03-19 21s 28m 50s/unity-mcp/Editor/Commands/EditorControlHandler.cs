using UnityEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;
using Newtonsoft.Json.Linq;
using System;
using System.Reflection;
using System.Collections.Generic;
using System.Linq;  // Add LINQ namespace for Select extension method
using System.Globalization;

/// <summary>
/// Handles editor control commands like undo, redo, play, pause, stop, and build operations.
/// </summary>
public static class EditorControlHandler
{
    /// <summary>
    /// Handles editor control commands
    /// </summary>
    public static object HandleEditorControl(JObject @params)
    {
        string command = (string)@params["command"];
        JObject commandParams = (JObject)@params["params"];

        switch (command.ToUpper())
        {
            case "UNDO":
                return HandleUndo();
            case "REDO":
                return HandleRedo();
            case "PLAY":
                return HandlePlay();
            case "PAUSE":
                return HandlePause();
            case "STOP":
                return HandleStop();
            case "BUILD":
                return HandleBuild(commandParams);
            case "EXECUTE_COMMAND":
                return HandleExecuteCommand(commandParams);
            case "READ_CONSOLE":
                return ReadConsole(commandParams);
            default:
                return new { error = $"Unknown editor control command: {command}" };
        }
    }

    private static object HandleUndo()
    {
        Undo.PerformUndo();
        return new { message = "Undo performed successfully" };
    }

    private static object HandleRedo()
    {
        Undo.PerformRedo();
        return new { message = "Redo performed successfully" };
    }

    private static object HandlePlay()
    {
        if (!EditorApplication.isPlaying)
        {
            EditorApplication.isPlaying = true;
            return new { message = "Entered play mode" };
        }
        return new { message = "Already in play mode" };
    }

    private static object HandlePause()
    {
        if (EditorApplication.isPlaying)
        {
            EditorApplication.isPaused = !EditorApplication.isPaused;
            return new { message = EditorApplication.isPaused ? "Game paused" : "Game resumed" };
        }
        return new { message = "Not in play mode" };
    }

    private static object HandleStop()
    {
        if (EditorApplication.isPlaying)
        {
            EditorApplication.isPlaying = false;
            return new { message = "Exited play mode" };
        }
        return new { message = "Not in play mode" };
    }

    private static object HandleBuild(JObject @params)
    {
        string platform = (string)@params["platform"];
        string buildPath = (string)@params["buildPath"];

        try
        {
            BuildTarget target = GetBuildTarget(platform);
            if ((int)target == -1)
            {
                return new { error = $"Unsupported platform: {platform}" };
            }

            BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
            buildPlayerOptions.scenes = GetEnabledScenes();
            buildPlayerOptions.target = target;
            buildPlayerOptions.locationPathName = buildPath;

            BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
            return new
            {
                message = "Build completed successfully",
                summary = report.summary
            };
        }
        catch (System.Exception e)
        {
            return new { error = $"Build failed: {e.Message}" };
        }
    }

    private static object HandleExecuteCommand(JObject @params)
    {
        string commandName = (string)@params["commandName"];
        try
        {
            EditorApplication.ExecuteMenuItem(commandName);
            return new { message = $"Executed command: {commandName}" };
        }
        catch (System.Exception e)
        {
            return new { error = $"Failed to execute command: {e.Message}" };
        }
    }

    /// <summary>
    /// Reads log messages from the Unity Console
    /// </summary>
    /// <param name="params">Parameters containing filtering options</param>
    /// <returns>Object containing console messages filtered by type</returns>
    public static object ReadConsole(JObject @params)
    {
        // Default values for show flags
        bool showLogs = true;
        bool showWarnings = true;
        bool showErrors = true;
        string searchTerm = string.Empty;

        // Get filter parameters if provided
        if (@params != null)
        {
            if (@params["show_logs"] != null) showLogs = (bool)@params["show_logs"];
            if (@params["show_warnings"] != null) showWarnings = (bool)@params["show_warnings"];
            if (@params["show_errors"] != null) showErrors = (bool)@params["show_errors"];
            if (@params["search_term"] != null) searchTerm = (string)@params["search_term"];
        }

        try
        {
            // Get required types and methods via reflection
            Type logEntriesType = Type.GetType("UnityEditor.LogEntries,UnityEditor");
            Type logEntryType = Type.GetType("UnityEditor.LogEntry,UnityEditor");

            if (logEntriesType == null || logEntryType == null)
                return new { error = "Could not find required Unity logging types", entries = new List<object>() };

            // Get essential methods
            MethodInfo getCountMethod = logEntriesType.GetMethod("GetCount", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            MethodInfo getEntryMethod = logEntriesType.GetMethod("GetEntryAt", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic) ??
                                        logEntriesType.GetMethod("GetEntryInternal", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);

            if (getCountMethod == null || getEntryMethod == null)
                return new { error = "Could not find required Unity logging methods", entries = new List<object>() };

            // Get stack trace method if available
            MethodInfo getStackTraceMethod = logEntriesType.GetMethod("GetEntryStackTrace", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                null, new[] { typeof(int) }, null) ?? logEntriesType.GetMethod("GetStackTrace", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                null, new[] { typeof(int) }, null);

            // Get entry count and prepare result list
            int count = (int)getCountMethod.Invoke(null, null);
            var entries = new List<object>();

            // Create LogEntry instance to populate
            object logEntryInstance = Activator.CreateInstance(logEntryType);

            // Find properties on LogEntry type
            PropertyInfo modeProperty = logEntryType.GetProperty("mode") ?? logEntryType.GetProperty("Mode");
            PropertyInfo messageProperty = logEntryType.GetProperty("message") ?? logEntryType.GetProperty("Message");

            // Parse search terms if provided
            string[] searchWords = !string.IsNullOrWhiteSpace(searchTerm) ?
                searchTerm.ToLower().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries) : null;

            // Process each log entry
            for (int i = 0; i < count; i++)
            {
                try
                {
                    // Get log entry at index i
                    var methodParams = getEntryMethod.GetParameters();
                    if (methodParams.Length == 2 && methodParams[1].ParameterType == logEntryType)
                    {
                        getEntryMethod.Invoke(null, new object[] { i, logEntryInstance });
                    }
                    else if (methodParams.Length >= 1 && methodParams[0].ParameterType == typeof(int))
                    {
                        var parameters = new object[methodParams.Length];
                        parameters[0] = i;
                        for (int p = 1; p < parameters.Length; p++)
                        {
                            parameters[p] = methodParams[p].ParameterType.IsValueType ?
                                Activator.CreateInstance(methodParams[p].ParameterType) : null;
                        }
                        getEntryMethod.Invoke(null, parameters);
                    }
                    else continue;

                    // Extract log data
                    int logType = modeProperty != null ?
                        Convert.ToInt32(modeProperty.GetValue(logEntryInstance) ?? 0) : 0;

                    string message = messageProperty != null ?
                        (messageProperty.GetValue(logEntryInstance)?.ToString() ?? "") : "";

                    // If message is empty, try to get it via a field
                    if (string.IsNullOrEmpty(message))
                    {
                        var msgField = logEntryType.GetField("message", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                        if (msgField != null)
                        {
                            object msgValue = msgField.GetValue(logEntryInstance);
                            message = msgValue != null ? msgValue.ToString() : "";
                        }

                        // If still empty, try alternate approach with Console window
                        if (string.IsNullOrEmpty(message))
                        {
                            // Access ConsoleWindow and its data
                            Type consoleWindowType = Type.GetType("UnityEditor.ConsoleWindow,UnityEditor");
                            if (consoleWindowType != null)
                            {
                                try
                                {
                                    // Get Console window instance
                                    var getWindowMethod = consoleWindowType.GetMethod("GetWindow",
                                        BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                                        null, new[] { typeof(bool) }, null) ??
                                        consoleWindowType.GetMethod("GetConsoleWindow",
                                        BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);

                                    if (getWindowMethod != null)
                                    {
                                        object consoleWindow = getWindowMethod.Invoke(null,
                                            getWindowMethod.GetParameters().Length > 0 ? new object[] { false } : null);

                                        if (consoleWindow != null)
                                        {
                                            // Try to find log entries collection
                                            foreach (var prop in consoleWindowType.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
                                            {
                                                if (prop.PropertyType.IsArray ||
                                                   (prop.PropertyType.IsGenericType && prop.PropertyType.GetGenericTypeDefinition() == typeof(List<>)))
                                                {
                                                    try
                                                    {
                                                        var logItems = prop.GetValue(consoleWindow);
                                                        if (logItems != null)
                                                        {
                                                            if (logItems.GetType().IsArray && i < ((Array)logItems).Length)
                                                            {
                                                                var entry = ((Array)logItems).GetValue(i);
                                                                if (entry != null)
                                                                {
                                                                    var entryType = entry.GetType();
                                                                    var entryMessageProp = entryType.GetProperty("message") ??
                                                                                         entryType.GetProperty("Message");
                                                                    if (entryMessageProp != null)
                                                                    {
                                                                        object value = entryMessageProp.GetValue(entry);
                                                                        if (value != null)
                                                                        {
                                                                            message = value.ToString();
                                                                            break;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    catch
                                                    {
                                                        // Ignore errors in this fallback approach
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                catch
                                {
                                    // Ignore errors in this fallback approach
                                }
                            }
                        }

                        // If still empty, try one more approach with log files
                        if (string.IsNullOrEmpty(message))
                        {
                            // This is our last resort - try to get log messages from the most recent Unity log file
                            try
                            {
                                string logPath = string.Empty;

                                // Determine the log file path based on the platform
                                if (Application.platform == RuntimePlatform.WindowsEditor)
                                {
                                    logPath = System.IO.Path.Combine(
                                        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                        "Unity", "Editor", "Editor.log");
                                }
                                else if (Application.platform == RuntimePlatform.OSXEditor)
                                {
                                    logPath = System.IO.Path.Combine(
                                        Environment.GetFolderPath(Environment.SpecialFolder.Personal),
                                        "Library", "Logs", "Unity", "Editor.log");
                                }
                                else if (Application.platform == RuntimePlatform.LinuxEditor)
                                {
                                    logPath = System.IO.Path.Combine(
                                        Environment.GetFolderPath(Environment.SpecialFolder.Personal),
                                        ".config", "unity3d", "logs", "Editor.log");
                                }

                                if (!string.IsNullOrEmpty(logPath) && System.IO.File.Exists(logPath))
                                {
                                    // Read last few lines from the log file
                                    var logLines = ReadLastLines(logPath, 100);
                                    if (logLines.Count > i)
                                    {
                                        message = logLines[logLines.Count - 1 - i];
                                    }
                                }
                            }
                            catch
                            {
                                // Ignore errors in this fallback approach
                            }
                        }
                    }

                    // Get stack trace if method available
                    string stackTrace = "";
                    if (getStackTraceMethod != null)
                    {
                        stackTrace = getStackTraceMethod.Invoke(null, new object[] { i })?.ToString() ?? "";
                    }

                    // Filter by type
                    bool typeMatch = (logType == 0 && showLogs) ||
                                    (logType == 1 && showWarnings) ||
                                    (logType == 2 && showErrors);
                    if (!typeMatch) continue;

                    // Filter by search term
                    bool searchMatch = true;
                    if (searchWords != null && searchWords.Length > 0)
                    {
                        string lowerMessage = message.ToLower();
                        string lowerStackTrace = stackTrace.ToLower();

                        foreach (string word in searchWords)
                        {
                            if (!lowerMessage.Contains(word) && !lowerStackTrace.Contains(word))
                            {
                                searchMatch = false;
                                break;
                            }
                        }
                    }
                    if (!searchMatch) continue;

                    // Add matching entry to results
                    string typeStr = logType == 0 ? "Log" : logType == 1 ? "Warning" : "Error";
                    entries.Add(new
                    {
                        type = typeStr,
                        message = message,
                        stackTrace = stackTrace
                    });
                }
                catch (Exception)
                {
                    // Skip entries that cause errors
                    continue;
                }
            }

            // Return filtered results
            return new
            {
                message = "Console logs retrieved successfully",
                entries = entries,
                total_entries = count,
                filtered_count = entries.Count,
                show_logs = showLogs,
                show_warnings = showWarnings,
                show_errors = showErrors
            };
        }
        catch (Exception e)
        {
            return new
            {
                error = $"Failed to read console logs: {e.Message}",
                entries = new List<object>()
            };
        }
    }

    private static MethodInfo FindMethod(Type type, string[] methodNames)
    {
        foreach (var methodName in methodNames)
        {
            var method = type.GetMethod(methodName,
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            if (method != null)
                return method;
        }
        return null;
    }

    private static BuildTarget GetBuildTarget(string platform)
    {
        BuildTarget target;
        switch (platform.ToLower())
        {
            case "windows": target = BuildTarget.StandaloneWindows64; break;
            case "mac": target = BuildTarget.StandaloneOSX; break;
            case "linux": target = BuildTarget.StandaloneLinux64; break;
            case "android": target = BuildTarget.Android; break;
            case "ios": target = BuildTarget.iOS; break;
            case "webgl": target = BuildTarget.WebGL; break;
            default: target = (BuildTarget)(-1); break; // Invalid target
        }
        return target;
    }

    private static string[] GetEnabledScenes()
    {
        var scenes = new System.Collections.Generic.List<string>();
        for (int i = 0; i < EditorBuildSettings.scenes.Length; i++)
        {
            if (EditorBuildSettings.scenes[i].enabled)
            {
                scenes.Add(EditorBuildSettings.scenes[i].path);
            }
        }
        return scenes.ToArray();
    }

    /// <summary>
    /// Helper method to get information about available properties and fields in a type
    /// </summary>
    private static Dictionary<string, object> GetTypeInfo(Type type)
    {
        var result = new Dictionary<string, object>();

        // Get all public and non-public properties
        var properties = type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic |
                                           BindingFlags.Static | BindingFlags.Instance);
        var propList = new List<string>();
        foreach (var prop in properties)
        {
            propList.Add($"{prop.PropertyType.Name} {prop.Name}");
        }
        result["Properties"] = propList;

        // Get all public and non-public fields
        var fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic |
                                   BindingFlags.Static | BindingFlags.Instance);
        var fieldList = new List<string>();
        foreach (var field in fields)
        {
            fieldList.Add($"{field.FieldType.Name} {field.Name}");
        }
        result["Fields"] = fieldList;

        // Get all public and non-public methods
        var methods = type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic |
                                     BindingFlags.Static | BindingFlags.Instance);
        var methodList = new List<string>();
        foreach (var method in methods)
        {
            if (!method.Name.StartsWith("get_") && !method.Name.StartsWith("set_"))
            {
                var parameters = string.Join(", ", method.GetParameters()
                    .Select(p => $"{p.ParameterType.Name} {p.Name}"));
                methodList.Add($"{method.ReturnType.Name} {method.Name}({parameters})");
            }
        }
        result["Methods"] = methodList;

        return result;
    }

    /// <summary>
    /// Helper method to get all property and field values from an object
    /// </summary>
    private static Dictionary<string, string> GetObjectValues(object obj)
    {
        if (obj == null) return new Dictionary<string, string>();

        var result = new Dictionary<string, string>();
        var type = obj.GetType();

        // Get all property values
        var properties = type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        foreach (var prop in properties)
        {
            try
            {
                var value = prop.GetValue(obj);
                result[$"Property:{prop.Name}"] = value?.ToString() ?? "null";
            }
            catch (Exception)
            {
                result[$"Property:{prop.Name}"] = "ERROR";
            }
        }

        // Get all field values
        var fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        foreach (var field in fields)
        {
            try
            {
                var value = field.GetValue(obj);
                result[$"Field:{field.Name}"] = value?.ToString() ?? "null";
            }
            catch (Exception)
            {
                result[$"Field:{field.Name}"] = "ERROR";
            }
        }

        return result;
    }

    /// <summary>
    /// Reads the last N lines from a file
    /// </summary>
    private static List<string> ReadLastLines(string filePath, int lineCount)
    {
        var result = new List<string>();

        using (var stream = new System.IO.FileStream(filePath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite))
        using (var reader = new System.IO.StreamReader(stream))
        {
            string line;
            var circularBuffer = new List<string>(lineCount);
            int currentIndex = 0;

            // Read all lines keeping only the last N in a circular buffer
            while ((line = reader.ReadLine()) != null)
            {
                if (circularBuffer.Count < lineCount)
                {
                    circularBuffer.Add(line);
                }
                else
                {
                    circularBuffer[currentIndex] = line;
                    currentIndex = (currentIndex + 1) % lineCount;
                }
            }

            // Reorder the circular buffer so that lines are returned in order
            if (circularBuffer.Count == lineCount)
            {
                for (int i = 0; i < lineCount; i++)
                {
                    result.Add(circularBuffer[(currentIndex + i) % lineCount]);
                }
            }
            else
            {
                result.AddRange(circularBuffer);
            }
        }

        return result;
    }
}