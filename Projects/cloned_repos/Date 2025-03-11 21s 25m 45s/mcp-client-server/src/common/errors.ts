export class MCPTestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MCPTestError";
  }
}

export class ServerDeploymentError extends MCPTestError {
  constructor(message: string) {
    super(message);
    this.name = "ServerDeploymentError";
  }
}

export class TestExecutionError extends MCPTestError {
  constructor(message: string) {
    super(message);
    this.name = "TestExecutionError";
  }
}

export class ServerNotFoundError extends MCPTestError {
  serverName: string;
  
  constructor(serverName: string) {
    super(`Server '${serverName}' not found in test environment`);
    this.name = "ServerNotFoundError";
    this.serverName = serverName;
  }
}

export class DockerConnectionError extends MCPTestError {
  constructor(message: string) {
    super(`Docker connection error: ${message}`);
    this.name = "DockerConnectionError";
  }
}

export class ToolCallError extends MCPTestError {
  serverName: string;
  toolName: string;
  
  constructor(serverName: string, toolName: string, message: string) {
    super(`Error calling tool '${toolName}' on server '${serverName}': ${message}`);
    this.name = "ToolCallError";
    this.serverName = serverName;
    this.toolName = toolName;
  }
}