import Dockerode from 'dockerode';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';
import crypto from 'crypto';
import {
  DockerConnectionError,
  ServerDeploymentError,
  ServerNotFoundError
} from '../common/errors.js';
import { Logger } from '../common/logger.js';
import {
  DeployServerInput,
  DeployServerResponse,
  GetLogsInput,
  GetLogsResponse,
  ListServersInput,
  ServerInfo,
  ServerOperationInput,
  ServerOperationResponse
} from '../types/schemas.js';

// Initialize Docker client
const docker = new Dockerode({
  socketPath: process.env.DOCKER_HOST || '/var/run/docker.sock'
});

// Container name prefix to avoid conflicts
const CONTAINER_PREFIX = 'mcp-test-';

// Storage for running server processes (for stdio communication)
interface ServerProcess {
  process: ReturnType<typeof spawn>;
  stdin: NodeJS.WritableStream;
  stdout: NodeJS.ReadableStream;
  name: string;
  source_path: string;
  deployed_at: Date;
}

// Map of running server processes by name
const runningServers = new Map<string, ServerProcess>();

// Deploy a server to a Docker container
export async function deployServer(input: DeployServerInput): Promise<DeployServerResponse> {
  const { name, source_path, env_vars, persistent } = input;
  const containerId = `${CONTAINER_PREFIX}${name}`;
  
  try {
    // Check if source path exists
    if (!fs.existsSync(source_path)) {
      throw new ServerDeploymentError(`Source path not found: ${source_path}`);
    }
    
    // Check if package.json exists
    const packageJsonPath = path.join(source_path, 'package.json');
    if (!fs.existsSync(packageJsonPath)) {
      throw new ServerDeploymentError(`package.json not found in ${source_path}`);
    }
    
    // Check if server with this name already exists
    if (runningServers.has(name)) {
      throw new ServerDeploymentError(`Server with name '${name}' is already running`);
    }
    
    // Read package.json to determine how to run the server
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
    
    // Determine the start command
    let startCommand = 'node dist/index.js';
    if (packageJson.scripts && packageJson.scripts.start) {
      startCommand = 'npm run start';
    }
    
    // Set up environment variables
    const environment: Record<string, string> = {};
    
    // Add current environment, filtering out undefined values
    if (process.env) {
      Object.entries(process.env).forEach(([key, value]) => {
        if (value !== undefined) {
          environment[key] = value;
        }
      });
    }
    
    // Add provided environment variables
    if (env_vars) {
      Object.entries(env_vars).forEach(([key, value]) => {
        environment[key] = value;
      });
    }
    
    // Start the server process
    Logger.info(`Starting server '${name}' from ${source_path}`);
    const serverProcess = spawn('sh', ['-c', startCommand], {
      cwd: source_path,
      env: environment,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Set up logging
    const logDir = path.join(process.cwd(), 'logs');
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
    
    const logFile = path.join(logDir, `${name}.log`);
    const logStream = fs.createWriteStream(logFile, { flags: 'a' });
    
    serverProcess.stdout.pipe(logStream);
    serverProcess.stderr.pipe(logStream);
    
    // Store server process information
    const deployedAt = new Date();
    runningServers.set(name, {
      process: serverProcess,
      stdin: serverProcess.stdin,
      stdout: serverProcess.stdout,
      name,
      source_path,
      deployed_at: deployedAt
    });
    
    // Handle process exit
    serverProcess.on('exit', (code) => {
      Logger.info(`Server '${name}' exited with code ${code}`);
      
      // If not persistent, remove from running servers
      if (!persistent) {
        runningServers.delete(name);
      }
    });
    
    // Give server time to initialize
    await new Promise((resolve) => setTimeout(resolve, 1000));
    
    return {
      name,
      id: containerId,
      status: 'running'
    };
  } catch (error) {
    Logger.error(`Error deploying server '${name}':`, error);
    
    if (error instanceof ServerDeploymentError) {
      throw error;
    }
    
    throw new ServerDeploymentError(
      error instanceof Error ? error.message : String(error)
    );
  }
}

// Get server process by name
export function getServerProcess(name: string): ServerProcess {
  const server = runningServers.get(name);
  if (!server) {
    throw new ServerNotFoundError(name);
  }
  return server;
}

// Get logs from a server
export async function getServerLogs(input: GetLogsInput): Promise<GetLogsResponse> {
  const { server_name, lines } = input;
  
  try {
    // Make sure server exists
    getServerProcess(server_name);
    
    // Read log file
    const logDir = path.join(process.cwd(), 'logs');
    const logFile = path.join(logDir, `${server_name}.log`);
    
    if (!fs.existsSync(logFile)) {
      return {
        logs: `No logs found for server '${server_name}'`
      };
    }
    
    // Read the last N lines from log file
    const logs = await readLastLines(logFile, lines);
    
    return {
      logs
    };
  } catch (error) {
    Logger.error(`Error getting logs for server '${server_name}':`, error);
    
    if (error instanceof ServerNotFoundError) {
      throw error;
    }
    
    return {
      logs: "",
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

// Read last N lines from a file
async function readLastLines(filePath: string, lineCount: number): Promise<string> {
  return new Promise((resolve, reject) => {
    try {
      const lines: string[] = [];
      
      // Create read stream with high water mark to avoid loading too much at once
      const stream = fs.createReadStream(filePath, {
        encoding: 'utf-8',
        highWaterMark: 1024  // 1KB chunks
      });
      
      let buffer = '';
      
      stream.on('data', (chunk) => {
        buffer += chunk.toString();
        const linesToAdd = buffer.split('\n');
        buffer = linesToAdd.pop() || '';
        
        lines.push(...linesToAdd);
        
        // Keep only the last N+1 lines (accounting for potential incomplete line in buffer)
        if (lines.length > lineCount) {
          lines.splice(0, lines.length - lineCount);
        }
      });
      
      stream.on('end', () => {
        // Add any remaining content in buffer
        if (buffer.length > 0) {
          lines.push(buffer);
        }
        
        // Return the last N lines
        resolve(lines.slice(-lineCount).join('\n'));
      });
      
      stream.on('error', (err) => {
        reject(err);
      });
    } catch (error) {
      reject(error);
    }
  });
}

// List running servers
export async function listServers(input: ListServersInput): Promise<ServerInfo[]> {
  try {
    return Array.from(runningServers.entries()).map(([name, server]) => {
      return {
        name,
        id: `${CONTAINER_PREFIX}${name}`,
        status: 'running',
        source_path: server.source_path,
        deployed_at: server.deployed_at.toISOString()
      };
    });
  } catch (error) {
    Logger.error('Error listing servers:', error);
    throw new DockerConnectionError(
      error instanceof Error ? error.message : String(error)
    );
  }
}

// Stop a server
export async function stopServer(input: ServerOperationInput): Promise<ServerOperationResponse> {
  const { server_name } = input;
  
  try {
    const server = getServerProcess(server_name);
    
    // Kill the process
    server.process.kill();
    
    // Remove from running servers
    runningServers.delete(server_name);
    
    return {
      name: server_name,
      status: 'stopped'
    };
  } catch (error) {
    Logger.error(`Error stopping server '${server_name}':`, error);
    
    if (error instanceof ServerNotFoundError) {
      throw error;
    }
    
    throw new Error(
      error instanceof Error ? error.message : String(error)
    );
  }
}