#!/usr/bin/env node

import readline from 'readline';
import { deployServer, listServers, getServerLogs, stopServer } from './operations/docker.js';
import { callTool, runTests } from './operations/mcp-client.js';
import { Logger } from './common/logger.js';

// Initialize logger
Logger.init();

// Create readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Display menu
function showMenu() {
  console.log('\n=== MCP Test Client CLI ===');
  console.log('1. Deploy server');
  console.log('2. List servers');
  console.log('3. Get server logs');
  console.log('4. Stop server');
  console.log('5. Call tool');
  console.log('6. Run tests');
  console.log('7. Exit');
  rl.question('Select an option: ', handleMenuOption);
}

// Handle menu options
async function handleMenuOption(option: string) {
  try {
    switch (option) {
      case '1':
        await handleDeployServer();
        break;
      case '2':
        await handleListServers();
        break;
      case '3':
        await handleGetLogs();
        break;
      case '4':
        await handleStopServer();
        break;
      case '5':
        await handleCallTool();
        break;
      case '6':
        await handleRunTests();
        break;
      case '7':
        rl.close();
        process.exit(0);
        break;
      default:
        console.log('Invalid option');
        showMenu();
        break;
    }
  } catch (error) {
    console.error('Error:', error);
    showMenu();
  }
}

// Handle deploy server
async function handleDeployServer() {
  rl.question('Server name: ', (name) => {
    rl.question('Source path: ', (source_path) => {
      rl.question('Environment variables (JSON, empty for none): ', async (envVarsJson) => {
        try {
          const env_vars = envVarsJson.trim() ? JSON.parse(envVarsJson) : undefined;
          const result = await deployServer({
            name,
            source_path,
            env_vars,
            persistent: true
          });
          
          console.log('Server deployed successfully:');
          console.log(JSON.stringify(result, null, 2));
        } catch (error) {
          console.error('Error deploying server:', error);
        }
        
        showMenu();
      });
    });
  });
}

// Handle list servers
async function handleListServers() {
  try {
    const servers = await listServers({ status: "running" });
    console.log('Servers:');
    console.log(JSON.stringify(servers, null, 2));
  } catch (error) {
    console.error('Error listing servers:', error);
  }
  
  showMenu();
}

// Handle get logs
async function handleGetLogs() {
  rl.question('Server name: ', async (server_name) => {
    try {
      const result = await getServerLogs({
        server_name,
        lines: 20
      });
      
      console.log('Logs:');
      console.log(result.logs);
    } catch (error) {
      console.error('Error getting logs:', error);
    }
    
    showMenu();
  });
}

// Handle stop server
async function handleStopServer() {
  rl.question('Server name: ', async (server_name) => {
    try {
      const result = await stopServer({
        server_name
      });
      
      console.log('Server stopped:');
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error('Error stopping server:', error);
    }
    
    showMenu();
  });
}

// Handle call tool
async function handleCallTool() {
  rl.question('Server name: ', (server_name) => {
    rl.question('Tool name: ', (tool_name) => {
      rl.question('Arguments (JSON): ', async (argsJson) => {
        try {
          const args = JSON.parse(argsJson);
          const result = await callTool({
            server_name,
            tool_name,
            arguments: args
          });
          
          console.log('Tool call result:');
          console.log(JSON.stringify(result, null, 2));
        } catch (error) {
          console.error('Error calling tool:', error);
        }
        
        showMenu();
      });
    });
  });
}

// Handle run tests
async function handleRunTests() {
  rl.question('Server name: ', async (server_name) => {
    try {
      const result = await runTests({
        server_name,
        interactive: false
      });
      
      console.log('Test results:');
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error('Error running tests:', error);
    }
    
    showMenu();
  });
}

// Start the CLI
console.log('MCP Test Client CLI');
showMenu();

// Handle exit
rl.on('close', () => {
  console.log('Exiting...');
  process.exit(0);
});

// Handle SIGINT (Ctrl+C)
process.on('SIGINT', () => {
  console.log('\nReceived SIGINT. Cleaning up...');
  rl.close();
});

// Handle SIGTERM
process.on('SIGTERM', () => {
  console.log('\nReceived SIGTERM. Cleaning up...');
  rl.close();
});