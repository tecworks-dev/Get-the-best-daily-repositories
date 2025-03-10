import { z } from 'zod';

// Server deployment schema
export const DeployServerSchema = z.object({
  name: z.string().describe("Name for the deployed server"),
  source_path: z.string().describe("Absolute path to the server source code"),
  env_vars: z.record(z.string()).optional().describe("Environment variables to pass to the server"),
  persistent: z.boolean().optional().default(true).describe("Whether to keep the server running after tests")
});

export type DeployServerInput = z.infer<typeof DeployServerSchema>;
export type DeployServerResponse = {
  name: string;
  id: string;
  status: "running" | "error";
  url?: string;
  error?: string;
};

// Tool call schema
export const CallToolSchema = z.object({
  server_name: z.string().describe("Name of the deployed server to call"),
  tool_name: z.string().describe("Name of the tool to call"),
  arguments: z.record(z.any()).describe("Arguments to pass to the tool")
});

export type CallToolInput = z.infer<typeof CallToolSchema>;
export type CallToolResponse = {
  result: any;
  error?: string;
  duration_ms: number;
};

// Server logs schema
export const GetLogsSchema = z.object({
  server_name: z.string().describe("Name of the deployed server"),
  lines: z.number().optional().default(100).describe("Number of log lines to return")
});

export type GetLogsInput = z.infer<typeof GetLogsSchema>;
export type GetLogsResponse = {
  logs: string;
  error?: string;
};

// List servers schema
export const ListServersSchema = z.object({
  status: z.enum(["running", "all"]).optional().default("running").describe("Status of servers to list")
}).optional().default({
  status: "running"
});

export type ListServersInput = z.infer<typeof ListServersSchema>;
export type ServerInfo = {
  name: string;
  id: string;
  status: string;
  source_path: string;
  deployed_at: string;
};

// Test execution schema
export const RunTestsSchema = z.object({
  server_name: z.string().describe("Name of the deployed server to test"),
  test_suite: z.string().optional().describe("Name of the test suite to run"),
  interactive: z.boolean().optional().default(false).describe("Whether to run tests interactively")
});

export type RunTestsInput = z.infer<typeof RunTestsSchema>;
export type TestResult = {
  name: string;
  passed: boolean;
  message?: string;
  duration_ms: number;
  error?: string;
};

export type RunTestsResponse = {
  results: TestResult[];
  summary: {
    total: number;
    passed: number;
    failed: number;
    duration_ms: number;
  };
};

// Server operation schema
export const ServerOperationSchema = z.object({
  server_name: z.string().describe("Name of the deployed server")
});

export type ServerOperationInput = z.infer<typeof ServerOperationSchema>;
export type ServerOperationResponse = {
  name: string;
  status: string;
};

// Test case definition
export type TestCase = {
  name: string;
  description: string;
  tool: string;
  input: Record<string, any>;
  expected?: {
    type: "contains" | "equals" | "regex";
    value: any;
  };
};

// Test suite definition
export type TestSuite = {
  name: string;
  description: string;
  tests: TestCase[];
};