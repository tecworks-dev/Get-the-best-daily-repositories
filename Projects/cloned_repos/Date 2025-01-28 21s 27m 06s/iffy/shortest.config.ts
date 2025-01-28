import type { ShortestConfig } from "@antiwork/shortest";

export default {
  headless: false,
  baseUrl: "http://localhost:3000",
  testPattern: "e2e/**/*.test.ts",
  anthropicKey: process.env.SHORTEST_ANTHROPIC_API_KEY,
} satisfies ShortestConfig;
