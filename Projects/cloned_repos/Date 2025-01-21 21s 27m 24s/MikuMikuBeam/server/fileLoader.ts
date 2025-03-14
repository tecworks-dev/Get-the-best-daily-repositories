import fs from "fs";
import { join } from "path";

import { Proxy } from "./lib";

const currentPath = () => {
  const path = process.cwd();
  return path === "/" ? "." : path;
};

const loadFileLines = (filePath: string) => {
  try {
    return fs
      .readFileSync(filePath, "utf8")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .filter((line) => !line.startsWith("#"));
  } catch (err) {
    console.error(`Error reading ${filePath}:`, err);
    return [];
  }
};

export function loadUserAgents() {
  return loadFileLines(join(currentPath(), "data/uas.txt"));
}

export function loadProxies(): Proxy[] {
  const lines = loadFileLines(join(currentPath(), "data/proxies.txt"));
  return lines.map((line) => {
    const [protocol, addr] = line.split("://");
    const [host, port] = addr.split(":");
    return { protocol, host, port: parseInt(port) };
  });
}
