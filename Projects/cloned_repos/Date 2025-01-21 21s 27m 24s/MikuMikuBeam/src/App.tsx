import { Bot, Wand2, Wifi, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://localhost:3000");

function App() {
  const [isAttacking, setIsAttacking] = useState(false);
  const [actuallyAttacking, setActuallyAttacking] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [target, setTarget] = useState("");
  const [attackMethod, setAttackMethod] = useState("http_flood");
  const [packetSize, setPacketSize] = useState(64);
  const [duration, setDuration] = useState(60);
  const [packetDelay, setPacketDelay] = useState(100);
  const [stats, setStats] = useState({
    pps: 0,
    bots: 0,
    totalPackets: 0,
  });
  const [lastUpdatedPPS, setLastUpdatedPPS] = useState(Date.now());
  const [lastTotalPackets, setLastTotalPackets] = useState(0);
  const [currentTask, setCurrentTask] = useState<NodeJS.Timeout | null>(null);
  const [audioVol, setAudioVol] = useState(100);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audioRef.current) {
      const audio = audioRef.current;
      const handler = () => {
        if (!audio.paused && audio.currentTime > 17.53) {
          audio.currentTime = 15.86;
        }
      };

      audio.addEventListener("timeupdate", handler);
      return () => {
        audio.removeEventListener("timeupdate", handler);
      };
    }
  }, [audioRef]);

  useEffect(() => {
    if (!isAttacking) {
      setActuallyAttacking(false);

      const audio = audioRef.current;
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }

      if (currentTask) {
        clearTimeout(currentTask);
      }
    }
  }, [isAttacking, currentTask]);

  useEffect(() => {
    const now = Date.now();
    if (now - lastUpdatedPPS >= 500) {
      setLastUpdatedPPS(now);
      setStats((old) => ({
        pps: (old.totalPackets - lastTotalPackets) / (now - lastUpdatedPPS),
        bots: old.bots,
        totalPackets: old.totalPackets,
      }));
      setLastTotalPackets(stats.totalPackets);
    }
  }, [lastUpdatedPPS, lastTotalPackets, stats.totalPackets]);

  useEffect(() => {
    socket.on("stats", (data) => {
      setStats((old) => ({
        pps: data.pps || old.pps,
        bots: data.bots || old.bots,
        totalPackets: data.totalPackets || old.totalPackets,
      }));
      if (data.log) addLog(data.log);
      setProgress((prev) => (prev + 10) % 100);
    });

    socket.on("attackEnd", () => {
      setIsAttacking(false);
    });

    return () => {
      socket.off("stats");
      socket.off("attackEnd");
    };
  }, []);

  
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = audioVol / 100;
    }
  }, [audioVol])
 

  const addLog = (message: string) => {
    setLogs((prev) => [message, ...prev].slice(0, 12));
  };

  const startAttack = (isQuick?: boolean) => {
    if (!target.trim()) {
      alert("Please enter a target!");
      return;
    }

    setIsAttacking(true);
    setStats((old) => ({
      pps: 0,
      bots: old.bots,
      totalPackets: 0,
    }));
    addLog("🍮 Preparing attack...");

    // Play audio
    if (audioRef.current) {
      audioRef.current.currentTime = isQuick ? 9.5 : 0;
      audioRef.current.volume = audioVol / 100;
      audioRef.current.play();
    }

    // Start attack after audio intro
    const timeout = setTimeout(
      () => {
        setActuallyAttacking(true);
        socket.emit("startAttack", {
          target,
          packetSize,
          duration,
          packetDelay,
          attackMethod,
        });
      },
      isQuick ? 700 : 10250
    );
    setCurrentTask(timeout);
  };

  const stopAttack = () => {
    socket.emit("stopAttack");
    setIsAttacking(false);
  };

  return (
    <div
      className={`w-screen h-screen bg-gradient-to-br from-pink-100 to-blue-100 p-8 overflow-y-auto ${
        actuallyAttacking ? "shake" : ""
      }`}
    >
      <audio ref={audioRef} src="/audio.mp3" />

      <div className="max-w-2xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="mb-2 text-4xl font-bold text-pink-500">
            Miku Miku Beam
          </h1>
          <p className="text-gray-600">
            Because DDoS attacks are also cute and even more so when Miku does
            them.
          </p>
        </div>

        <div className="relative p-6 overflow-hidden bg-white rounded-lg shadow-xl">
          {/* Miku GIF */}
          <div
            className="flex justify-center w-full h-48 mb-6"
            style={{
              backgroundImage: "url('/miku.gif')",
              backgroundRepeat: "no-repeat",
              backgroundPosition: "center",
              backgroundSize: "cover",
            }}
          ></div>

          {/* Attack Configuration */}
          <div className="mb-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <input
                type="text"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="Enter target URL or IP"
                className="px-4 py-2 border border-pink-200 rounded-lg outline-none focus:border-pink-500 focus:ring-2 focus:ring-pink-200"
                disabled={isAttacking}
              />
              <div className="flex items-center gap-2">
                <button
                  onClick={() => (isAttacking ? stopAttack() : startAttack())}
                  className={`
                  px-8 py-2 rounded-lg font-semibold text-white transition-all w-full
                  ${
                    isAttacking
                      ? "bg-red-500 hover:bg-red-600"
                      : "bg-pink-500 hover:bg-pink-600"
                  }
                  flex items-center justify-center gap-2
                `}
                >
                  <Wand2 className="w-5 h-5" />
                  {isAttacking ? "Stop Beam" : "Start Miku Beam"}
                </button>
                <button
                  onClick={() =>
                    isAttacking ? stopAttack() : startAttack(true)
                  }
                  className={`
                  px-2 py-2 rounded-lg font-semibold text-white transition-all
                  ${
                    isAttacking
                      ? "bg-gray-500 hover:bg-red-600"
                      : "bg-cyan-500 hover:bg-cyan-600"
                  }
                  flex items-center justify-center gap-2
                `}
                >
                  <Zap className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-4">
              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700">
                  Attack Method
                </label>
                <select
                  value={attackMethod}
                  onChange={(e) => setAttackMethod(e.target.value)}
                  className="w-full px-4 py-2 border border-pink-200 rounded-lg outline-none focus:border-pink-500 focus:ring-2 focus:ring-pink-200"
                  disabled={isAttacking}
                >
                  <option value="http_flood">HTTP/Flood</option>
                  <option value="http_slowloris">HTTP/Slowloris</option>
                  <option value="tcp_flood">TCP/Flood</option>
                  <option value="minecraft_ping">Minecraft/Ping</option>
                </select>
              </div>
              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700">
                  Packet Size (kb)
                </label>
                <input
                  type="number"
                  value={packetSize}
                  onChange={(e) => setPacketSize(Number(e.target.value))}
                  className="w-full px-4 py-2 border border-pink-200 rounded-lg outline-none focus:border-pink-500 focus:ring-2 focus:ring-pink-200"
                  disabled={isAttacking}
                  min="1"
                  max="1500"
                />
              </div>
              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700">
                  Duration (seconds)
                </label>
                <input
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="w-full px-4 py-2 border border-pink-200 rounded-lg outline-none focus:border-pink-500 focus:ring-2 focus:ring-pink-200"
                  disabled={isAttacking}
                  min="1"
                  max="300"
                />
              </div>
              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700">
                  Packet Delay (ms)
                </label>
                <input
                  type="number"
                  value={packetDelay}
                  onChange={(e) => setPacketDelay(Number(e.target.value))}
                  className="w-full px-4 py-2 border border-pink-200 rounded-lg outline-none focus:border-pink-500 focus:ring-2 focus:ring-pink-200"
                  disabled={isAttacking}
                  min="1"
                  max="1000"
                />
              </div>
            </div>
          </div>

          {/* Stats Widgets */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="p-4 rounded-lg bg-gradient-to-br from-pink-500/10 to-blue-500/10">
              <div className="flex items-center gap-2 mb-2 text-pink-600">
                <Zap className="w-4 h-4" />
                <span className="font-semibold">Packets/sec</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {stats.pps.toLocaleString()}
              </div>
            </div>
            <div className="p-4 rounded-lg bg-gradient-to-br from-pink-500/10 to-blue-500/10">
              <div className="flex items-center gap-2 mb-2 text-pink-600">
                <Bot className="w-4 h-4" />
                <span className="font-semibold">Active Bots</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {stats.bots.toLocaleString()}
              </div>
            </div>
            <div className="p-4 rounded-lg bg-gradient-to-br from-pink-500/10 to-blue-500/10">
              <div className="flex items-center gap-2 mb-2 text-pink-600">
                <Wifi className="w-4 h-4" />
                <span className="font-semibold">Total Packets</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {stats.totalPackets.toLocaleString()}
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="h-4 mb-6 overflow-hidden bg-gray-200 rounded-full">
            <div
              className="h-full transition-all duration-500 bg-gradient-to-r from-pink-500 to-blue-500"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Logs Section */}
          <div className="p-4 font-mono text-sm bg-gray-900 rounded-lg">
            <div className="text-green-400">
              {logs.map((log, index) => (
                <div key={index} className="py-1">
                  {`> ${log}`}
                </div>
              ))}
              {logs.length === 0 && (
                <div className="italic text-gray-500">
                  {">"} Waiting for Miku's power...
                </div>
              )}
            </div>
          </div>

          {/* Cute Animation Overlay */}
          {isAttacking && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 bg-gradient-to-r from-pink-500/10 to-blue-500/10 animate-pulse" />
              <div className="absolute top-0 -translate-x-1/2 left-1/2">
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" />
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col items-center">
          <span className="text-sm text-center text-gray-500"> 
          🎵 v1.0 made by{" "}
          <a
            href="https://github.com/sammwyy/mikumikubeam"
            target="_blank"
            rel="noreferrer"
          >
            @Sammwy
          </a>{" "}
          🎵
          </span>
          <span>
          <input type="range" min="0" max="100" step="5" draggable="false" value={audioVol} onChange={(e) => setAudioVol(parseInt(e.target?.value))} />
          </span>
        </div>
      </div>
    </div>
  );
}

export default App;
