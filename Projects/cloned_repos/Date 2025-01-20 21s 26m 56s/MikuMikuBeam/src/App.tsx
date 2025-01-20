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
  const [attackMethod, setAttackMethod] = useState("http");
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

  const addLog = (message: string) => {
    setLogs((prev) => [message, ...prev].slice(0, 12));
  };

  const startAttack = () => {
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
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    }

    // Start attack after audio intro
    const timeout = setTimeout(() => {
      setActuallyAttacking(true);
      socket.emit("startAttack", {
        target,
        packetSize,
        duration,
        packetDelay,
        attackMethod,
      });
    }, 10250);
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
          <h1 className="text-4xl font-bold text-pink-500 mb-2">
            Miku Miku Beam
          </h1>
          <p className="text-gray-600">
            Because DDoS attacks are also cute and even more so when Miku does
            them.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-xl p-6 relative overflow-hidden">
          {/* Miku GIF */}
          <div
            className="flex justify-center mb-6 h-48 w-full"
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
                className="px-4 py-2 rounded-lg border border-pink-200 focus:border-pink-500 focus:ring-2 focus:ring-pink-200 outline-none"
                disabled={isAttacking}
              />
              <button
                onClick={isAttacking ? stopAttack : startAttack}
                className={`
                  px-8 py-2 rounded-lg font-semibold text-white transition-all
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
            </div>

            <div className="grid grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Attack Method
                </label>
                <select
                  value={attackMethod}
                  onChange={(e) => setAttackMethod(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-pink-200 focus:border-pink-500 focus:ring-2 focus:ring-pink-200 outline-none"
                  disabled={isAttacking}
                >
                  <option value="http">HTTP</option>
                  <option value="tcp">TCP</option>
                  <option value="udp">UDP</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Packet Size (kb)
                </label>
                <input
                  type="number"
                  value={packetSize}
                  onChange={(e) => setPacketSize(Number(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg border border-pink-200 focus:border-pink-500 focus:ring-2 focus:ring-pink-200 outline-none"
                  disabled={isAttacking}
                  min="1"
                  max="1500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Duration (seconds)
                </label>
                <input
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg border border-pink-200 focus:border-pink-500 focus:ring-2 focus:ring-pink-200 outline-none"
                  disabled={isAttacking}
                  min="1"
                  max="300"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Packet Delay (ms)
                </label>
                <input
                  type="number"
                  value={packetDelay}
                  onChange={(e) => setPacketDelay(Number(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg border border-pink-200 focus:border-pink-500 focus:ring-2 focus:ring-pink-200 outline-none"
                  disabled={isAttacking}
                  min="1"
                  max="1000"
                />
              </div>
            </div>
          </div>

          {/* Stats Widgets */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gradient-to-br from-pink-500/10 to-blue-500/10 p-4 rounded-lg">
              <div className="flex items-center gap-2 text-pink-600 mb-2">
                <Zap className="w-4 h-4" />
                <span className="font-semibold">Packets/sec</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {stats.pps.toLocaleString()}
              </div>
            </div>
            <div className="bg-gradient-to-br from-pink-500/10 to-blue-500/10 p-4 rounded-lg">
              <div className="flex items-center gap-2 text-pink-600 mb-2">
                <Bot className="w-4 h-4" />
                <span className="font-semibold">Active Bots</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {stats.bots.toLocaleString()}
              </div>
            </div>
            <div className="bg-gradient-to-br from-pink-500/10 to-blue-500/10 p-4 rounded-lg">
              <div className="flex items-center gap-2 text-pink-600 mb-2">
                <Wifi className="w-4 h-4" />
                <span className="font-semibold">Total Packets</span>
              </div>
              <div className="text-2xl font-bold text-gray-800">
                {stats.totalPackets.toLocaleString()}
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-gray-200 rounded-full h-4 mb-6 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-pink-500 to-blue-500 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Logs Section */}
          <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm">
            <div className="text-green-400">
              {logs.map((log, index) => (
                <div key={index} className="py-1">
                  {`> ${log}`}
                </div>
              ))}
              {logs.length === 0 && (
                <div className="text-gray-500 italic">
                  {">"} Waiting for Miku's power...
                </div>
              )}
            </div>
          </div>

          {/* Cute Animation Overlay */}
          {isAttacking && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 bg-gradient-to-r from-pink-500/10 to-blue-500/10 animate-pulse" />
              <div className="absolute top-0 left-1/2 -translate-x-1/2">
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" />
              </div>
            </div>
          )}
        </div>

        <div className="text-center text-sm text-gray-500">
          🎵 v1.0 made by{" "}
          <a
            href="https://github.com/sammwyy/mikumikubeam"
            target="_blank"
            rel="noreferrer"
          >
            @Sammwy
          </a>{" "}
          🎵
        </div>
      </div>
    </div>
  );
}

export default App;
