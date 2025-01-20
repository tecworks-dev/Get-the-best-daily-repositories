# Miku Miku Beam 💥⚡ (Network Stresser)

A fun and visually appealing stress testing server with a **Miku-themed** frontend, where you can configure and run attacks while enjoying a banger song in the background! 🎤✨

![Screenshot](docs/screenshot.png)

## Features 🎉

- 🌐 **Real-time Attack Visualization**: View your attack’s progress and statistics in real-time as it runs. 🔥
- 🎶 **Miku-themed UI**: A cute and vibrant design with Miku’s vibe to make the process more fun. Includes a banger song to keep you pumped! 🎧
- 🧑‍💻 **Configurable Attack Parameters**: Easily set the attack method, packet size, duration, and packet delay via the frontend interface.
- 🛠️ **Worker-Based Attack Handling**: The server processes attacks in separate workers for optimal performance and scalability.
- 📊 **Live Stats**: Track the success and failure of each attack in real-time. See how many packets are sent and whether they succeed or fail.
- 🖼️ **Aesthetic Design**: A visually cute interface to make your experience enjoyable. 🌸
- 📡 **Attack Methods:**:
  - `HTTP` - Send HTTP requests

## Setup 🛠️

### Prerequisites 📦

Make sure you have the following installed:

- Node.js (v14 or above) 🌱
- npm (Node Package Manager) 📦

### Installation 💻

1. Clone this repository:

   ```bash
   git clone https://github.com/sammwyy/mikumikubeam.git
   cd mikumikubeam
   ```

2. Install the required dependencies:

   ```bash
   npm install
   ```

3. Create the necessary files:
   - `proxies.txt` - List of proxies.
   - `uas.txt` - List of user agents.

4. Run the server:

   ```bash
   npm run dev
   ```

   The server will run on port `3000` by default. 🌐

5. Open the frontend (usually accessible at `http://localhost:5173`), where you can configure and visualize your attacks.

## Usage ⚙️

Once the server is up and running, you can interact with it via the frontend:

1. **Start Attack**:
   - Set up the attack parameters: target URL, attack method (HTTP, etc.), packet size, duration, and delay.
   - Press "Start Attack" to initiate the stress test.

2. **Stop Attack**:
   - Press "Stop Attack" to terminate the ongoing attack.

### Example Request

```json
{
  "target": "http://example.com",
  "attackMethod": "http",
  "packetSize": 512,
  "duration": 60,
  "packetDelay": 500
}
```

## Worker-Based Attack Handling 🔧💡

Each attack type is handled in a separate worker thread, ensuring that the main server remains responsive. The attack workers are dynamically loaded based on the selected attack method (HTTP, etc...).

## To-Do 📝

- Add more attack methods:
  - Minecraft 🎮
  - TCP 💻
  - UDP 🌐
  - DNS 📡
  - And more! 🔥

- Enhance attack statistics and reporting for better real-time monitoring. 📊

## Contributing 💖

Feel free to fork the repo and open pull requests with new attack protocols, bug fixes, or improvements. If you have an idea for a new feature, please share it! 😄

### Adding New Attack Methods ⚡

To extend the server with new attack methods (e.g., Minecraft, TCP, UDP, DNS), you can create new worker files and add them to the server configuration.

For example:

- Add a new attack method in the frontend settings.
- Create the corresponding worker file (e.g., `minecraftAttack.js`).
- Update the attack handler configuration to include the new method.

```javascript
const attackHandlers = {
  http: "./workers/httpAttack.js",
  minecraft: "./workers/minecraftAttack.js",
  udp: "./workers/udpAttack.js",
  tcp: "./workers/tcpAttack.js",
  dns: "./workers/dnsAttack.js",
  // Add more protocols as needed!
};
```

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer 🚨

Please note that this project is for educational purposes only and should not be used for malicious purposes.

---

### (｡♥‿♥｡) Happy Hacking 💖🎶
