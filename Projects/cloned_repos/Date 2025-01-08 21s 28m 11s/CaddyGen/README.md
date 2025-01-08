
# **CaddyGen - Caddy Config Generator**

CaddyGen is a user-friendly web interface for generating [Caddy server](https://caddyserver.com) configurations. Easily create and manage reverse proxy and file server configurations with support for SSL, compression, security headers, and more.

## **Features**

- 🚀 **Visual configuration builder** for the Caddy server
- 🔒 **SSL/TLS configuration** with Let's Encrypt support
- 🔄 **Reverse proxy** and **file server** setup
- 📦 Pre-configured **presets** for popular applications
- 💾 **Local storage** for saving configurations
- 📤 **Import/Export** for Caddyfiles
- 🛡️ Advanced **security options** (CSP, rate limiting, IP filtering)
- ⚡ **Performance optimizations** (compression, caching)
- 🌐 **CORS configuration**
- 📁 File server options (directory listing, PHP support)

---

## **Live Demo**

Try CaddyGen now: [caddygen.site](https://caddygen.site)

### **How to Use**
1. Click the **"Add New Host"** button.
2. Configure your settings:
   - Choose between **reverse proxy** or **file server**.
   - Select from pre-configured **application presets**.
   - Configure SSL, compression, and security options.
3. View, copy, or download your generated **Caddyfile**.

---

## **Docker Deployment**

Deploy CaddyGen using Docker with ease:

### **Run the Pre-Built Image**
Pull the latest image from Docker Hub and run it:
```bash
docker pull wardy784/caddygen:latest
docker run -d --restart unless-stopped -p 8189:80 wardy784/caddygen:latest
```

### **Run with Docker Compose**
Use the following `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    image: wardy784/caddygen:latest
    ports:
      - "8189:80"
    restart: unless-stopped
    container_name: caddygen
```

To deploy:
```bash
docker compose up -d
```

Access the app at `http://localhost:8189`.

### **Build Locally**
If you prefer to build the image yourself:
```bash
docker build -t caddygen .
docker run -p 8189:80 caddygen
```

---

## **Development Setup**

This project is built using the following technologies:

- **Vue 3** with TypeScript
- **Vite** for fast builds
- **Tailwind CSS** for styling
- **Prism.js** for syntax highlighting
- **Lucide Icons** for UI elements

### **Getting Started**
1. Clone the repository:
   ```bash
   git clone https://github.com/wardy784/caddygen.git
   cd caddygen
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open the app in your browser at `http://localhost:3000`.

---

## **Contributing**

Contributions are welcome! Whether it's fixing a bug, suggesting a new feature, or improving the documentation, we’d love your help. Feel free to open an issue or submit a pull request.

---

## **License**

CaddyGen is open-source and available under the **MIT License**. Feel free to use, modify, and distribute it as you wish!
