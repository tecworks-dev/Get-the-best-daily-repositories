Join our Discord community:  
<a href="https://discord.gg/7vCkqfyn"><img src="https://img.icons8.com/color/24/000000/discord-logo.png" alt="Discord Logo" /></a> 

# Project Wildberry

<img src="https://i.imgur.com/Hdt9TXr.png">

**Wildberry** is an open-source, privacy-focused alternative to revenuecat with zero vendor lock-in for in-app purchases and subscriptions management. 

## Features

- **No Vendor Lock-in**: Take control of your data and your business.
- **Privacy First**: We respect user privacy with no data sharing or tracking.
- **Flexible Integration**: Easy to integrate with various platforms and payment gateways.
- **Open Source**: Transparency and community contributions are welcome.
- **Modern Stack**: Built with Next.js frontend and PostgreSQL backend for optimal performance.

## Why Wildberry?

- **Freedom**: Escape from the constraints of proprietary systems.
- **Cost-Effective**: Reduce your expenses by leveraging community-driven development.
- **Customizability**: Tailor the system to your exact needs without waiting on a third-party.
- **Scalable**: Built on reliable technologies that can handle growth.

## Support Wildberry

**Support Wildberry's Growth:**  
<a href="https://buymeacoffee.com/rcopensource"><img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=rcopensource&button_colour=FF5F5F&font_colour=ffffff&font_family=Cookie&outline_colour=000000&coffee_colour=FFDD00"></a>

Your support is crucial for maintaining and improving this open-source project. Every contribution helps!

## Tech Stack

- **Frontend**: Next.js
- **Backend**: Node.js with Express
- **Database**: PostgreSQL
- **Deployment**: Docker & Coolify

## Prerequisites

- Node.js (version 18.x or higher)
- PostgreSQL (version 14.x or higher)
- Docker
- 2GB RAM, 2vCPU (minimum)

## Local Development

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/wildberry.git
   cd wildberry
   ```

2. **Install Dependencies**:
   ```sh
   npm install
   ```
   
3. **Configure Environment**:
   ```sh
   cp .env.example .env
   ```
   Update the `.env` file with your PostgreSQL credentials and other configurations.

4. **Run Development Server**:
   ```sh
   npm run dev
   ```

## Docker Compose Setup

You can quickly set up the entire application stack using Docker Compose:

1. **Configure Environment Variables**:
   Create a `.env` file with the following variables:
   ```
   POSTGRES_USER=wildberry
   POSTGRES_PASSWORD=your_secure_password
   POSTGRES_DB=wildberry
   ```

2. **Start the Services**:
   ```sh
   docker-compose up -d
   ```
   This will start both the Next.js application and PostgreSQL database.

3. **Check Services**:
   ```sh
   docker-compose ps
   ```
   Verify that both services are running.

4. **Access the Application**:
   - Frontend: http://localhost:3000
   - Database: localhost:5432

5. **View Logs**:
   ```sh
   docker-compose logs -f
   ```

6. **Stop Services**:
   ```sh
   docker-compose down
   ```
   Add `-v` flag to remove volumes: `docker-compose down -v`

## Deployment with Coolify

### Prerequisites
- A server with Coolify installed
- Docker installed on your deployment server
- PostgreSQL database (can be hosted on Coolify)

### Deployment Steps

1. **Database Setup**:
   - In Coolify dashboard, create a new PostgreSQL database
   - Save the connection credentials

2. **Application Deployment**:
   - Connect your Git repository to Coolify
   - Choose "Docker" as deployment method
   - Set the following environment variables:
     ```
     DATABASE_URL=postgresql://user:password@host:5432/dbname
     NODE_ENV=production
     ```
   - Use the provided `Dockerfile` in the root directory
   - Set the build command: `npm run build`
   - Set the start command: `npm start`

3. **Configure Domain and SSL**:
   - Add your domain in Coolify
   - Enable SSL (Coolify handles this automatically)

4. **Deploy**:
   - Click "Deploy" in Coolify dashboard
   - Monitor the build and deployment logs

### Monitoring and Maintenance

- Use Coolify's built-in monitoring tools
- Check logs through Coolify dashboard
- Set up alerts for critical events

## Contributing

We welcome contributions! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

Please read our Contribution Guidelines (CONTRIBUTING.md) before making changes.

## License

Wildberry is licensed under the MIT License (LICENSE).

## Acknowledgements

- Thanks to the Next.js team for the amazing framework
- PostgreSQL community for the robust database
- Coolify team for the deployment platform
- The open-source community for their invaluable contributions

Thank you for using Wildberry! We're excited to see what you'll build with it.
