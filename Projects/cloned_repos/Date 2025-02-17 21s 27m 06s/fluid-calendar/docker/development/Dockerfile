FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
COPY prisma ./prisma/

# Install dependencies and development tools
RUN npm install
RUN npm install -g prisma

# Generate Prisma Client
RUN mkdir -p /app/prisma
RUN chown -R node:node /app

# Switch to non-root user
USER node

# Copy the rest of the code
COPY --chown=node:node . .

# Expose port 3000
EXPOSE 3000

# Start development server with database setup
CMD ["npm", "run", "dev"]
