import { createServer } from 'http';
import { join } from 'path';
import { parse as parseUrl } from 'url';
import { BaitRoute } from '../../src/core/baitroute';

class HTTPBaitRoute extends BaitRoute {
  async handleRequest(req: any, res: any) {
    const url = parseUrl(req.url || '');
    const path = url.pathname || '/';
    const method = req.method || 'GET';

    // Handle real endpoint
    if (path === '/') {
      res.statusCode = 200;
      res.setHeader('Content-Type', 'text/plain');
      res.end('Welcome to my web application!');
      return;
    }

    // Check if this is a bait endpoint
    const endpoints = this.getEndpoints();
    const baitEndpoint = endpoints.find(
      e => e.path === path && e.method.toUpperCase() === method.toUpperCase()
    );

    if (baitEndpoint) {
      // Set headers
      if (baitEndpoint.headers) {
        Object.entries(baitEndpoint.headers).forEach(([key, value]) => {
          res.setHeader(key, value);
        });
      }

      // Set content type
      if (baitEndpoint['content-type']) {
        res.setHeader('Content-Type', baitEndpoint['content-type']);
      }

      // Create alert
      await this.handleAlert({
        path,
        method,
        remoteAddr: req.socket.remoteAddress || '',
        headers: req.headers as Record<string, string>,
        body: ''
      });

      // Send response
      res.statusCode = baitEndpoint.status;
      res.end(baitEndpoint.body);
      return;
    }

    // Handle 404
    res.statusCode = 404;
    res.end('Not Found');
  }
}

const port = process.env.PORT || 3000;

// Initialize baitroute
const baitroute = new HTTPBaitRoute({
  rulesDir: join(__dirname, '../../../rules'),
  // Optional: specify which rules to load (example)
  
  selectedRules: [
    'exposures/aws-credentials',
    'exposures/circleci-ssh-config',
    'vulnerabilities/sql-injection'
  ],
  
});

// Set up alert handler
baitroute.setAlertHandler(async (alert) => {
  // Basic console logging
  console.log('🚨 Bait endpoint accessed:', {
    path: alert.path,
    method: alert.method,
    remoteAddr: alert.remoteAddr,
    headers: alert.headers,
    body: alert.body
  });

  // Example: Sentry Integration
  /* 
  // import * as Sentry from '@sentry/node';
  // Sentry.init({ dsn: "your-sentry-dsn" });

  Sentry.withScope((scope) => {
    scope.setLevel('warning');
    scope.setExtra('source_ip', alert.remoteAddr);
    scope.setExtra('true_client_ip', alert.trueClientIP);
    scope.setExtra('x_forwarded_for', alert.xForwardedFor);
    scope.setExtra('rule_name', alert.ruleName);
    scope.setExtra('method', alert.method);
    scope.setExtra('path', alert.path);
    scope.setTag('event_type', 'bait_hit');
    Sentry.captureMessage('Security Alert: bait Endpoint Hit');
  });
  */
});

// Create HTTP server
const server = createServer((req, res) => baitroute.handleRequest(req, res));

// Initialize and start server
baitroute.initialize()
  .then(() => {
    server.listen(port, () => {
      console.log(`Server running at http://localhost:${port}`);
    });
  })
  .catch(error => {
    console.error('Failed to initialize baitroute:', error);
    process.exit(1);
  }); 