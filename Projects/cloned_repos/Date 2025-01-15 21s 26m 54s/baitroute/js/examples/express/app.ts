import express from 'express';
import { join } from 'path';
import { ExpressBaitRoute } from '../../src/integrations/express';

const app = express();
const port = process.env.PORT || 3000;

// Create a real endpoint
app.get('/', (req, res) => {
  res.send('Welcome to my web application!');
});

// Initialize baitroute
const baitroute = new ExpressBaitRoute({
  rulesDir: join(__dirname, '../../../rules'),
  // Optional: specify which rules to load (example)
  /* 
  selectedRules: [
    'exposures/aws-credentials',
    'exposures/circleci-ssh-config',
    'vulnerabilities/sql-injection'
  ],
  */
});

// Set up alert handler
baitroute.setAlertHandler(async (alert) => {
  // Basic console logging
  console.log('🚨 Bait endpoint accessed:', {
    path: alert.path,
    method: alert.method,
    sourceIP: alert.remoteAddr,
    headers: alert.headers,
    body: alert.body
  });

  // Example: Sentry Integration
  /* 
  // import * as Sentry from '@sentry/node';
  // Sentry.init({ dsn: "your-sentry-dsn" });

  Sentry.withScope((scope) => {
    scope.setLevel('warning');
    scope.setExtra('source_ip', alert.sourceIP);
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

// Register baitroute endpoints
baitroute.registerWithExpress(app)
  .then(() => {
    app.listen(port, () => {
      console.log(`Example app listening at http://localhost:${port}`);
    });
  })
  .catch((error) => {
    console.error('Failed to initialize baitroute:', error);
    process.exit(1);
  }); 