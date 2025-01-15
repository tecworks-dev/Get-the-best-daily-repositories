import type { Express, Request, Response } from 'express';
import { BaitRoute } from '../core/baitroute';
import type { Alert, EndpointConfig } from '../core/types';

export class ExpressBaitRoute extends BaitRoute {
  async registerWithExpress(app: Express): Promise<void> {
    // Initialize bait endpoints
    await this.initialize();

    const endpoints = this.getEndpoints();
    console.log(`Successfully loaded ${endpoints.length} bait endpoints`);

    // Register bait endpoints
    for (const endpoint of endpoints) {
      const handler = this.createExpressHandler(endpoint);

      switch (endpoint.method.toUpperCase()) {
        case 'GET':
          app.get(endpoint.path, handler);
          break;
        case 'POST':
          app.post(endpoint.path, handler);
          break;
        case 'PUT':
          app.put(endpoint.path, handler);
          break;
        case 'DELETE':
          app.delete(endpoint.path, handler);
          break;
        case 'PATCH':
          app.patch(endpoint.path, handler);
          break;
        case 'HEAD':
          app.head(endpoint.path, handler);
          break;
        case 'OPTIONS':
          app.options(endpoint.path, handler);
          break;
      }
    }
  }

  private createExpressHandler(endpoint: EndpointConfig) {
    return async (req: Request, res: Response) => {
      // Set custom headers
      if (endpoint.headers) {
        Object.entries(endpoint.headers).forEach(([key, value]) => {
          res.setHeader(key, value);
        });
      }

      // Set content type
      if (endpoint['content-type']) {
        res.setHeader('Content-Type', endpoint['content-type']);
      }

      // Create and send alert
      const alert: Alert = {
        path: req.path,
        method: req.method,
        remoteAddr: req.ip || req.socket.remoteAddress || 'unknown',
        headers: req.headers as Record<string, string>,
        body: typeof req.body === 'string' ? req.body : JSON.stringify(req.body)
      };

      await this.handleAlert(alert);

      // Send response
      res.status(endpoint.status).send(endpoint.body);
    };
  }
} 