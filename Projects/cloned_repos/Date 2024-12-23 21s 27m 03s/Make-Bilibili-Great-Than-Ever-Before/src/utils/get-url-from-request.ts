import { logger } from '../logger';

export function getUrlFromRequest(request: RequestInfo | URL): string | null {
  if (typeof request === 'string') {
    return request;
  }
  if (request instanceof URL) {
    return request.href;
  }
  if (request instanceof Request) {
    return request.url;
  }

  const _: never = request;
  logger.error('Invalid requestInfo', request);
  return null;
}
