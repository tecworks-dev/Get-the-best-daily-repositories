// Directory: /config/api.ts

/**
 * API configuration for the application
 */
export const API_CONFIG = {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080',
    wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080',
    endpoints: {
      health: '/',
      hello: '/api/v1/hello',
      ws: '/ws',
      processVoice: '/process-voice',
      setTimer: '/set-timer',
    },
  } as const;
  
/**
 * Get the full WebSocket URL
 */
export const getWsUrl = () => `${API_CONFIG.wsUrl}${API_CONFIG.endpoints.ws}`;

/**
 * Get the full API URL for a given endpoint
 */
export const getApiUrl = (endpoint: keyof typeof API_CONFIG.endpoints) => 
  `${API_CONFIG.baseUrl}${API_CONFIG.endpoints[endpoint]}`;
  
