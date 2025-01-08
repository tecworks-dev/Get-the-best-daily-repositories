export interface CaddyHost {
  id: string;
  domain: string;
  fileServer?: {
    root: string;
    browse: boolean;
    php: boolean;
    hide: string[];
  };
  presetName?: string;
  reverseProxy?: string;
  tls?: {
    email?: string;
    selfSigned?: boolean;
  };
  gzip?: boolean;
  basicAuth?: {
    username: string;
    password: string;
  }[];
  headers?: {
    name: string;
    value: string;
  }[];
  cors?: {
    enabled: boolean;
    allowOrigins: string[];
    allowMethods: string[];
    allowHeaders: string[];
  };
  security?: {
    ipFilter: {
      enabled: boolean;
      allow: string[];
      block: string[];
    };
    rateLimit: {
      enabled: boolean;
      requests: number;
      window: string;
    };
    cspEnabled: boolean;
    csp: string;
    forwardAuth: {
      enabled: boolean;
      url: string;
      verifyHeader?: string;
      verifyValue?: string;
    };
  };
  performance?: {
    brotli: boolean;
    cacheControlEnabled: boolean;
    cacheControl: string;
  };
}

export interface PresetConfig {
  name: string;
  port: number;
  description: string;
  category: 'Media & Streaming' | 'Downloaders & File Sharing' | 'Media Management & Automation' | 
    'Home Automation & IoT' | 'Development & Code Hosting' | 'Monitoring & Analytics' | 
    'Productivity & Collaboration' | 'Authentication & Identity' | 'Security & Networking' | 
    'Container & Server Management' | 'Password & Secrets Management' | 'Messaging & Communication';
  webLink?: string;
  githubLink?: string;
  logo?: string;
}
