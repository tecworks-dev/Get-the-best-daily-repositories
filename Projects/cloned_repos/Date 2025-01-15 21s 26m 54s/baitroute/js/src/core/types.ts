export interface EndpointConfig {
  path: string;
  method: string;
  status: number;
  'content-type'?: string;
  headers?: Record<string, string>;
  body?: string;
}

export interface Alert {
  path: string;
  method: string;
  remoteAddr: string;
  headers: Record<string, string>;
  body?: string;
}

export type AlertHandler = (alert: Alert) => void | Promise<void>;

export interface BaitRouteOptions {
  rulesDir: string;
  selectedRules?: string[];
}