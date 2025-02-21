export type SubscriptionStatus = 'active' | 'inactive' | 'unknown';

export interface SuperwallConfig {
  apiKey: string;
  debugMode?: boolean;
  options?: {
    logging?: boolean;
    paywallTimeout?: number;
  };
} 