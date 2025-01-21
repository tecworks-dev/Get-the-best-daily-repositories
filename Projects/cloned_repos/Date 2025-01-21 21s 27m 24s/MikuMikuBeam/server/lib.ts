export type ProxyProtocol = "http" | "https" | "socks4" | "socks5" | string;

export interface Proxy {
  protocol: ProxyProtocol;
  host: string;
  port: number;
}

export type AttackMethod =
  | "http_flood"
  | "http_slowloris"
  | "tcp_flood"
  | "minecraft_ping";
