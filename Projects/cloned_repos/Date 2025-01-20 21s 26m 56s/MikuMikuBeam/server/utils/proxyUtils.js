const METHODS = {
  http: ["http", "https"],
  tcp: ["socks4", "socks5"],
  udp: ["socks4", "socks5"],
};

export function filterProxies(proxies, method) {
  return proxies.filter((proxy) => METHODS[method].includes(proxy.protocol));
}
