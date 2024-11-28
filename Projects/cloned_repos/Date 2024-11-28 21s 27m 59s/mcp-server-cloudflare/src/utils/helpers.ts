// Debug logging
const debug = true
export function log(...args: any[]) {
  const msg = `[DEBUG ${new Date().toISOString()}] ${args.join(' ')}\n`
  process.stderr.write(msg)
}

// Config
export const config = {
  accountId: process.env.CLOUDFLARE_ACCOUNT_ID,
  apiToken: process.env.CLOUDFLARE_API_TOKEN,
}

export { version as mcpCloudflareVersion } from '../../package.json'
