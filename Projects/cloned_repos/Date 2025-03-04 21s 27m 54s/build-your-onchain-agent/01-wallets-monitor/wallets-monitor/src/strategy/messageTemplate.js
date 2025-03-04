import { formatTimeAgo } from '../utils/txsAnalyzer.js';

// Formats a number to a readable currency string with appropriate suffixes
function formatNumber(number) {
  // Ensure number is a numeric type
  const num = Number(number);
  
  // Check if it's a valid number
  if (isNaN(num)) {
    return '$0.00';
  }
  if (num >= 1_000_000) {
    return `$${(num / 1_000_000).toFixed(1)}M`;
  } else if (num >= 1_000) {
    return `$${Math.round(num / 1_000)}K`;
  }
  return `$${Math.round(num)}`;
}

// Formats smart money wallet data into a readable string
function formatSmartMoney(analysis) {
  let details = '';
  for (const [address, data] of Object.entries(analysis)) {
    details += `\u{25AB}<a href="https://solscan.io/account/${address}">${data.walletName}</a> bought ${formatNumber(data.totalBuyCost)} at MC ${formatNumber(data.averageMarketCap)}(${data.buyTime}), Holds: ${data.holdsPercentage}\n`;
  }
  return details.trim();
}

// Creates a formatted message with token information and smart money analysis
export function createMsg(tokenInfo, analysis) {
  const smartMoneyCount = Object.keys(analysis).length;
  
  return `
\u{1F436} Multi Buy Token: <b>$${tokenInfo.symbol}</b>
<code>${tokenInfo.address}</code>

\u{1F90D} <b>Solana</b>
\u{1F49B} <b>MC:</b> <code>${formatNumber(tokenInfo.marketCap)}</code>
\u{1F90E} <b>Vol/24h:</b> <code>${formatNumber(tokenInfo.volumeH24)}</code>
\u{1F90D} <b>Vol/1h:</b> <code>${formatNumber(tokenInfo.volumeH1)}</code>
\u{1F49B} <b>Liq:</b> <code>${formatNumber(tokenInfo.liquidity)}</code>
\u{1F90E} <b>USD:</b> <code>$${Number(tokenInfo.priceUSD).toFixed(6)}</code>
\u{1F90D} <b>Age:</b> <code>${formatTimeAgo(tokenInfo.createdAt)}</code>
\u{1F49B} <b>6H:</b> <code>${tokenInfo.changeH6}%</code>
\u{1F90E} <b>SmartMoney:</b>
${smartMoneyCount} wallets bought $${tokenInfo.symbol}

${formatSmartMoney(analysis)}

<a href="https://dexscreener.com/solana/${tokenInfo.address}">DexScreener</a> | <a href="https://gmgn.ai/sol/token/${tokenInfo.address}">GMGN</a>${tokenInfo.website ? ` | <a href="${tokenInfo.website}">Website</a>` : ''}${tokenInfo.twitter ? ` | <a href="${tokenInfo.twitter}">Twitter</a>` : ''}
`.trim();
}

