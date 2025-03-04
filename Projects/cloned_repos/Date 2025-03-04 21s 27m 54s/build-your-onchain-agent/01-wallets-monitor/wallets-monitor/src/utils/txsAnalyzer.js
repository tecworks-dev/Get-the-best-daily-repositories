import BigNumber from 'bignumber.js';
import { solPrice } from './solPrice.js';
import { createClient } from '@supabase/supabase-js';
import 'dotenv/config';
import { USDC_ADDRESS, SOL_ADDRESS } from './swapProcessor.js';
import { DexScreener } from './dexscreener.js';
import { Connection, PublicKey } from '@solana/web3.js';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

// Retrieves transaction records for a specific token
async function getTokenTxs(tokenAddress) {
  try {
    // Query transactions for the token
    const { data: txs } = await supabase
      .from('txs')
      .select('account, token_in_address, token_in_amount, token_out_address, token_out_amount, timestamp')
      .or(`token_in_address.eq.${tokenAddress},token_out_address.eq.${tokenAddress}`)
      .order('timestamp', { ascending: true });

    if (!txs || txs.length === 0) {
      return {};
    }

    // Group transactions by account
    const accountTxs = {};
    txs.forEach(tx => {
      if (!accountTxs[tx.account]) {
        accountTxs[tx.account] = [];
      }
      accountTxs[tx.account].push(tx);
    });

    return accountTxs;
  } catch (error) {
    console.error(`Error fetching txs for token ${tokenAddress}:`, error);
    throw error;
  }
}

// Gets the current price of a token
async function getTokenPrice(tokenAddress) {
  try {
    if (tokenAddress === SOL_ADDRESS) {
      return new BigNumber(await solPrice.getPrice());
    } else if (tokenAddress === USDC_ADDRESS) {
      return new BigNumber(1);
    } else {
      const tokenInfo = await DexScreener.getTokenInfo('solana', tokenAddress);
      return new BigNumber(tokenInfo.priceUSD || 0);
    }
  } catch (error) {
    console.error(`Error getting price for token ${tokenAddress}:`, error);
    return new BigNumber(0);
  }
}

// Retrieves the total supply of a token
async function getTokenSupply(tokenAddress) {
  try {
    // Connect to Solana network
    const connection = new Connection(process.env.RPC_ENDPOINT, 'confirmed');
    
    // Create token PublicKey
    const mintPubkey = new PublicKey(tokenAddress);
    
    // Get token information
    const tokenInfo = await connection.getTokenSupply(mintPubkey);
    const totalSupply = tokenInfo.value.uiAmountString;
    
    // Return default value if not available
    return totalSupply || '1000000000';
  } catch (error) {
    console.error(`Error getting supply for token ${tokenAddress}:`, error);
    // Return default value on error
    return '1000000000';
  }
}

// Formats a timestamp into a human-readable time ago string
export function formatTimeAgo(timestamp) {
  const now = Math.floor(Date.now() / 1000);
  const diff = now - timestamp;
  
  // Time units in seconds
  const minute = 60;
  const hour = minute * 60;
  const day = hour * 24;
  
  if (diff < minute) {
    return `${diff}s ago`;
  } else if (diff < hour) {
    const minutes = Math.floor(diff / minute);
    return `${minutes}m ago`;
  } else if (diff < day) {
    const hours = Math.floor(diff / hour);
    return `${hours}h ago`;
  } else {
    const days = Math.floor(diff / day);
    return `${days}d ago`;
  }
}

// Analyzes transaction data for a token and calculates key metrics
async function analyzeTxs(accountTxs, tokenAddress) {
  const totalSupply = new BigNumber(await getTokenSupply(tokenAddress));
  const result = {};
  
  for (const [account, txs] of Object.entries(accountTxs)) {
    const buyTxs = txs.filter(tx => tx.token_out_address === tokenAddress);
    const sellTxs = txs.filter(tx => tx.token_in_address === tokenAddress);
    
    if (buyTxs.length === 0) continue;

    let totalBuyCost = new BigNumber(0);
    let totalBuyAmount = new BigNumber(0);
    let latestBuyTime = 0;

    // Calculate buy-related data
    for (const tx of buyTxs) {
      const tokenInPrice = await getTokenPrice(tx.token_in_address);
      const tokenInAmount = new BigNumber(tx.token_in_amount);
      const txCost = tokenInPrice.multipliedBy(tokenInAmount);
      
      totalBuyCost = totalBuyCost.plus(txCost);
      totalBuyAmount = totalBuyAmount.plus(new BigNumber(tx.token_out_amount));
      latestBuyTime = Math.max(latestBuyTime, tx.timestamp);
    }

    // Calculate total sell amount
    const totalSellAmount = sellTxs.reduce(
      (sum, tx) => sum.plus(new BigNumber(tx.token_in_amount)), 
      new BigNumber(0)
    );
    
    // Calculate holding percentage
    const remainingAmount = BigNumber.maximum(0, totalBuyAmount.minus(totalSellAmount));
    const holdsPercentage = remainingAmount.dividedBy(totalBuyAmount).multipliedBy(100);
    
    // Calculate average buy price
    const averageBuyPrice = totalBuyAmount.isZero() ? 
      new BigNumber(0) : 
      totalBuyCost.dividedBy(totalBuyAmount);
    // Calculate average market cap at buy time
    const averageMarketCap = averageBuyPrice.multipliedBy(totalSupply);

    result[account] = {
      totalBuyCost: totalBuyCost.toFixed(0),
      averageBuyPrice: averageBuyPrice.toFixed(6),
      averageMarketCap: averageMarketCap.toFixed(0),
      buyTime: formatTimeAgo(latestBuyTime), 
      holdsPercentage: holdsPercentage.toFixed(2) + '%'
    };
  }

  // Get all wallet addresses
  const walletAddresses = Object.keys(result);
  
  // Query wallet names from supabase
  const { data: wallets } = await supabase
    .from('wallets')
    .select('address, name')
    .in('address', walletAddresses);
    
  // Create address to name mapping
  const addressToName = {};
  if (wallets) {
    wallets.forEach(wallet => {
      addressToName[wallet.address] = wallet.name;
    });
  }
  
  // Add wallet names to analysis results
  const resultWithNames = Object.entries(result).reduce((acc, [address, data]) => {
    acc[address] = {
      ...data,
      walletName: addressToName[address] || 'Unknown'
    };
    return acc;
  }, {});

  return resultWithNames;
}

// Main function to analyze transactions for a token
export async function analyzeTokenTxs(tokenAddress) {
  try {
    const transactionData = await getTokenTxs(tokenAddress);
    const analysis = await analyzeTxs(transactionData, tokenAddress);
    return analysis;
  } catch (error) {
    console.error(`Error analyzing transactions for token ${tokenAddress}:`, error);
    throw error;
  }
}


