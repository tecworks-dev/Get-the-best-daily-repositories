import { DexScreener } from '../utils/dexscreener.js';
import { createClient } from '@supabase/supabase-js';
import { SOL_ADDRESS, USDC_ADDRESS } from '../utils/swapProcessor.js';
import { sendTelegramMessage } from '../utils/telegram.js';
import { analyzeTokenTxs } from '../utils/txsAnalyzer.js';
import { createMsg } from './messageTemplate.js';
import { sendSumMessage } from '../utils/aiSummary.js';
import dotenv from 'dotenv';

dotenv.config();
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

// Configuration constants
const MAX_AGE_DAYS = 7;
const MIN_MARKET_CAP = 100000; // 100k

const getTimeStamp = () => {
  return new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' });
};

// Check if token meets filtering criteria
async function checkFilter(tokenAddress) {
  try {
    const tokenInfo = await DexScreener.getTokenInfo('solana', tokenAddress);   
    if (!tokenInfo) return;
    
    const pairAge = (Date.now() / 1000 - tokenInfo.createdAt) / (60 * 60 * 24);
    if (pairAge <= MAX_AGE_DAYS && tokenInfo.marketCap >= MIN_MARKET_CAP) {
      const analysis = await analyzeTokenTxs(tokenAddress);
      
      // Create and send message to Telegram
      const message = createMsg(tokenInfo, analysis);
      const tgResponse = await sendTelegramMessage(message);
      
      if (tgResponse?.ok === true) {
        const messageId = tgResponse.result.message_id;
        // Send AI summary message
        await sendSumMessage(tokenInfo, messageId);
        console.log(`[${getTimeStamp()}] Successfully sent analysis for token ${tokenAddress} to Telegram`);
      } 
    } 
  } catch (error) {
    console.error(`[${getTimeStamp()}] Error checking token ${tokenAddress}:`, error);
  }
}

// Subscribe to INSERT events on the txs table
async function startMonitor() {
  supabase
    .channel('txs_monitor')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'txs',
      },
      async (payload) => {
        const newTx = payload.new;
        const tokenOutAddress = newTx.token_out_address;
        const currentAccount = newTx.account;
        const currentTimestamp = newTx.timestamp;
        
        // Check if it's not SOL or USDC (buy transaction)
        if (tokenOutAddress !== SOL_ADDRESS && tokenOutAddress !== USDC_ADDRESS) {
          const sixHoursAgo = new Date(currentTimestamp);
          sixHoursAgo.setHours(sixHoursAgo.getHours() - 6);
          const sixHoursAgoTimestamp = Math.floor(sixHoursAgo.getTime() / 1000); 
          
          // Query if other wallets bought this token in the last 6 hours
          const { data, error } = await supabase
            .from('txs')
            .select('*')
            .eq('token_out_address', tokenOutAddress)
            .neq('account', currentAccount)
            .gte('timestamp', sixHoursAgoTimestamp)
            .limit(1);
            
          if (error) {
            console.error(`[${getTimeStamp()}] Query error:`, error);
            return;
          }
          
          // If transactions from other wallets found
          if (data && data.length > 0) {
            console.log(`[${getTimeStamp()}] Detected new multi-wallet transaction for token: ${tokenOutAddress}`);
            // Call filter check function
            await checkFilter(tokenOutAddress);
          }
        }
      }
    )
    .on('error', (error) => {
      console.error(`[${getTimeStamp()}] Supabase realtime connection error:`, error);
    })
    .subscribe((status) => {
      console.log(`[${getTimeStamp()}] Monitoring started... Subscription status:`, status);
    })
}

// Start monitoring
startMonitor().catch(error => {
  console.error(`[${getTimeStamp()}] Monitor program error:`, error);
  process.exit(1);
});


