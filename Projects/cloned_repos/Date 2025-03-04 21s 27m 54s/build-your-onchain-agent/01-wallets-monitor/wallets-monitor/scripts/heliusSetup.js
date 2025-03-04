import dotenv from 'dotenv';
import { Helius } from 'helius-sdk';
import { TransactionType, WebhookType } from 'helius-sdk';
import { createClient } from '@supabase/supabase-js';

dotenv.config();
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const HELIUS_API_KEY = process.env.HELIUS_API_KEY;
const WEBHOOK_URL = process.env.WEBHOOK_URL;

if (!HELIUS_API_KEY) {
  throw new Error('HELIUS_API_KEY is not defined in environment variables.');
}

const helius = new Helius(HELIUS_API_KEY);

// Set up SWAP type Webhook
export const setupSwapWebhook = async () => {
  try {
    const { data, error } = await supabase.from('wallets').select('address');
    if (error) {
      throw new Error('Failed to fetch wallet addresses from Supabase');
    }

    const accountAddresses = data.map(row => row.address).filter(addr => addr);

    if (accountAddresses.length === 0) {
      throw new Error('No valid wallet addresses found in wallets.txt.');
    }

    // Create Webhook configuration
    const webhookConfig = {
      accountAddresses,
      transactionTypes: [TransactionType.SWAP], 
      webhookURL: WEBHOOK_URL,
      authHeader: `Bearer ${HELIUS_API_KEY}`,
      webhookType: WebhookType.ENHANCED, 
    };

    const response = await helius.createWebhook(webhookConfig);
    console.log('Webhook created successfully:', response);
  } catch (error) {
    console.error('Error creating webhook:', error);
  }
};
