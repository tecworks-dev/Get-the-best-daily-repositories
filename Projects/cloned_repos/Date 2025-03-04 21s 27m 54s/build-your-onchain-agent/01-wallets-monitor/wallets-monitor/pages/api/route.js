import { createClient } from '@supabase/supabase-js';
import { processSwapData } from '../../src/utils/swapProcessor';
import { solParser } from '../../src/utils/txParser';

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

// Handle Webhook request
export default async function handler(req, res) {
  // Check request method and authorization
  if (req.method !== 'POST') return res.status(405).json({ message: 'Method not allowed' });
  if (req.headers.authorization !== `Bearer ${process.env.HELIUS_API_KEY}`) {
    return res.status(401).json({ message: 'Unauthorized' });
  }

  // Get transaction data
  const txData = Array.isArray(req.body) ? req.body[0] : req.body;
  if (!txData) {
    console.error('Empty transaction data received', txData);
    return res.status(200).json({ skipped: true, message: 'Empty data' });
  }

  // Process transaction data
  let processedData = null;
  
  if (txData.events?.swap) {
    processedData = processSwapData(txData);
  } else if (txData.signature) {
    processedData = await solParser(txData.signature);
    if (!processedData) {
      console.error('Failed to parse tx:', txData.signature);
      return res.status(200).json({ skipped: true, message: 'Parse failed', signature: txData.signature });
    }
  } else {
    return res.status(200).json({ skipped: true, message: 'No swap data' });
  }

  // Store to database
  const { error } = await supabase.from('txs').insert([{
    ...processedData,
    signature: txData.signature
  }]);
  if (error) {
    console.error('Error inserting into Supabase:', error);
    return res.status(500).json({ error: error });
  }
  console.log('Successfully processed and stored with parser:', txData.events?.swap ? 'helius' : 'shyft');
  return res.status(200).json({ 
    success: true
  });
}