import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

// Parses Solana transaction data using the Shyft API
export async function solParser(signature) {
  const BASE_URL = "https://api.shyft.to/sol/v1";
  
  const response = await axios.get(`${BASE_URL}/transaction/parsed`, {
    params: {
      network: 'mainnet-beta',
      txn_signature: signature
    },
    headers: {
      'x-api-key': process.env.SHYFT_API_KEY
    }
  }).catch(error => {
    console.error('Error fetching transaction:', error);
    return { data: null };
  });
  
  if (!response || !response.data) {
    return null;
  }
  
  // Check if successful and is a SWAP type transaction
  if (response.data.success && response.data.result) {
    const result = response.data.result;   
    console.log(JSON.stringify(result, null, 2));
    // Find action containing tokens_swapped
    const swapAction = result.actions.find(action => 
      action.info && action.info.tokens_swapped
    );
    
    if (swapAction) {
      // Convert ISO timestamp to seconds timestamp
      const timestamp = Math.floor(new Date(result.timestamp).getTime() / 1000);
      
      return {
        account: swapAction.info.swapper,
        token_in_address: swapAction.info.tokens_swapped.in.token_address,
        token_in_amount: swapAction.info.tokens_swapped.in.amount,
        token_out_address: swapAction.info.tokens_swapped.out.token_address,
        token_out_amount: swapAction.info.tokens_swapped.out.amount,
        timestamp: timestamp,
        description: null
      };
    }
  }
  
  return response.data;
}
