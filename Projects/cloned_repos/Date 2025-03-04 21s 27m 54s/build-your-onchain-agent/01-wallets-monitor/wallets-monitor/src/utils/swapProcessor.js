import BigNumber from 'bignumber.js';

export const SOL_ADDRESS = 'So11111111111111111111111111111111111111112';
export const USDC_ADDRESS = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';

// Formats token amount by dividing by the appropriate decimal power
export function formatAmount(amount, decimals) {
  return new BigNumber(amount)
    .dividedBy(new BigNumber(10).pow(decimals))
    .toFixed();
}

// Processes swap event data from webhook into a standardized format
export function processSwapData(webhookData) {
  const swapEvent = webhookData.events.swap;
  let processedData = {};

  // Process account and token_in information
  if (swapEvent.nativeInput && swapEvent.nativeInput.amount) {
    processedData = {
      account: webhookData.feePayer,
      token_in_address: SOL_ADDRESS,
      token_in_amount: formatAmount(parseInt(swapEvent.nativeInput.amount), 9)
    };
  } else if (swapEvent.tokenInputs && swapEvent.tokenInputs.length > 0) {
    const tokenInput = swapEvent.tokenInputs[0];
    processedData = {
      account: webhookData.feePayer,
      token_in_address: tokenInput.mint,
      token_in_amount: formatAmount(
        parseInt(tokenInput.rawTokenAmount.tokenAmount),
        tokenInput.rawTokenAmount.decimals
      )
    };
  }

  // Process token_out information
  if (swapEvent.nativeOutput && swapEvent.nativeOutput.amount) {
    processedData.token_out_address = SOL_ADDRESS;
    processedData.token_out_amount = formatAmount(parseInt(swapEvent.nativeOutput.amount), 9);
  } else if (swapEvent.tokenOutputs && swapEvent.tokenOutputs.length > 0) {
    const tokenOutput = swapEvent.tokenOutputs[0];
    processedData.token_out_address = tokenOutput.mint;
    processedData.token_out_amount = formatAmount(
      parseInt(tokenOutput.rawTokenAmount.tokenAmount),
      tokenOutput.rawTokenAmount.decimals
    );
  }

  // Add timestamp and description
  processedData.timestamp = webhookData.timestamp;
  processedData.description = webhookData.description;

  return processedData;
}
