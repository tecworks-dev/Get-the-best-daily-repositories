import axios from 'axios';
import axiosRetry from 'axios-retry';

// Create axios client with custom configuration
const client = axios.create({
  timeout: 5000,
  headers: {
    'User-Agent': 'Mozilla/5.0'
  }
});

// Configure retry mechanism
axiosRetry(client, { 
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) => {
    // Custom retry conditions
    return axiosRetry.isNetworkOrIdempotentRequestError(error) || error.code === 'ECONNRESET';
  }
});

// Token information class to parse and store token data
export class TokenInfo {
  constructor(data) {
    const pair = data[0];
    const baseToken = pair.baseToken;
    
    this.name = baseToken.name;
    this.symbol = baseToken.symbol;
    this.address = baseToken.address;
    this.chain = pair.chainId;
    this.liquidity = pair.liquidity?.usd;
    this.marketCap = pair.marketCap;
    this.priceUSD = pair.priceUsd;
    this.createdAt = Math.floor(pair.pairCreatedAt / 1000);  // Convert to seconds timestamp
    
    // Volume data
    const volume = pair.volume || {};
    this.volumeH24 = volume.h24;
    this.volumeH6 = volume.h6;
    this.volumeH1 = volume.h1;
    this.volumeM5 = volume.m5;
    
    // Price changes
    this.changeH6 = pair.priceChange?.h6;
    
    // Website and social media info
    if (pair.info) {
      this.website = pair.info.websites?.[0]?.url;
      
      const twitter = pair.info.socials?.find(s => s.type === 'twitter');
      this.twitter = twitter?.url;
    }
  }
}

// DexScreener API wrapper class
export class DexScreener {
  // Fetches token information from DexScreener API
  static async getTokenInfo(chainId, tokenAddress) {
    const response = await client.get(
      `https://api.dexscreener.com/tokens/v1/${chainId}/${tokenAddress}`
    ).catch(error => {
      console.error('DexScreener API Error:', error.message);
      throw error;
    });
    
    if (!response.data || response.data.length === 0) {
      throw new Error('No data returned from DexScreener');
    }
    
    return new TokenInfo(response.data);
  }
}

