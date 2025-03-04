import axios from 'axios';

// Cache class for SOL price to minimize API calls
class SolPriceCache {
  constructor() {
    this.price = null;
    this.lastUpdate = 0;
    this.CACHE_DURATION = 10 * 60 * 1000; // 10 minutes in milliseconds
  }

  // Fetches SOL price with caching mechanism
  async getPrice() {
    const now = Date.now();
    
    // Return cached price if it exists and hasn't expired
    if (this.price && (now - this.lastUpdate) < this.CACHE_DURATION) {
      // console.log('Returning cached SOL price:', this.price);
      return this.price;
    }

    try {
      // Get latest price from DexScreener
      const response = await axios.get('https://api.dexscreener.com/tokens/v1/solana/So11111111111111111111111111111111111111112', {
        headers: {}
      })
      const data = await response.data;
      
      // Extract SOL price from response data
      const solPrice = parseFloat(data[0].priceUsd);
      
      // Update cache
      this.price = solPrice;
      this.lastUpdate = now;
      
      // console.log('Fetched new SOL price:', this.price);
      return this.price;
    } catch (error) {
      console.error('Failed to fetch SOL price:', error);
      // Return cached price if available when API call fails
      if (this.price) {
        console.log('API call failed, returning cached price:', this.price);
        return this.price;
      }
      throw error;
    }
  }
}

// Create singleton instance
export const solPrice = new SolPriceCache();