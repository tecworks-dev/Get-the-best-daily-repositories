import {
  ChainsTypeKeyNum,
  strKeyImg,
  strKeyStr,
} from '@/interface';
import sol from '@/public/sol.png';

export const chainLogo: strKeyImg = {
  sol: sol,
};
export const chainMain: strKeyStr = {
  sol: "SOL",
};
// 定义一个对象，用于根据链类型获取精度
export const precision: ChainsTypeKeyNum = {
  sol: 9,
};
export const orderBy = {
  created_timestamp: "created_timestamp",
  liquidity: "liquidity",
  init_liquidity: "init_liquidity",
  market_cap: "market_cap",
  swaps: "swaps",
  volume: "volume",
  price_change_: "price_change_",
  quote_price: "quote_price",
};
export const orderByType = {
  asc: "asc",
  desc: "desc",
};
export const Scan = {
  sol: "https://solscan.io/tx/",
};
