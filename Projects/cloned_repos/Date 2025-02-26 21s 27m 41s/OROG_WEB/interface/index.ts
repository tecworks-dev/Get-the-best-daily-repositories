import { ChainsType } from "@/i18n/routing";
import { StaticImageData } from "next/image";
import { ReactNode } from "react";

export interface WindowWithWallets extends Window {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}
export interface strKeyImg {
  [key: string]: StaticImageData;
}
export interface strKeyReactNode {
  [key: string]: ReactNode;
}
export interface strKeyStr {
  [key: string]: string;
}
export type ChainsTypeKeyFun = {
  [key in ChainsType]: Function;
};
export type ChainsTypeKeyStr = {
  [key in ChainsType]: string;
};
export type ChainsTypeKeyNum = {
  [key in ChainsType]: number;
};
export type ChainsTypeKeyArr<T> = {
  [key in ChainsType]: T[];
};
// axios 返回的数据格式
export interface MyResponseType<T = any> {
  code: number;
  msg: string;
  data: T;
}
// 返回的token
export interface TokenType {
  token: string;
}
// 搜索返回的数据
export interface SearchTokenType {
  follow: boolean; // 是否关注
  logo: string; // logo
  symbol: string; // 货币符号
  chain: ChainsType; // 链
  address: string; // // 市场地址
  quote_mint_address: string; // 代币地址
  liquidity: number; // 24小时池流动性
  volume_24h: number; // 池交易量
}
// 搜索返回的数据列表
export interface SearchListType {
  list: SearchTokenType[];
}
// 搜索返回的板块数据
export interface XtopPlateTokenType {
  logo: string; // logo
  symbol: string; // 货币符号
  chain: ChainsType; // 链
  address: string; // // 市场地址
  quote_mint_address: string; // 代币地址
  market_cap: number; // 市值
  volume: number; // 池交易量
  price: number; // 价格
  initial_liquidity: number; // 初始池流动性
  liquidity: number; // 24小时池流动性
  trend: boolean; // 板块趋势
  telegram: string; // telegram
  twitter_username: string; // twitter_username
  website: string; // 官网
}
// xtop 排行板块以及里面的数和header默认数据
export type XtopPlateType = {
  plate: string; // 板块名称
  plate_trend: boolean; // 板块趋势
  plate_market_cap: number; // 板块总市值
  plate_swaps: number; // 板块交易数
  data: XtopPlateTokenType[];
}[];
export interface MyWalletInfoType {
  nickname: string; // 昵称
  initial_funding: number; // 初始资金
  funding: number; // 资金
  money_per_day: number[]; // 资金每日变化数组
  unrealized_profits: number; // 未实现盈亏
  total_profit: number; // 总盈亏
  buy: number; // 买入次数
  sell: number; // 卖出次数
}
// 市场
export interface ListParams {
  address: string; // 市场地址
  base_mint_address: string; // 基础代币铸造地址 wsol
  base_price: number; // 基础价格 wsol
  base_token_address: string; // 基础代币地址 wsol
  buys: number; // 买入笔数
  chain: ChainsType; // 所属链
  creator_close: boolean; // 是否关闭创建者权限
  creator_token_status: string; // 创建者代币状态
  holder_count: number; // 持有人数量
  hot_level: number; // 热度等级
  id: number; // 唯一标识符
  initial_liquidity: number; // 初始流动性
  is_show_alert: boolean; // 是否显示警告
  launchpad: string; // 启动平台
  liquidity: number; // 流动性
  logo: string; // 代币 logo 链接
  market_cap: number; // 市值
  open_timestamp: number; // 开放时间戳
  pool_creation_timestamp: number; // 池子创建时间戳
  price: number; // 价格
  price_change_percent1h: number; // 1 小时价格变动百分比
  price_change_percent1m: number; // 1 分钟价格变动百分比
  price_change_percent5m: number; // 5 分钟价格变动百分比
  price_change_percent6h: number; // 6 小时价格变动百分比
  price_change_percent24h: number; // 24 小时价格变动百分比
  price_change_percent30m: number; // 30 分钟价格变动百分比
  quote_mint_address: string; // 代币铸造地址
  quote_token_address: string; // 代币地址
  renounced_freeze_account: number; // 放弃冻结账户权限
  renounced_mint: number; // 放弃铸造权限
  audit_lp_burned: number; // LP 是否已销毁
  sells: number; // 卖出笔数
  swaps: number; // 交换笔数
  symbol: string; // 代币符号
  telegram: string; // Telegram 链接
  top_10_holder_rate: number; // 前 10 持有人占比
  twitter_username: string; // Twitter 用户名
  volume: number; // 交易量
  website: string; // 官网链接
}
export interface EarningsListType {
  price: number; // 代币价格
  price_change_percent24h: number; // 24 小时价格变动百分比
  price_change24h: number; // 24 小时价格变动
  id: number; // 唯一标识符
  chain: ChainsType; // 所属链
  logo: string; // 代币 logo 链接
  address: string; // // 市场地址
  quote_mint_address: string; // 代币铸造地址
  sells: number; // 卖出笔数
  sells_num: number; // 卖出数量
  sells_num_usd: number; // 卖入数量（美元）
  buys_num_usd: number; // 买入数量（美元）
  buys: number; // 买入笔数
  buys_num: number; // 买入数量
  swaps: number; // 30天交换数量
  symbol: string; // 代币符号
  telegram: string; // Telegram 链接
  twitter_username: string; // Twitter 用户名
  volume: number; // 交易量
  website: string; // 官网链接
  enter_address: number; // 代币进入地址的时间戳
  unrealized_profits: number; // 未实现盈亏
  unrealized_profits_percent: number; // 未实现盈亏百分比
  total_profit: number; // 总盈亏
  total_profit_percent: number; // 总盈亏百分比
  token_price_usd: number; // 余额（USDT）
  token_num: number; // 代币数量当前地址
  position_percent: number; // 仓位百分比
  bought_avg_price: number; // 买入均价
  sold_avg_price: number; // 卖出均价
  main_price: number; // 主网币价
}
// meme 列表接口
export interface MemeListType {
  total: number;
  list: ListParams[];
}
//查询热门市场接口
export interface TrendListType {
  total: number;
  list: ListParams[];
}
// xtop 列表接口
export interface XtopListType {
  total: number;
  list: ListParams[];
}
export interface FollowListParams extends ListParams {
  price_change24h: number; // 24 小时价格变动
}
// follow 列表接口
export interface FollowListType {
  total: number;
  list: FollowListParams[];
}
// 我的钱包list
export interface MyWalletListType {
  total: number;
  list: EarningsListType[];
}

export interface TokenInfoType {
  address: string;
  biggest_pool_address: string;
  circulating_supply: number;
  creation_timestamp: number;
  decimals: number;
  base_price: number;
  dev: {
    address: string;
    creator_address: string;
    creator_token_balance: number;
    creator_token_status: boolean;
    telegram: string;
    twitter_username: string;
    website: string;
  };
  follow: boolean;
  holder_count: number;
  liquidity: number;
  logo: string;
  max_supply: number;
  name: string;
  open_timestamp: number;
  pool: {
    address: string;
    base_address: string;
    base_reserve: number;
    base_reserve_value: number;
    base_vault_address: string;
    creation_timestamp: number;
    creator: string;
    exchange: string;
    initial_base_reserve: number;
    initial_liquidity: number;
    initial_quote_reserve: number;
    liquidity: number;
    quote_address: string;
    quote_mint_address: string;
    quote_reserve: number;
    quote_reserve_value: number;
    quote_symbol: string;
    quote_vault_address: string;
    token0_address: string;
    token1_address: string;
  };
  price: {
    address: string;
    buy_volume_1h: number;
    buy_volume_1m: number;
    buy_volume_5m: number;
    buy_volume_6h: number;
    buy_volume_24h: number;
    buys: number;
    buys_1h: number;
    buys_1m: number;
    buys_5m: number;
    buys_6h: number;
    buys_24h: number;
    market_cap: number;
    price: number;
    price_1h: number;
    price_1m: number;
    price_5m: number;
    price_6h: number;
    price_24h: number;
    sell_volume_1h: number;
    sell_volume_1m: number;
    sell_volume_5m: number;
    sell_volume_6h: number;
    sell_volume_24h: number;
    sells: number;
    sells_1h: number;
    sells_1m: number;
    sells_5m: number;
    sells_6h: number;
    sells_24h: number;
    swaps: number;
    volume: number;
    volume_1h: number;
    volume_1m: number;
    volume_5m: number;
    volume_6h: number;
    volume_24h: number;
  };
  symbol: string;
  total_supply: number;
}

export interface transactionType {
  base_amount: number; // 主网币数量
  base_price: number; // 主网币价格
  chain: ChainsType;
  maker_address: string; // 交易员地址
  market_address: string; // 交易市场地址
  market_type: string; // 交易市场类型
  quote_amount: number; // 应用币数量
  quote_price: number; // 应用币价格
  slot: number; // 区块高度
  swap_type: 0 | 1 | 2 | 3 | 4; // 买入卖出类型
  timestamp: number; // 时间戳
  tx_hash: string; // 交易哈希
  volume: number; // 交易量
}
export interface transactionListType {
  total: number;
  list: transactionType[];
}
// 蜡烛图历史数据
export interface historicalTickerCandleType {
  close: number;
  high: number;
  low: number;
  open: number;
  volume: number;
  time: number;
}
// 蜡烛图历史数据数组
export interface historicalTickerCandleListType {
  items: historicalTickerCandleType[];
}
