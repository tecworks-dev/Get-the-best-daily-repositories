CREATE TABLE txs (
  id bigint primary key generated always as identity,
  account text not null,                -- Solana 钱包地址
  token_in_address text not null,       -- 输入代币的合约地址
  token_in_amount numeric not null,     -- 输入代币数量
  token_out_address text not null,      -- 输出代币的合约地址
  token_out_amount numeric not null,    -- 输出代币数量
  timestamp bigint not null,            -- Unix 时间戳
  signature text not null,              -- 交易签名
  description text,                     -- 交易描述
  created_at timestamptz default now()  -- 记录创建时间
);

-- 添加索引以提高查询性能
create index idx_txs_account on txs(account);
create index idx_txs_timestamp on txs(timestamp);

CREATE TABLE wallets (
  id serial primary key,
  address text unique not null,
  name text
);