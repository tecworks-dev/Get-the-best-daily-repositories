import {
  getRequest,
  postRequest,
  getRequestToken,
  postRequestToken,
} from "./requestFun";

// 获取token
export const getTokenHttp = <T>(publickey: string, signature: any) => {
  return postRequest<T>("/account/v1/get_token", {
    publickey,
    signature: [...signature],
  });
};
// meme列表
export const memeListHttp = <T>(
  chain: string,
  pumpOrRaydium: string,
  params: {
    page: number;
    size: number;
    order_by: string;
    direction: "desc" | "asc";
    performance?: string;
  }
) => {
  return getRequest<T>(
    `/defi/quotation/v1/meme/rank/${chain}/${pumpOrRaydium}`,
    { ...params }
  );
};
// 查询热门市场（Rank）
export const trendListHttp = <T>(
  chain: string,
  period: string,
  params: {
    page: number;
    size: number;
    order_by: string;
    direction: "desc" | "asc";
    filters?: [];
  }
) => {
  return getRequest<T>(
    `/defi/quotation/v1/market/rank/${chain}/swaps/${period}`,
    { ...params }
  );
};
// xtop板块
export const xTopPlateHttp = <T>(
  chain: string,
  period: string,
  params: {
    page: number;
    size: number;
  }
) => {
  return getRequest<T>(`/defi/quotation/v1/x/rank/${chain}/${period}`, {
    ...params,
  });
};
// Xtop列表
export const xTopListHttp = <T>(
  chain: string,
  period: string,
  params: {
    page: number;
    size: number;
    order_by: string;
    direction: "desc" | "asc";
    filters?: [];
  }
) => {
  return getRequest<T>(
    `/defi/quotation/v1/xtop/rank/${chain}/swaps/${period}`,
    {
      ...params,
    }
  );
};
// 获取用户的个人信息
export const getUserInfoHttp = <T>(address: string, period: string) => {
  return getRequest<T>(`/account/v1/get_account`, { address, period });
};
export function quoteHis(data: any) {
  return getRequest<any>(
    `/defi/quotation/v1/market/kline/sol/${data.pair_ca}`,
    {
      ...data,
    }
  );
}
// 搜索接口
export const searchHttp = <T>(
  chain: string,
  params: {
    q: string;
  }
) => {
  return getRequestToken<T>(`/defi/quotation/v1/tokens/${chain}/search`, {
    ...params,
    page: 1,
    size: 20,
  });
};
// 关注
export const followHttp = <T>(params: { chain_id: string; token: string }) => {
  return postRequestToken<T>(`/account/v1/add_follow_token`, {
    ...params,
  });
};
// 取消关注
export const removeFollowHttp = <T>(params: {
  chain_id: string;
  token: string;
}) => {
  return postRequestToken<T>(`/account/v1/cancel_follow_token`, {
    ...params,
  });
};
// 修改用户昵称
// export const updateNickNameHttp = <T>(params: { name: string }) => {
//   return postRequest<T>(`/account/v1/update_account`, {
//     ...params,
//   });
// };
// 查询交易详情信息
export const transactionInfoHttp = <T>(
  chain: string,
  market_address: string
) => {
  return getRequest<T>(
    `/defi/quotation/v1/market/${chain}/detail/${market_address}`
  );
};
// 交易数据列表
export const transactionListHttp = <T>(
  chain: string,
  market_address: string,
  params: {
    page: number;
    size: number;
    order_by: string;
    direction: "desc" | "asc";
  }
) => {
  return getRequest<T>(
    `/defi/quotation/v1/market/activity/${chain}/${market_address}`,
    {
      ...params,
    }
  );
};
// 蜡烛图历史数据
export const historicalTickerCandleHttp = <T>(
  market_address: string,
  params: {
    internal: string;
    from: number;
    to: number;
  }
) => {
  return getRequest<T>(
    `/defi/quotation/v1/market/kline/sol/${market_address}`,
    {
      ...params,
    }
  );
};
// 获取关注列表
export const getFollowListHttp = <T>(
  chain: string,
  period: string,
  params: {
    page: number;
    size: number;
    order_by: string;
    direction: "desc" | "asc";
    filters?: [];
  }
) => {
  return getRequestToken<T>(
    `/defi/quotation/v1/follow/rank/${chain}/swaps/${period}`,
    {
      ...params,
    }
  );
};
