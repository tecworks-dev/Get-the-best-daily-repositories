import {
  CurrencyAmount,
  Liquidity,
  LIQUIDITY_STATE_LAYOUT_V4,
  LiquidityPoolKeys,
  Market,
  MARKET_STATE_LAYOUT_V3,
  Percent,
  SPL_ACCOUNT_LAYOUT,
  Token,
  TokenAccount,
  TokenAmount,
} from '@raydium-io/raydium-sdk'; // 导入 Raydium SDK 中用于流动性池操作的相关模块
import { TOKEN_PROGRAM_ID } from '@solana/spl-token'; // 导入 SPL Token 相关库
import {
  Connection,
  PublicKey,
  Transaction,
} from '@solana/web3.js'; // 导入 Solana 的连接、公共密钥、交易模块

const RAYDIUM_V4_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"; // Raydium V4 程序 ID

// 代币转换函数：从 wSOL 转换为 USDC
const transactionSol = async (
  connection: Connection, // Solana 网络连接对象
  provider: any, // 提供者对象，用于签署和发送交易
  address: string, // 用户的地址（钱包地址）
  quoteMintAddress: string, // 目标代币（如 USDC）的 mint 地址
  baseMintAddress: string, //  默认 wSOL 的 mint 地址
  rawAmountIn: number, // 输入金额
  slippage: number // 滑点
) => {
  // 创建公共密钥对象
  const addressPublicKey = new PublicKey(address);
  const quoteMintAddressPublicKey = new PublicKey(quoteMintAddress);
  const baseMintAddressPublicKey = new PublicKey(baseMintAddress);

  // 获取 Raydium 程序账户
  const getProgramAccounts = async (
    baseMint: PublicKey, // 基础代币（如 wSOL）的 mint 地址
    quoteMint: PublicKey // 目标代币（如 USDC）的 mint 地址
  ) => {
    try {
      // 查询 Raydium 程序账户，使用过滤器来找到匹配的账户
      const data = await connection.getProgramAccounts(
        new PublicKey(RAYDIUM_V4_PROGRAM_ID),
        {
          filters: [
            { dataSize: LIQUIDITY_STATE_LAYOUT_V4.span }, // 过滤出数据大小匹配的账户
            {
              memcmp: {
                offset: LIQUIDITY_STATE_LAYOUT_V4.offsetOf("baseMint"), // 查找 baseMint 匹配的账户
                bytes: baseMint.toBase58(), // 使用 baseMint 地址进行比较
              },
            },
            {
              memcmp: {
                offset: LIQUIDITY_STATE_LAYOUT_V4.offsetOf("quoteMint"), // 查找 quoteMint 匹配的账户
                bytes: quoteMint.toBase58(), // 使用 quoteMint 地址进行比较
              },
            },
          ],
        }
      );
      return data; // 返回找到的账户数据
    } catch (e) {
      console.error("Error in getProgramAccounts:", e); // 捕获并输出错误信息
      return null; // 错误时返回 null
    }
  };

  // 获取 Raydium 池的所有相关账户
  const getProgramAccountsAll = async (
    baseMint: PublicKey,
    quoteMint: PublicKey
  ) => {
    try {
      // 使用 Promise.all 同时查询两个方向的账户数据（baseMint -> quoteMint 和 quoteMint -> baseMint）
      const response = await Promise.all([
        getProgramAccounts(baseMint, quoteMint),
        getProgramAccounts(quoteMint, baseMint),
      ]);
      return response.filter((r) => (r || []).length > 0)[0] || []; // 返回第一个非空的结果，如果没有则返回空数组
    } catch (e) {
      console.error("Error in getProgramAccountsAll:", e); // 捕获并输出错误信息
      return []; // 错误时返回空数组
    }
  };

  // 获取流动性池信息
  const getPoolInfo = async (baseMint: PublicKey, quoteMint: PublicKey) => {
    try {
      const layout = LIQUIDITY_STATE_LAYOUT_V4; // 获取流动性池的布局

      const programData = await getProgramAccountsAll(baseMint, quoteMint); // 获取程序账户数据
      if (!programData || programData.length === 0) return null; // 如果没有数据，返回 null

      // 解析流动性池数据，使用 layout 解码
      const collectedPoolResults = programData
        .map((info) => ({
          id: new PublicKey(info.pubkey),
          version: 4,
          programId: new PublicKey(RAYDIUM_V4_PROGRAM_ID),
          ...layout.decode(info.account.data),
        }))
        .flat();

      const pool = collectedPoolResults[0]; // 获取池的第一个结果

      if (!pool) return null; // 如果没有池数据，返回 null

      // 获取市场数据
      const marketData = await connection.getAccountInfo(pool.marketId);
      if (!marketData) return null; // 如果市场数据为空，返回 null

      const market = {
        programId: marketData.owner,
        ...MARKET_STATE_LAYOUT_V3.decode(marketData.data), // 使用 layout 解码市场数据
      };

      // 获取流动性池的授权地址
      const authority = Liquidity.getAssociatedAuthority({
        programId: new PublicKey(RAYDIUM_V4_PROGRAM_ID),
      }).publicKey;

      const marketProgramId = market.programId; // 获取市场程序 ID

      // 构建池的关键信息对象
      const poolKeys = {
        id: pool.id,
        baseMint: pool.baseMint,
        quoteMint: pool.quoteMint,
        lpMint: pool.lpMint,
        baseDecimals: Number.parseInt(pool.baseDecimal.toString()),
        quoteDecimals: Number.parseInt(pool.quoteDecimal.toString()),
        lpDecimals: Number.parseInt(pool.baseDecimal.toString()),
        version: pool.version,
        programId: pool.programId,
        openOrders: pool.openOrders,
        targetOrders: pool.targetOrders,
        baseVault: pool.baseVault,
        quoteVault: pool.quoteVault,
        marketVersion: 3,
        authority: authority,
        marketProgramId,
        marketId: market.ownAddress,
        marketAuthority: Market.getAssociatedAuthority({
          programId: marketProgramId,
          marketId: market.ownAddress,
        }).publicKey,
        marketBaseVault: market.baseVault,
        marketQuoteVault: market.quoteVault,
        marketBids: market.bids,
        marketAsks: market.asks,
        marketEventQueue: market.eventQueue,
        withdrawQueue: pool.withdrawQueue,
        lpVault: pool.lpVault,
        lookupTableAccount: PublicKey.default,
      } as LiquidityPoolKeys;

      return poolKeys; // 返回池的关键信息
    } catch (e) {
      console.error("Error in getPoolInfo:", e); // 捕获并输出错误信息
      return null; // 错误时返回 null
    }
  };

  // 计算交易输出（根据输入金额计算输出金额）
  const calcAmountOut = async (
    poolKeys: LiquidityPoolKeys, // 流动性池的关键信息
    rawAmountIn: number, // 输入金额
    slippage: number = 10, // 滑点，默认值为 5%
    swapInDirection: boolean // 是否按照指定的方向交换
  ) => {
    try {
      // 获取池的最新信息
      const poolInfo = await Liquidity.fetchInfo({
        connection: connection,
        poolKeys,
      });
      let currencyInMint = poolKeys.baseMint; // 输入代币的 mint 地址
      let currencyInDecimals = poolInfo.baseDecimals; // 输入代币的精度
      let currencyOutMint = poolKeys.quoteMint; // 输出代币的 mint 地址
      let currencyOutDecimals = poolInfo.quoteDecimals; // 输出代币的精度

      // 根据交换方向调整输入和输出代币
      if (!swapInDirection) {
        currencyInMint = poolKeys.quoteMint;
        currencyInDecimals = poolInfo.quoteDecimals;
        currencyOutMint = poolKeys.baseMint;
        currencyOutDecimals = poolInfo.baseDecimals;
      }

      // 创建输入代币和输出代币的 Token 实例
      const currencyIn = new Token(
        TOKEN_PROGRAM_ID,
        currencyInMint,
        currencyInDecimals
      );
      const amountIn = new TokenAmount(
        currencyIn,
        // parseUnits(rawAmountIn + "", currencyInDecimals),
        rawAmountIn.toFixed(currencyInDecimals),
        false
      );
      const currencyOut = new Token(
        TOKEN_PROGRAM_ID,
        currencyOutMint,
        currencyOutDecimals
      );

      // 创建滑点参数
      const slippageX = new Percent(slippage, 100); // 5% slippage

      // 计算输出金额、最小输出金额等信息
      const result = Liquidity.computeAmountOut({
        poolKeys,
        poolInfo,
        amountIn,
        currencyOut,
        slippage: slippageX,
      });

      if (!result) return null; // 如果没有计算结果，返回 null

      const {
        amountOut,
        minAmountOut,
        currentPrice,
        executionPrice,
        priceImpact,
        fee,
      } = result;

      // 返回计算结果
      return {
        amountIn,
        amountOut,
        minAmountOut,
        currentPrice,
        executionPrice,
        priceImpact,
        fee,
      };
    } catch (e) {
      console.error("Error in calcAmountOut:", e); // 捕获并输出错误信息
      return null; // 错误时返回 null
    }
  };

  // 获取钱包的代币账户
  const getOwnerTokenAccounts = async () => {
    try {
      const walletTokenAccount = await connection.getTokenAccountsByOwner(
        addressPublicKey,
        {
          programId: TOKEN_PROGRAM_ID,
        }
      );

      return walletTokenAccount.value.map((i) => ({
        pubkey: i.pubkey,
        programId: i.account.owner,
        accountInfo: SPL_ACCOUNT_LAYOUT.decode(i.account.data),
      }));
    } catch (e) {
      console.error("Error in getOwnerTokenAccounts:", e); // 捕获并输出错误信息
      return []; // 错误时返回空数组
    }
  };
  // 创建交换指令
  const makeSwapInstruction = (
    poolKeys: LiquidityPoolKeys,
    userTokenAccounts: TokenAccount[],
    amountIn: TokenAmount,
    minAmountOut: CurrencyAmount
  ) => {
    try {
      if (!poolKeys) return null;
      const data = Liquidity.makeSwapInstructionSimple({
        connection,
        makeTxVersion: false ? 0 : 1, // 设置交易版本（默认使用版本 1）
        poolKeys: poolKeys,
        userKeys: {
          tokenAccounts: userTokenAccounts,
          owner: addressPublicKey,
        },
        amountIn: amountIn,
        amountOut: minAmountOut,
        fixedSide: "in", // 固定输入金额
        config: {
          bypassAssociatedCheck: false, // 不跳过关联账户检查
        },
        // computeBudgetConfig: {
        //   microLamports: 100000, // 计算预算配置，指定最大费用
        // },
      });
      return data;
    } catch (e) {
      console.error("Error in makeSwapInstruction:", e); // 捕获并输出错误信息
      return null; // 错误时返回 null
    }
  };
  try {
    // 获取用户代币账户信息
    const userTokenAccounts = await getOwnerTokenAccounts();
    // 获取流动性池信息
    const poolKeys = await getPoolInfo(
      baseMintAddressPublicKey,
      quoteMintAddressPublicKey
    );
    if (!poolKeys) throw new Error(`transactionSol error poolKey:${poolKeys}`);
    // 计算输入金额对应的输出金额
    const calcAmountOutData = await calcAmountOut(
      poolKeys,
      rawAmountIn,
      slippage,
      quoteMintAddress === poolKeys.quoteMint.toString()
    );
    if (!calcAmountOutData)
      throw new Error(
        `transactionSol error calcAmountOutData:${calcAmountOutData}`
      );
    const { minAmountOut, amountIn } = calcAmountOutData;
    // 获取并构建交易指令
    const swapInstruction = await makeSwapInstruction(
      poolKeys,
      userTokenAccounts,
      amountIn,
      minAmountOut
    );
    if (!swapInstruction)
      throw new Error(
        `transactionSol error swapInstruction:${swapInstruction}`
      );

    const instructions =
      swapInstruction.innerTransactions[0].instructions.filter(Boolean); // 获取内部交易指令

    // 获取最新的区块哈希
    const { blockhash } = await connection.getLatestBlockhash("max");

    // 创建交易对象并添加指令
    const transaction = new Transaction();
    if (swapInstruction) {
      transaction.add(...instructions);
      transaction.recentBlockhash = blockhash; // 设置区块哈希
      transaction.feePayer = addressPublicKey; // 设置费用支付者
    }

    // 签署并发送交易
    const { signature } = await provider.signAndSendTransaction(transaction);
    return signature; // 返回交易hash
  } catch (e) {
    throw new Error(`transactionSol error  ${e}`);
  }
};

export default transactionSol;
