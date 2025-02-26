'use client'
// 导入必要的 React 钩子、组件、请求模块以及类型定义
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation'; // 获取路由参数的钩子

import { orderBy } from '@/app/chainLogo';
import useBrowserTitle from '@/hook/useBrowserTitle';
import { ChainsType } from '@/i18n/routing'; // 链类型定义
import {
  ListParams,
  MemeListType,
} from '@/interface'; // 接口和类型定义
import { memeListHttp } from '@/request';
import {
  useEffectFiltrateStore,
  useFiltrateStore,
} from '@/store/filtrate'; // 筛选条件的状态管理
import { formatLargeNumber } from '@/utils'; // 工具函数，用于格式化时间和数值

const Table = dynamic(() => import('@/component/Table'));
const TableSortIcon = dynamic(() => import('@/component/TableSortIcon'));
const TableLiqInitial = dynamic(() => import('@/component/TableLiqInitial'));
const TableMc = dynamic(() => import('@/component/TableMc'));
const TableTxns = dynamic(() => import('@/component/TableTxns'));
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const TableTooltip = dynamic(() => import('@/component/TableTooltip'));
const TableAge = dynamic(() => import('@/component/TableAge'));
const TableTokenInfo = dynamic(() => import('@/component/TableTokenInfo'));


// 主组件
const Home = () => {
  const t = useTranslations('Meme')
  // 获取当前链类型参数
  const { chain }: { chain: ChainsType } = useParams();
  const { setFiltrate, filtrate } = useFiltrateStore(); // 筛选条件的状态管理
  const { meme } = filtrate

  // 定义组件状态
  const [shouldPoll, setShouldPoll] = useState(false); // 是否已初始化轮询
  const [stopPolling, setStopPolling] = useState(true); // 是否停止轮询
  const [loading, setLoading] = useState(true); // 是否加载中
  const [dataTable, setDataTable] = useState<ListParams[]>([]); // 表格数据
  const [pumpOrRaydium, setPumpOrRaydium] = useState(''); // 条件选择 Pump 或 Raydium
  const [condition, setCondition] = useState<string>(''); // 排序条件
  const [direction, setDirection] = useState<'desc' | 'asc' | ''>(''); // 排序方向
  const [check, setCheck] = useState(''); // 筛选条件

  // 初始化筛选条件，避免服务端和客户端渲染结果不一致
  useEffectFiltrateStore(({ meme }) => {
    setPumpOrRaydium(meme.pumpOrRaydium); // 设置 Pump 或 Raydium
    setCondition(meme.condition); // 设置排序条件
    setDirection(meme.direction as 'desc' | 'asc'); // 设置排序方向
    setCheck(meme.check); // 设置筛选条件
  });

  // 点击筛选条件时的处理函数：更新 Pump 或 Raydium 的状态
  const getPumpOrRaydium = useCallback((value: string) => {
    setPumpOrRaydium(value); // 更新状态
    setFiltrate({
      ...filtrate,
      meme: {
        ...filtrate.meme,
        pumpOrRaydium: value, // 更新筛选条件
      },
    });
  }, [filtrate])

  // 点击筛选条件时的处理函数：更新具体的筛选条件
  const getCheck = useCallback((value: string) => {
    setCheck(value); // 更新状态
    setFiltrate({
      ...filtrate,
      meme: {
        ...filtrate.meme,
        check: value, // 更新筛选条件
      },
    });
  }, [filtrate])

  // 点击排序时的处理函数：更新排序条件和方向
  const getSelectedFilter = useCallback((direction: 'desc' | 'asc', key: string) => {
    setDirection(direction); // 更新排序方向
    setCondition(key); // 更新排序字段
    setFiltrate({
      ...filtrate,
      meme: {
        ...filtrate.meme,
        direction: direction, // 更新排序方向
        condition: key, // 更新排序字段
      },
    });
  }, [filtrate]);
  // 定义筛选条件的数据
  const checkboxData = useMemo(() => [
    { label: t('newPair'), value: 'new_pair' }, // 新配对
    { label: t('soaring'), value: 'soaring' }, // 飙升
    { label: t('completed'), value: 'completed' }, // 已完成
  ], [t])

  // 定义顶部的筛选按钮
  const titleData = useMemo(() => [
    // { label: t('pump'), value: 'Pump' },
    { label: t('raydium'), value: 'Raydium' },
  ], [t])
  // 异步获取数据的方法,注意这里使用store里面存放的数据
  const memeListHttpFun = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
    isLoading && setLoading(true); // 如果需要显示加载状态
    isStopPolling && setStopPolling(true); // 如果需要停止轮询
    try {
      // 调用数据请求方法
      const { data } = await memeListHttp<MemeListType>(
        chain, // 当前链
        meme.pumpOrRaydium,
        // meme.pumpOrRaydium === titleData[1].value ? { page: 1, size: 100, order_by: meme.condition, direction: meme.direction }
        //   : { page: 1, size: 100, order_by: meme.condition, direction: meme.direction, performance: meme.check } // 请求参数
        { page: 1, size: 100, order_by: meme.condition, direction: meme.direction, performance: meme.check }
      );
      setDataTable(data.list); // 更新表格数据
    } catch (e) {
      console.log('memeListHttpFun', e); // 错误处理
    }
    isLoading && setLoading(false); // 关闭加载状态
    isStopPolling && setStopPolling(false); // 重置轮询状态
  }, [chain, meme]);



  // 定义表格列配置
  const columns = useMemo(() => [
    // 配对信息列
    {
      title: <div className="text-xl pl-5.25">{t('pairInfo')}</div>, // 列标题，显示为“Pair Info”
      dataIndex: "address", // 数据字段对应的键名
      width: "14.5rem", // 列宽度
      align: "left", // 左对齐
      fixed: 'left', // 固定在表格的左侧
      render: (_value: string, data: ListParams) => {
        // 自定义渲染，使用 TableTokenInfo 组件显示配对信息
        return <TableTokenInfo
          className="pl-5.25"
          logo={data.logo}
          symbol={data.symbol}
          chain={data.chain}
          address={data.quote_mint_address}
          twitter_username={data.twitter_username}
          website={data.website}
          telegram={data.telegram}
        />;
      }
    },
    // 年龄列
    {
      title: <TableSortIcon
        onClick={getSelectedFilter}
        lightKey={orderBy.created_timestamp}
        checkedKey={condition}
        direction={direction}
        value={t('age')}
      />, // 带排序功能的列标题
      dataIndex: "pool_creation_timestamp", // 数据字段对应的键名
      align: "center", // 居中对齐
      width: "8rem", // 列宽度
      className: "font-semibold text-sm", // 自定义样式
      render: (value: number) => <TableAge value={value} />, // 格式化年龄数据
    },
    // 流动性/初始列
    {
      title: <div className="flex items-center justify-center">
        <TableSortIcon
          onClick={getSelectedFilter}
          checkedKey={condition}
          lightKey={orderBy.liquidity}
          direction={direction}
          value={t('liq')} />
        <div className="mx-1">/</div>
        <TableSortIcon
          onClick={getSelectedFilter}
          checkedKey={condition}
          lightKey={orderBy.init_liquidity}
          direction={direction}
          value={t('initial')} />
      </div>, // 列标题，包含两个排序图标
      dataIndex: "liquidity", // 数据字段对应的键名
      align: "center", // 居中对齐
      width: "11rem", // 列宽度
      render: (_value: number, data: ListParams) => (
        <TableLiqInitial liquidity={data.liquidity} initialLiquidity={data.initial_liquidity} />
      ), // 自定义渲染，使用 TableLiqInitial 显示流动性数据
    },
    // 市值列
    {
      title: <TableSortIcon
        onClick={getSelectedFilter}
        checkedKey={condition}
        lightKey={orderBy.market_cap}
        direction={direction}
        value={t('mc')} />, // 带排序功能的列标题
      dataIndex: "market_cap", // 数据字段对应的键名
      align: "center", // 居中对齐
      width: "7.85rem", // 列宽度
      render: (value: number) => (
        <TableMc marketCap={value} />
      ), // 自定义渲染，使用 TableMc 格式化市值
    },
    // 交易次数列
    {
      title: <TableSortIcon
        onClick={getSelectedFilter}
        checkedKey={condition}
        lightKey={orderBy.swaps}
        direction={direction}
        value={t('txns')} />, // 带排序功能的列标题
      dataIndex: "swaps", // 数据字段对应的键名
      align: "center", // 居中对齐
      width: "6rem", // 列宽度
      render: (_value: number, data: ListParams) => (
        <TableTxns swaps={data.swaps} buys={data.buys} sells={data.sells} />
      ), // 自定义渲染，使用 TableTxns 显示交易次数数据
    },
    // 交易量列
    {
      title: <TableSortIcon
        onClick={getSelectedFilter}
        checkedKey={condition}
        lightKey={orderBy.volume}
        direction={direction}
        value={t('volume')} />, // 带排序功能的列标题
      dataIndex: "volume", // 数据字段对应的键名
      align: "center", // 居中对齐
      width: "7rem", // 列宽度
      className: "font-semibold text-sm", // 自定义样式
      render: (value: number) => formatLargeNumber(value), // 格式化交易量数据
    },
    // 价格列
    {
      title: <TableSortIcon
        onClick={getSelectedFilter}
        checkedKey={condition}
        lightKey={orderBy.quote_price}
        direction={direction}
        value={t('price')} />, // 带排序功能的列标题
      dataIndex: "price", // 数据字段对应的键名
      align: "center", // 居中对齐
      width: "7rem", // 列宽度
      className: "font-semibold text-sm", // 自定义样式
      render: (value: number) => (`$${formatLargeNumber(value)}`), // 格式化价格数据并加上货币符号
    },
    // // 布尔类型列：Mint Auth Disabled
    // {
    //   title: t('mintAuthDisabled'), // 列标题
    //   dataIndex: 'renounced_mint', // 数据字段对应的键名
    //   align: "center", // 居中对齐
    //   width: "7rem", // 列宽度
    //   render: (value: number) => (
    //     value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />
    //   ), // 根据布尔值显示正确或错误图标
    // },
    // // 布尔类型列：Freeze Auth Disabled
    // {
    //   title: t('freezeAuthDisabled'), // 列标题
    //   dataIndex: 'renounced_freeze_account', // 数据字段对应的键名
    //   align: "center", // 居中对齐
    //   width: "8rem", // 列宽度
    //   render: (value: number) => (
    //     value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />
    //   ), // 根据布尔值显示正确或错误图标
    // },
    // // 布尔类型列：LP Burned
    // {
    //   title: t('lpBurned'), // 列标题
    //   dataIndex: 'audit_lp_burned', // 数据字段对应的键名
    //   align: "center", // 居中对齐
    //   width: "8rem", // 列宽度
    //   render: (value: number) => (
    //     value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />
    //   ), // 根据布尔值显示正确或错误图标
    // },
    // // 布尔类型列：Top 10 Holders
    // {
    //   title: t('top10Holders'), // 列标题
    //   dataIndex: 'top_10_holder_rate', // 数据字段对应的键名
    //   align: "center", // 居中对齐
    //   width: "7rem", // 列宽度
    //   render: (value: number) => (
    //     <TableTooltip>
    //       {value < 0.3 ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />}
    //     </TableTooltip>
    //   ), // 根据布尔值显示正确或错误图标
    // },
  ], [t, getSelectedFilter, condition, direction])

  // 页面初始化时或时间范围变化时获取数据
  useEffect(() => {
    if (!shouldPoll) setShouldPoll(true); // 初始化轮询状态
    shouldPoll ? memeListHttpFun(false, true) : memeListHttpFun(true, true); // 获取数据
  }, [memeListHttpFun]);
  // 轮询逻辑
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (!stopPolling) {
      timer = setInterval(() => {
        memeListHttpFun(false, true);
      }, 3000);
    }
    return () => clearInterval(timer); // 清除定时器
  }, [stopPolling, memeListHttpFun]);
  // header title
  useBrowserTitle('OROG是最牛逼的交易工具,爱了爱了，❤️❤️❤️❤️❤️❤️❤️❤️ OROG ❤️❤️❤️❤️❤️❤️❤️❤️  ');
  return (
    <div className="px-5.75 pt-5 custom830:px-0">
      <div className="flex items-end justify-between pl-3 ml-3 pb-3.5 border-b dark:border-26 border-d6 ">
        {/* 顶部筛选条件 */}
        <div className="flex items-end custom830:flex-col custom830:justify-start custom830:items-start ">
          <div className="flex items-end custom830:mb-2">
            {titleData.map((item, index) => (
              <div
                key={index}
                onClick={() => getPumpOrRaydium(item.value)}
                className={` text-3.25xl font-extrabold ${pumpOrRaydium === item.value ? 'dark:text-white text-28' : 'dark:text-6b text-6d'} mr-6 cursor-pointer`}
              >
                {item.label}
              </div>
            ))}
          </div>

          {/* <div className={`${pumpOrRaydium === titleData[1].value ? 'hidden' : 'flex'}  items-end custom830:ml-2`}>
            {checkboxData.map((item, index) => (
              <div
                className="flex items-center mr-3.75 cursor-pointer dark:text-82 text-6d"
                onClick={() => getCheck(item.value)}
                key={index}
              >
                <div className={`w-5 h-5 rounded-4 ${check === item.value ? 'bg-6f' : 'dark:bg-53 bg-white'} mr-3.75`} />
                <div>{item.label}</div>
              </div>
            ))}
          </div> */}
        </div>
      </div>
      {/* 数据表格 */}
      <div className="custom830:hidden">
        <Table mh="16.2rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
      </div>
      <div className="hidden custom830:block">
        <Table mh="22rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
      </div>
    </div>
  );
}
export default memo(Home);