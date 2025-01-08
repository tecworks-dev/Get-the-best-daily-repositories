from typing import Dict, Any, List
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import json 
import numpy as np


def get_financial_metrics(symbol: str) -> Dict[str, Any]:
    """获取财务指标数据"""
    try:
        print(f"\n正在获取 {symbol} 的财务指标数据...")

        # 获取实时行情数据（用于市值和估值比率）
        print("获取实时行情...")
        realtime_data = ak.stock_zh_a_spot_em()
        if realtime_data is None or realtime_data.empty:
            print("警告：无法获取实时行情数据")
            return [{}]

        stock_data = realtime_data[realtime_data['代码'] == symbol]
        if stock_data.empty:
            print(f"警告：未找到股票 {symbol} 的实时行情数据")
            return [{}]

        stock_data = stock_data.iloc[0]
        print("成功获取实时行情数据")

        # 获取新浪财务指标
        print("\n获取新浪财务指标...")
        current_year = datetime.now().year
        financial_data = ak.stock_financial_analysis_indicator(
            symbol=symbol, start_year=str(current_year-1))
        if financial_data is None or financial_data.empty:
            print("警告：无法获取新浪财务指标数据")
            return [{}]

        # 按日期排序并获取最新的数据
        financial_data['日期'] = pd.to_datetime(financial_data['日期'])
        financial_data = financial_data.sort_values('日期', ascending=False)
        latest_financial = financial_data.iloc[0] if not financial_data.empty else pd.Series(
        )
        print(f"成功获取新浪财务指标数据，共 {len(financial_data)} 条记录")
        print(f"最新数据日期：{latest_financial.get('日期')}")

        # 获取利润表数据（用于计算 price_to_sales）
        print("\n获取利润表数据...")
        try:
            income_statement = ak.stock_financial_report_sina(
                stock=f"sh{symbol}", symbol="利润表")
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                print("成功获取利润表数据")
            else:
                print("警告：无法获取利润表数据")
                latest_income = pd.Series()
        except Exception as e:
            print(f"获取利润表数据时出错：{e}")
            latest_income = pd.Series()

        # 构建完整指标数据
        print("\n构建指标数据...")
        try:
            def convert_percentage(value: float) -> float:
                """将百分比值转换为小数"""
                try:
                    return float(value) / 100.0 if value is not None else 0.0
                except:
                    return 0.0

            all_metrics = {
                # 市场数据
                "market_cap": float(stock_data.get("总市值", 0)),
                "float_market_cap": float(stock_data.get("流通市值", 0)),

                # 盈利数据
                "revenue": float(latest_income.get("营业总收入", 0)),
                "net_income": float(latest_income.get("净利润", 0)),
                "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)),
                "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)),
                "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)),

                # 增长指标
                "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)),
                "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)),
                "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)),

                # 财务健康指标
                "current_ratio": float(latest_financial.get("流动比率", 0)),
                "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)),
                "free_cash_flow_per_share": float(latest_financial.get("每股经营性现金流(元)", 0)),
                "earnings_per_share": float(latest_financial.get("加权每股收益(元)", 0)),

                # 估值比率
                "pe_ratio": float(stock_data.get("市盈率-动态", 0)),
                "price_to_book": float(stock_data.get("市净率", 0)),
                "price_to_sales": float(stock_data.get("总市值", 0)) / float(latest_income.get("营业总收入", 1)) if float(latest_income.get("营业总收入", 0)) > 0 else 0,
            }

            # 只返回 agent 需要的指标
            agent_metrics = {
                # 盈利能力指标
                "return_on_equity": all_metrics["return_on_equity"],
                "net_margin": all_metrics["net_margin"],
                "operating_margin": all_metrics["operating_margin"],

                # 增长指标
                "revenue_growth": all_metrics["revenue_growth"],
                "earnings_growth": all_metrics["earnings_growth"],
                "book_value_growth": all_metrics["book_value_growth"],

                # 财务健康指标
                "current_ratio": all_metrics["current_ratio"],
                "debt_to_equity": all_metrics["debt_to_equity"],
                "free_cash_flow_per_share": all_metrics["free_cash_flow_per_share"],
                "earnings_per_share": all_metrics["earnings_per_share"],

                # 估值比率
                "pe_ratio": all_metrics["pe_ratio"],
                "price_to_book": all_metrics["price_to_book"],
                "price_to_sales": all_metrics["price_to_sales"],
            }

            print("成功构建指标数据")

            # 打印所有获取到的指标数据（用于调试）
            print("\n获取到的完整指标数据：")
            for key, value in all_metrics.items():
                print(f"{key}: {value}")

            print("\n传递给 agent 的指标数据：")
            for key, value in agent_metrics.items():
                print(f"{key}: {value}")

            return [agent_metrics]

        except Exception as e:
            print(f"构建指标数据时出错：{e}")
            return [{}]

    except Exception as e:
        print(f"获取财务指标时出错：{e}")
        return [{}]


def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """获取财务报表数据"""
    try:
        print(f"\n正在获取 {symbol} 的财务报表数据...")

        # 获取资产负债表数据
        print("\n获取资产负债表数据...")
        try:
            balance_sheet = ak.stock_financial_report_sina(
                stock=f"sh{symbol}", symbol="资产负债表")
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[0]
                previous_balance = balance_sheet.iloc[1] if len(
                    balance_sheet) > 1 else balance_sheet.iloc[0]
                print("成功获取资产负债表数据")
            else:
                print("警告：无法获取资产负债表数据")
                latest_balance = pd.Series()
                previous_balance = pd.Series()
        except Exception as e:
            print(f"获取资产负债表数据时出错：{e}")
            latest_balance = pd.Series()
            previous_balance = pd.Series()

        # 获取利润表数据
        print("\n获取利润表数据...")
        try:
            income_statement = ak.stock_financial_report_sina(
                stock=f"sh{symbol}", symbol="利润表")
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                previous_income = income_statement.iloc[1] if len(
                    income_statement) > 1 else income_statement.iloc[0]
                print("成功获取利润表数据")
            else:
                print("警告：无法获取利润表数据")
                latest_income = pd.Series()
                previous_income = pd.Series()
        except Exception as e:
            print(f"获取利润表数据时出错：{e}")
            latest_income = pd.Series()
            previous_income = pd.Series()

        # 获取现金流量表数据
        print("\n获取现金流量表数据...")
        try:
            cash_flow = ak.stock_financial_report_sina(
                stock=f"sh{symbol}", symbol="现金流量表")
            if not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[0]
                previous_cash_flow = cash_flow.iloc[1] if len(
                    cash_flow) > 1 else cash_flow.iloc[0]
                print("成功获取现金流量表数据")
            else:
                print("警告：无法获取现金流量表数据")
                latest_cash_flow = pd.Series()
                previous_cash_flow = pd.Series()
        except Exception as e:
            print(f"获取现金流量表数据时出错：{e}")
            latest_cash_flow = pd.Series()
            previous_cash_flow = pd.Series()

        # 构建财务数据
        line_items = []
        try:
            # 处理最新期间数据
            current_item = {
                # 从利润表获取
                "net_income": float(latest_income.get("净利润", 0)),
                "operating_revenue": float(latest_income.get("营业总收入", 0)),
                "operating_profit": float(latest_income.get("营业利润", 0)),

                # 从资产负债表计算营运资金
                "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(latest_balance.get("流动负债合计", 0)),

                # 从现金流量表获取
                "depreciation_and_amortization": float(latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(current_item)
            print("成功处理最新期间数据")

            # 处理上一期间数据
            previous_item = {
                "net_income": float(previous_income.get("净利润", 0)),
                "operating_revenue": float(previous_income.get("营业总收入", 0)),
                "operating_profit": float(previous_income.get("营业利润", 0)),
                "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(previous_balance.get("流动负债合计", 0)),
                "depreciation_and_amortization": float(previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(previous_item)
            print("成功处理上一期间数据")

        except Exception as e:
            print(f"处理财务数据时出错：{e}")
            default_item = {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            }
            line_items = [default_item, default_item]

        return line_items

    except Exception as e:
        print(f"获取财务报表时出错：{e}")
        default_item = {
            "net_income": 0,
            "operating_revenue": 0,
            "operating_profit": 0,
            "working_capital": 0,
            "depreciation_and_amortization": 0,
            "capital_expenditure": 0,
            "free_cash_flow": 0
        }
        return [default_item, default_item]


def get_market_data(symbol: str) -> Dict[str, Any]:
    """获取市场数据"""
    try:
        # 获取实时行情
        realtime_data = ak.stock_zh_a_spot_em()
        stock_data = realtime_data[realtime_data['代码'] == symbol].iloc[0]

        return {
            "market_cap": float(stock_data.get("总市值", 0)),
            "volume": float(stock_data.get("成交量", 0)),
            # A股没有平均成交量，暂用当日成交量
            "average_volume": float(stock_data.get("成交量", 0)),
            "fifty_two_week_high": float(stock_data.get("52周最高", 0)),
            "fifty_two_week_low": float(stock_data.get("52周最低", 0))
        }

    except Exception as e:
        print(f"Error getting market data: {e}")
        return {}


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """获取历史价格数据

    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
        end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
        adjust: 复权类型，可选值：
               - "": 不复权
               - "qfq": 前复权（默认）
               - "hfq": 后复权

    Returns:
        包含以下列的DataFrame：
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量（手）
        - amount: 成交额（元）
        - amplitude: 振幅（%）
        - pct_change: 涨跌幅（%）
        - change_amount: 涨跌额（元）
        - turnover: 换手率（%）

        技术指标：
        - momentum_1m: 1个月动量
        - momentum_3m: 3个月动量
        - momentum_6m: 6个月动量
        - volume_momentum: 成交量动量
        - historical_volatility: 历史波动率
        - volatility_regime: 波动率区间
        - volatility_z_score: 波动率Z分数
        - atr_ratio: 真实波动幅度比率
        - hurst_exponent: 赫斯特指数
        - skewness: 偏度
        - kurtosis: 峰度
    """
    try:
        # 获取当前日期和昨天的日期
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        # 如果没有提供日期，默认使用昨天作为结束日期
        if not end_date:
            end_date = yesterday  # 使用昨天作为结束日期
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            # 确保end_date不会超过昨天
            if end_date > yesterday:
                end_date = yesterday

        if not start_date:
            start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        print(f"\n正在获取 {symbol} 的历史行情数据...")
        print(f"开始日期：{start_date.strftime('%Y-%m-%d')}")
        print(f"结束日期：{end_date.strftime('%Y-%m-%d')}")

        def get_and_process_data(start_date, end_date):
            """获取并处理数据，包括重命名列等操作"""
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust=adjust
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列以匹配技术分析代理的需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover"
            })

            # 确保日期列为datetime类型
            df["date"] = pd.to_datetime(df["date"])
            return df

        # 获取历史行情数据
        df = get_and_process_data(start_date, end_date)

        if df is None or df.empty:
            print(f"警告：未获取到 {symbol} 的历史行情数据")
            return pd.DataFrame()

        # 检查数据量是否足够
        min_required_days = 120  # 至少需要120个交易日的数据
        if len(df) < min_required_days:
            print(
                f"警告：获取到的数据量（{len(df)}条）不足以计算所有技术指标（需要至少{min_required_days}条）")
            print("尝试获取更长时间范围的数据...")

            # 扩大时间范围到2年
            start_date = end_date - timedelta(days=730)
            df = get_and_process_data(start_date, end_date)

            if len(df) < min_required_days:
                print(f"警告：即使扩大时间范围，数据量（{len(df)}条）仍然不足")

        # 计算动量指标
        df["momentum_1m"] = df["close"].pct_change(periods=20)  # 20个交易日约等于1个月
        df["momentum_3m"] = df["close"].pct_change(periods=60)  # 60个交易日约等于3个月
        df["momentum_6m"] = df["close"].pct_change(
            periods=120)  # 120个交易日约等于6个月

        # 计算成交量动量（相对于20日平均成交量的变化）
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_momentum"] = df["volume"] / df["volume_ma20"]

        # 计算波动率指标
        # 1. 历史波动率 (20日)
        returns = df["close"].pct_change()
        df["historical_volatility"] = returns.rolling(
            window=20).std() * np.sqrt(252)  # 年化

        # 2. 波动率区间 (相对于过去120天的波动率的位置)
        volatility_120d = returns.rolling(window=120).std() * np.sqrt(252)
        vol_min = volatility_120d.rolling(window=120).min()
        vol_max = volatility_120d.rolling(window=120).max()
        vol_range = vol_max - vol_min
        df["volatility_regime"] = np.where(
            vol_range > 0,
            (df["historical_volatility"] - vol_min) / vol_range,
            0  # 当范围为0时返回0
        )

        # 3. 波动率Z分数
        vol_mean = df["historical_volatility"].rolling(window=120).mean()
        vol_std = df["historical_volatility"].rolling(window=120).std()
        df["volatility_z_score"] = (
            df["historical_volatility"] - vol_mean) / vol_std

        # 4. ATR比率
        tr = pd.DataFrame()
        tr["h-l"] = df["high"] - df["low"]
        tr["h-pc"] = abs(df["high"] - df["close"].shift(1))
        tr["l-pc"] = abs(df["low"] - df["close"].shift(1))
        tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["atr"] = tr["tr"].rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # 计算统计套利指标
        # 1. 赫斯特指数 (使用过去120天的数据)
        def calculate_hurst(series):
            """
            计算Hurst指数。

            Args:
                series: 价格序列

            Returns:
                float: Hurst指数，或在计算失败时返回np.nan
            """
            try:
                series = series.dropna()
                if len(series) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用对数收益率
                log_returns = np.log(series / series.shift(1)).dropna()
                if len(log_returns) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用更小的lag范围
                # 减少lag范围到2-10天
                lags = range(2, min(11, len(log_returns) // 4))

                # 计算每个lag的标准差
                tau = []
                for lag in lags:
                    # 计算滚动标准差
                    std = log_returns.rolling(window=lag).std().dropna()
                    if len(std) > 0:
                        tau.append(np.mean(std))

                # 基本的数值检查
                if len(tau) < 3:  # 进一步降低最小要求
                    return np.nan

                # 使用对数回归
                lags_log = np.log(list(lags))
                tau_log = np.log(tau)

                # 计算回归系数
                reg = np.polyfit(lags_log, tau_log, 1)
                hurst = reg[0] / 2.0

                # 只保留基本的数值检查
                if np.isnan(hurst) or np.isinf(hurst):
                    return np.nan

                return hurst

            except Exception as e:
                return np.nan

        # 使用对数收益率计算Hurst指数
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["hurst_exponent"] = log_returns.rolling(
            window=120,
            min_periods=60  # 要求至少60个数据点
        ).apply(calculate_hurst)

        # 2. 偏度 (20日)
        df["skewness"] = returns.rolling(window=20).skew()

        # 3. 峰度 (20日)
        df["kurtosis"] = returns.rolling(window=20).kurt()

        # 按日期升序排序
        df = df.sort_values("date")

        # 重置索引
        df = df.reset_index(drop=True)

        print(f"成功获取历史行情数据，共 {len(df)} 条记录")

        # 检查并报告NaN值
        nan_columns = df.isna().sum()
        if nan_columns.any():
            print("\n警告：以下指标存在NaN值：")
            for col, nan_count in nan_columns[nan_columns > 0].items():
                print(f"- {col}: {nan_count}条")

        return df

    except Exception as e:
        print(f"获取历史行情数据时出错：{e}")
        return pd.DataFrame()


def prices_to_df(prices):
    """Convert price data to DataFrame with standardized column names"""
    try:
        df = pd.DataFrame(prices)

        # 标准化列名映射
        column_mapping = {
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }

        # 重命名列
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # 确保必要的列存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # 使用0填充缺失的必要列

        return df
    except Exception as e:
        print(f"转换价格数据时出错: {str(e)}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取股票价格数据

    Args:
        ticker: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含价格数据的DataFrame
    """
    return get_price_history(ticker, start_date, end_date)
