from datetime import datetime, timedelta
import json
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
from src.tools.api import get_price_data
from src.main import run_hedge_fund
import sys
import matplotlib
import os

# 根据操作系统配置中文字体
if sys.platform.startswith('win'):
    # Windows系统
    matplotlib.rc('font', family='Microsoft YaHei')
elif sys.platform.startswith('linux'):
    # Linux系统
    matplotlib.rc('font', family='WenQuanYi Micro Hei')
else:
    # macOS系统
    matplotlib.rc('font', family='PingFang SC')

# 用来正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False


class Backtester:
    def __init__(self, agent, ticker, start_date, end_date, initial_capital, num_of_news):
        self.agent = agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.portfolio = {"cash": initial_capital, "stock": 0}
        self.portfolio_values = []
        self.num_of_news = num_of_news
        # 设置回测日志
        self.setup_backtest_logging()
        self.logger = self.setup_logging()

        # 初始化 API 调用管理
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0

        # 验证输入参数
        self.validate_inputs()

    def setup_logging(self):
        """设置日志记录器"""
        logger = logging.getLogger('backtester')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def validate_inputs(self):
        """验证输入参数的有效性"""
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("开始日期必须早于结束日期")
            if self.initial_capital <= 0:
                raise ValueError("初始资金必须大于0")
            if not isinstance(self.ticker, str) or len(self.ticker) != 6:
                raise ValueError("无效的股票代码格式")
            self.logger.info("输入参数验证通过")
        except Exception as e:
            self.logger.error(f"输入参数验证失败: {str(e)}")
            raise

    def get_agent_decision(self, current_date, lookback_start, portfolio):
        """获取智能体决策，包含 API 限制处理"""
        max_retries = 3

        # 检查并重置 API 时间窗口
        current_time = time.time()
        if current_time - self._api_window_start >= 60:
            self._api_call_count = 0
            self._api_window_start = current_time
            self.logger.debug("API 调用计数已重置")

        # 如果达到 API 限制，等待新的时间窗口
        if self._api_call_count >= 8:  # 预留余量
            wait_time = 60 - (current_time - self._api_window_start)
            if wait_time > 0:
                self.logger.info(f"已达到 API 限制，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                self._api_call_count = 0
                self._api_window_start = time.time()

        for attempt in range(max_retries):
            try:
                # 确保调用间隔至少 6 秒
                if self._last_api_call:
                    time_since_last_call = time.time() - self._last_api_call
                    if time_since_last_call < 6:
                        sleep_time = 6 - time_since_last_call
                        time.sleep(sleep_time)

                # 更新调用时间和计数
                self._last_api_call = time.time()
                self._api_call_count += 1

                # 调用智能体并解析结果
                result = self.agent(
                    ticker=self.ticker,
                    start_date=lookback_start,
                    end_date=current_date,
                    portfolio=portfolio,
                    num_of_news=self.num_of_news
                )

                try:
                    # 尝试解析返回的字符串为 JSON
                    if isinstance(result, str):
                        # 清理可能的markdown标记
                        result = result.replace(
                            '```json\n', '').replace('\n```', '').strip()
                        print(f"---------------result------------\n: {result}")
                        parsed_result = json.loads(result)

                        # 构建标准格式的结果
                        formatted_result = {
                            "decision": parsed_result,  # 保持原始决策结构
                            "analyst_signals": {}
                        }

                        # 处理智能体信号
                        if "agent_signals" in parsed_result:
                            formatted_result["analyst_signals"] = {
                                signal["agent"]: {
                                    "signal": signal.get("signal", "unknown"),
                                    "confidence": signal.get("confidence", 0)
                                }
                                for signal in parsed_result["agent_signals"]
                            }

                        self.logger.info(
                            f"解析后的决策: {formatted_result['decision']}")  # 添加日志
                        return formatted_result
                    return result
                except json.JSONDecodeError as e:
                    # 如果无法解析为 JSON，记录错误并返回默认决策
                    self.logger.warning(f"JSON解析错误: {str(e)}")
                    self.logger.warning(f"原始返回结果: {result}")
                    return {
                        "decision": {"action": "hold", "quantity": 0},
                        "analyst_signals": {}
                    }

            except Exception as e:
                if "AFC is enabled" in str(e):
                    self.logger.warning(f"触发 AFC 限制，等待 60 秒后重试...")
                    time.sleep(60)
                    self._api_call_count = 0
                    self._api_window_start = time.time()
                    continue

                self.logger.warning(
                    f"获取智能体决策失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {"decision": {"action": "hold", "quantity": 0}, "analyst_signals": {}}
                time.sleep(2 ** attempt)

    def parse_decision_from_text(self, text):
        """从文本中解析交易决策"""
        text = text.lower()

        # 默认决策
        decision = {"action": "hold", "quantity": 0}

        # 检查是否包含决策关键词
        if "buy" in text or "bullish" in text:
            decision["action"] = "buy"
            decision["quantity"] = 100  # 默认购买数量
        elif "sell" in text or "bearish" in text:
            decision["action"] = "sell"
            decision["quantity"] = 100  # 默认卖出数量

        return decision

    def execute_trade(self, action, quantity, current_price):
        """执行交易，验证组合约束"""
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # 计算最大可买数量
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.portfolio["stock"])
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0
        return 0

    def setup_backtest_logging(self):
        """设置回测日志"""
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # 创建回测日志记录器
        self.backtest_logger = logging.getLogger('backtest')
        self.backtest_logger.setLevel(logging.INFO)

        # 清除已存在的处理器
        if self.backtest_logger.handlers:
            self.backtest_logger.handlers.clear()

        # 设置文件处理器
        current_date = datetime.now().strftime('%Y%m%d')
        backtest_period = f"{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}"
        log_file = os.path.join(
            log_dir, f"backtest_{self.ticker}_{current_date}_{backtest_period}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('%(message)s')  # 简化格式，只显示消息
        file_handler.setFormatter(formatter)

        # 添加处理器
        self.backtest_logger.addHandler(file_handler)

        # 写入回测初始信息
        self.backtest_logger.info(
            f"回测开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.backtest_logger.info(f"股票代码: {self.ticker}")
        self.backtest_logger.info(f"回测区间: {self.start_date} 至 {self.end_date}")
        self.backtest_logger.info(f"初始资金: {self.initial_capital:,.2f}\n")
        self.backtest_logger.info("-" * 100)

    def run_backtest(self):
        """运行回测"""
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        self.logger.info("\n开始回测...")
        print(f"{'日期':<12} {'代码':<6} {'操作':<6} {'数量':>8} {'价格':>8} {'现金':>12} {'持仓':>8} {'总值':>12} {'看多':>8} {'看空':>8} {'中性':>8}")
        print("-" * 110)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)
                              ).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            # 获取智能体决策
            output = self.get_agent_decision(
                current_date_str, lookback_start, self.portfolio)

            # 记录每个智能体的信号和分析结果
            self.backtest_logger.info(f"\n交易日期: {current_date_str}")
            if "analyst_signals" in output:
                self.backtest_logger.info("\n各智能体分析结果:")
                for agent_name, signal in output["analyst_signals"].items():
                    self.backtest_logger.info(f"\n{agent_name}:")

                    # 记录信号和置信度
                    signal_str = f"- 信号: {signal.get('signal', 'unknown')}"
                    if 'confidence' in signal:
                        signal_str += f", 置信度: {signal.get('confidence', 0)*100:.0f}%"
                    self.backtest_logger.info(signal_str)

                    # 记录分析结果
                    if 'analysis' in signal:
                        self.backtest_logger.info("- 分析结果:")
                        analysis = signal['analysis']
                        if isinstance(analysis, dict):
                            for key, value in analysis.items():
                                self.backtest_logger.info(f"  {key}: {value}")
                        elif isinstance(analysis, list):
                            for item in analysis:
                                self.backtest_logger.info(f"  • {item}")
                        else:
                            self.backtest_logger.info(f"  {analysis}")

                    # 记录理由
                    if 'reason' in signal:
                        self.backtest_logger.info("- 决策理由:")
                        reason = signal['reason']
                        if isinstance(reason, list):
                            for item in reason:
                                self.backtest_logger.info(f"  • {item}")
                        else:
                            self.backtest_logger.info(f"  • {reason}")

                    # 记录其他可能的指标
                    for key, value in signal.items():
                        if key not in ['signal', 'confidence', 'analysis', 'reason']:
                            self.backtest_logger.info(f"- {key}: {value}")

                self.backtest_logger.info("\n综合决策:")

            agent_decision = output.get(
                "decision", {"action": "hold", "quantity": 0})
            action, quantity = agent_decision.get(
                "action", "hold"), agent_decision.get("quantity", 0)

            # 记录决策详情
            self.backtest_logger.info(f"行动: {action.upper()}")
            self.backtest_logger.info(f"数量: {quantity}")
            if "reason" in agent_decision:
                self.backtest_logger.info(f"决策理由: {agent_decision['reason']}")

            # 获取当前价格并执行交易
            df = get_price_data(self.ticker, lookback_start, current_date_str)
            if df is None or df.empty:
                continue

            current_price = df.iloc[-1]['close']
            executed_quantity = self.execute_trade(
                action, quantity, current_price)

            # 更新组合总值
            total_value = self.portfolio["cash"] + \
                self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value

            # 计算当日收益率
            if len(self.portfolio_values) > 0:
                daily_return = (
                    total_value / self.portfolio_values[-1]["Portfolio Value"] - 1) * 100
            else:
                daily_return = 0

            # 记录组合价值和收益率
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Daily Return": daily_return
            })

    def analyze_performance(self):
        """分析回测性能"""
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # 计算累计收益率
        performance_df["Cumulative Return"] = (
            performance_df["Portfolio Value"] / self.initial_capital - 1) * 100

        # 将金额转换为千元
        performance_df["Portfolio Value (K)"] = performance_df["Portfolio Value"] / 1000

        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), height_ratios=[1, 1])
        fig.suptitle("回测结果分析", fontsize=12)

        # 绘制资金变化图
        line1 = ax1.plot(performance_df.index,
                         performance_df["Portfolio Value (K)"], label="组合价值", marker='o')
        ax1.set_ylabel("组合价值 (千元)")
        ax1.set_title("组合价值变化")
        ax1.grid(True)

        # 在数据点上添加标签
        for x, y in zip(performance_df.index, performance_df["Portfolio Value (K)"]):
            ax1.annotate(f'{y:.1f}K',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=8)

        # 绘制收益率变化图
        line2 = ax2.plot(performance_df.index,
                         performance_df["Cumulative Return"], label="累计收益率", color='green', marker='o')
        ax2.set_ylabel("累计收益率 (%)")
        ax2.set_title("累计收益率变化")
        ax2.grid(True)

        # 在数据点上添加标签
        for x, y in zip(performance_df.index, performance_df["Cumulative Return"]):
            ax2.annotate(f'{y:.2f}%',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=8)

        # 设置x轴标签
        plt.xlabel("日期")

        # 自动调整布局以防止标签重叠
        plt.tight_layout()

        # 显示图表
        plt.show()

        # 计算和打印性能指标
        total_return = (
            self.portfolio["portfolio_value"] - self.initial_capital) / self.initial_capital
        print(f"\n总收益率: {total_return * 100:.2f}%")

        # 记录最终回测结果
        self.backtest_logger.info("\n" + "=" * 50)
        self.backtest_logger.info("回测结果汇总")
        self.backtest_logger.info("=" * 50)
        self.backtest_logger.info(f"初始资金: {self.initial_capital:,.2f}")
        self.backtest_logger.info(
            f"最终总值: {self.portfolio['portfolio_value']:,.2f}")
        self.backtest_logger.info(f"总收益率: {total_return * 100:.2f}%")

        # 计算夏普比率
        daily_returns = performance_df["Daily Return"] / 100  # 转换为小数
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * \
            (252 ** 0.5) if std_daily_return != 0 else 0
        # print(f"夏普比率: {sharpe_ratio:.2f}")
        self.backtest_logger.info(f"夏普比率: {sharpe_ratio:.2f}")

        # 计算最大回撤
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = (performance_df["Portfolio Value"] / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        # print(f"最大回撤: {max_drawdown:.2f}%")
        self.backtest_logger.info(f"最大回撤: {max_drawdown:.2f}%")

        return performance_df


if __name__ == "__main__":
    import argparse

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='运行回测模拟')
    parser.add_argument('--ticker', type=str, required=True,
                        help='股票代码 (例如: 600519)')
    parser.add_argument('--end-date', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'), help='结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--start-date', type=str, default=(datetime.now() -
                        timedelta(days=90)).strftime('%Y-%m-%d'), help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--initial-capital', type=float,
                        default=100000, help='初始资金 (默认: 100000)')
    parser.add_argument('--num-of-news', type=int, default=5,
                        help='Number of news articles to analyze for sentiment (default: 5)')

    args = parser.parse_args()

    # 创建回测器实例
    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        num_of_news=args.num_of_news
    )

    # 运行回测
    backtester.run_backtest()

    # 分析性能
    performance_df = backtester.analyze_performance()
