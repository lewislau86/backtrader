#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的Backtrader回测脚本
使用data文件夹下的daily_price.csv数据
包含高效的数据处理和多核支持
"""

import sys
import subprocess
import pkg_resources

# NumPy版本兼容性检查和修复
def check_and_fix_dependencies():
    """检查并修复NumPy版本兼容性问题"""
    try:
        import numpy
        numpy_version = pkg_resources.get_distribution("numpy").version
        print(f"当前NumPy版本: {numpy_version}")
        
        if numpy_version.startswith("2."):
            print("检测到NumPy 2.x，检查兼容性...")
            
            # 检查bottleneck是否安装
            try:
                import bottleneck
                print("检测到bottleneck模块可能与NumPy 2.x不兼容")
                print("正在尝试修复兼容性问题...")
                
                # 尝试重新安装bottleneck
                print("正在重新安装bottleneck以兼容NumPy 2.x...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "bottleneck"])
                print("bottleneck已更新")
                
            except ImportError:
                print("未安装bottleneck模块，跳过")
            except Exception as e:
                print(f"更新bottleneck失败: {e}")
                print("建议手动执行: pip install --upgrade bottleneck")
                print("或者降级NumPy: pip install numpy==1.24.3")
                sys.exit(1)
                
            # 检查其他可能的问题模块
            problematic_modules = ["pandas", "matplotlib", "scipy"]
            for module in problematic_modules:
                try:
                    __import__(module)
                    print(f"检查{module}...")
                except Exception as e:
                    print(f"{module}可能与NumPy 2.x不兼容，尝试更新...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", module])
                        print(f"{module}已更新")
                    except Exception as e:
                        print(f"更新{module}失败: {e}")
                        print(f"建议手动执行: pip install --upgrade {module}")
            
            print("依赖检查完成")
    except Exception as e:
        print(f"依赖检查失败: {e}")

# 运行依赖检查
check_and_fix_dependencies()

# 导入必要的库
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import time
import os
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个简单的均线交叉策略
class SmaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 5),    # 短期均线周期
        ('slow_period', 20),   # 长期均线周期
        ('verbose', False),    # 是否打印详细日志
    )

    def log(self, txt, dt=None):
        """日志函数"""
        if self.params.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        """初始化策略"""
        # 初始化指标
        self.fast_sma = {}
        self.slow_sma = {}

        # 为每个数据源创建指标
        for data in self.datas:
            # 获取股票代码
            ticker = data._name

            # 创建均线指标
            self.fast_sma[ticker] = bt.indicators.SMA(
                data.close,
                period=self.params.fast_period,
                plotname=f'SMA{self.params.fast_period}'
            )
            self.slow_sma[ticker] = bt.indicators.SMA(
                data.close,
                period=self.params.slow_period,
                plotname=f'SMA{self.params.slow_period}'
            )

            # 创建交叉信号
            bt.indicators.CrossOver(
                self.fast_sma[ticker],
                self.slow_sma[ticker],
                plotname=f'交叉信号'
            )

        # 记录订单
        self.orders = {}
        for data in self.datas:
            self.orders[data._name] = []

    def notify_order(self, order):
        """订单状态更新通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 获取股票代码
        ticker = order.data._name

        # 订单完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入 {ticker}: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
            else:
                self.log(f'卖出 {ticker}: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')

        # 订单取消/拒绝/保证金不足
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单被取消/拒绝/保证金不足: {order.status}')

        # 从未决订单列表中移除
        self.orders[ticker].remove(order)

    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return

        ticker = trade.data._name
        self.log(f'{ticker} 交易利润: 毛利润={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}')

    def next(self):
        """每个bar执行一次"""
        # 遍历所有数据源
        for data in self.datas:
            ticker = data._name

            # 如果没有持仓且快线上穿慢线，买入
            if not self.getposition(data).size and self.fast_sma[ticker][0] > self.slow_sma[ticker][0] and self.fast_sma[ticker][-1] <= self.slow_sma[ticker][-1]:
                # 计算买入数量 - 使用10%的资金买入
                size = int(self.broker.getcash() * 0.1 / data.close[0])
                if size > 0:
                    self.log(f'买入信号: {ticker}, 价格={data.close[0]:.2f}')
                    order = self.buy(data=data, size=size)
                    self.orders[ticker].append(order)

            # 如果有持仓且快线下穿慢线，卖出
            elif self.getposition(data).size > 0 and self.fast_sma[ticker][0] < self.slow_sma[ticker][0] and self.fast_sma[ticker][-1] >= self.slow_sma[ticker][-1]:
                self.log(f'卖出信号: {ticker}, 价格={data.close[0]:.2f}')
                order = self.sell(data=data, size=self.getposition(data).size)
                self.orders[ticker].append(order)


def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()

    # 检查数据文件是否存在
    data_file = "./data/daily_price.csv"
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 {data_file}")
        return

    # 读取数据
    print("读取数据...")
    daily_price = pd.read_csv(data_file, parse_dates=['datetime'])
    daily_price = daily_price.set_index(['datetime'])

    # 打印数据信息
    print(f"数据范围: {daily_price.index.min()} 到 {daily_price.index.max()}")
    print(f"股票数量: {len(daily_price['sec_code'].unique())}")
    print(f"总记录数: {len(daily_price)}")

    # 创建cerebro实例，启用多核
    cerebro = bt.Cerebro(maxcpus=16)

    # 设置初始资金
    initial_cash = 1000000.0
    cerebro.broker.setcash(initial_cash)

    # 设置佣金和滑点
    cerebro.broker.setcommission(commission=0.0003)  # 0.03%
    cerebro.broker.set_slippage_perc(perc=0.0001)    # 0.01%

    # 预处理数据
    print("预处理数据...")
    unique_dates = daily_price.index.unique()
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest']

    # 限制处理的股票数量，以加快测试速度
    stock_limit = 10  # 可以根据需要调整
    stocks = daily_price['sec_code'].unique()[:stock_limit]

    # 添加数据
    print(f"添加{len(stocks)}只股票的数据...")
    for i, stock in enumerate(stocks):
        # 提取数据并一次性完成所有处理
        df = daily_price.query(f"sec_code=='{stock}'")[required_columns]
        data_ = pd.DataFrame(index=unique_dates)
        data_ = pd.merge(data_, df, left_index=True, right_index=True, how='left')

        # 使用更高效的数据处理方式
        data_ = data_.assign(
            volume=data_['volume'].fillna(0),
            openinterest=data_['openinterest'].fillna(0),
            **data_[['open', 'high', 'low', 'close']].ffill().fillna(0)
        )

        # 创建数据源并添加到cerebro
        datafeed = bt.feeds.PandasData(
            dataname=data_,
            fromdate=datetime.datetime(2019, 1, 2),
            todate=datetime.datetime(2021, 1, 28)
        )
        cerebro.adddata(datafeed, name=stock)

        # 显示进度
        if (i+1) % 5 == 0 or i+1 == len(stocks):
            print(f"已添加 {i+1}/{len(stocks)} 只股票")

    # 添加策略
    cerebro.addstrategy(
        SmaCrossStrategy,
        verbose=False  # 设置为True可以查看详细日志
    )

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # 添加观测器
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Trades)

    # 运行回测
    print("\n开始回测...")
    run_start = time.time()
    results = cerebro.run()
    run_end = time.time()
    print(f"回测完成，耗时: {run_end - run_start:.2f} 秒")

    # 获取回测结果
    strat = results[0]

    # 打印分析结果
    print("\n==== 回测结果 ====")
    print(f"初始资金: {initial_cash:.2f}")
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    print(f"总收益率: {(cerebro.broker.getvalue() / initial_cash - 1) * 100:.2f}%")

    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"夏普比率: {sharpe.get('sharperatio', 0.0):.3f}")

    # 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"最大回撤: {drawdown.get('max', {}).get('drawdown', 0.0):.2f}%")
    print(f"最长回撤天数: {drawdown.get('max', {}).get('len', 0)}")

    # 交易统计
    trades = strat.analyzers.trades.get_analysis()
    if trades.get('total', {}).get('total', 0) > 0:
        print(f"总交易次数: {trades.get('total', {}).get('total', 0)}")
        print(f"盈利交易: {trades.get('won', {}).get('total', 0)}")
        print(f"亏损交易: {trades.get('lost', {}).get('total', 0)}")
        if trades.get('won', {}).get('total', 0) > 0:
            print(f"平均盈利: {trades.get('won', {}).get('pnl', {}).get('average', 0.0):.2f}")
        if trades.get('lost', {}).get('total', 0) > 0:
            print(f"平均亏损: {trades.get('lost', {}).get('pnl', {}).get('average', 0.0):.2f}")

    # 绘制结果
    print("\n生成回测图表...")
    cerebro.plot(style='candle', barup='red', bardown='green', volume=False, grid=True)

    # 总耗时
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
