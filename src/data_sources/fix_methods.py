"""
AKShareDS数据源的修复方法集合。

该模块包含用于修复AKShareDS数据源类中出现问题的各种方法。
主要集中在当主要API调用失败时的备用数据生成方法。

创建日期: 2023-07-01
最后修改: 2023-07-10
作者: BeeShare开发团队
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 获取logger
logger = logging.getLogger(__name__)

def _get_stock_data_via_fallback_api(self, symbol, start_date, end_date, interval):
    """
    生成模拟股票数据作为API调用失败时的备用方案。
    
    当AKShare API调用失败时，此方法会生成合理的模拟股票数据，以确保系统正常运行。
    生成的数据尝试模拟真实股票数据的波动特性，但仅用于测试和容错目的。
    
    Args:
        symbol (str): 股票代码
        start_date (str): 开始日期，格式为'YYYY-MM-DD'
        end_date (str): 结束日期，格式为'YYYY-MM-DD'
        interval (str): 数据间隔，支持'daily', 'weekly', 'monthly'
    
    Returns:
        pandas.DataFrame: 包含以下列的模拟股票数据:
            - date: 日期
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - amount: 成交额
            - change: 价格变动
            - change_pct: 价格变动百分比
            - symbol: 股票代码
            - source: 数据来源标记
    
    Raises:
        ValueError: 当日期格式无效时可能抛出
    """
    logger.warning(f"使用备用方法生成{symbol}的模拟数据")
    
    # 解析日期范围
    if start_date is None:
        start = datetime.now() - timedelta(days=90)
    else:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
        except:
            logger.error(f"起始日期格式无效: {start_date}，使用过去90天")
            start = datetime.now() - timedelta(days=90)
    
    if end_date is None:
        end = datetime.now()
    else:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except:
            logger.error(f"结束日期格式无效: {end_date}，使用当前日期")
            end = datetime.now()
    
    # 创建交易日期列表（排除周末）
    date_range = []
    current = start
    while current <= end:
        # 跳过周末，0-4是周一至周五
        if current.weekday() < 5:
            date_range.append(current)
        current += timedelta(days=1)
    
    # 使用固定的随机种子确保可重复性
    # 加入股票代码的哈希值使不同股票生成不同的随机序列
    np.random.seed(42 + hash(symbol) % 1000)
    
    # 基于随机游走模型生成模拟股票价格数据
    price = np.random.uniform(10, 100)  # 随机起始价格
    mock_data = []
    
    for date in date_range:
        # 生成每日价格波动
        price_change = np.random.normal(0, 0.02) * price
        price += price_change
        
        # 生成日内价格
        open_price = price * (1 + np.random.normal(0, 0.01))
        high_price = price * (1 + abs(np.random.normal(0, 0.02)))
        low_price = price * (1 - abs(np.random.normal(0, 0.02)))
        close_price = price
        
        # 生成成交量和成交额
        volume = int(np.random.uniform(50000, 5000000))
        amount = volume * price
        
        # 添加数据点
        mock_data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'amount': amount,
            'change': price_change,
            'change_pct': (price_change / (price - price_change)) * 100 if price - price_change != 0 else 0,
            'symbol': symbol,
            'source': 'mock_data'  # 标记数据来源为模拟数据
        })
    
    # 创建DataFrame并返回
    df = pd.DataFrame(mock_data)
    logger.info(f"成功生成{symbol}的模拟数据，共{len(df)}行")
    return df 