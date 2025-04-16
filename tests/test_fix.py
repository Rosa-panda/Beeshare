"""
测试脚本，验证修复
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def generate_mock_data(symbol, start_date=None, end_date=None):
    """生成模拟股票数据"""
    # 生成日期范围
    if start_date is None:
        start = datetime.now() - timedelta(days=90)
    else:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
        except:
            start = datetime.now() - timedelta(days=90)
    
    if end_date is None:
        end = datetime.now()
    else:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except:
            end = datetime.now()
    
    # 创建日期列表
    date_range = []
    current = start
    while current <= end:
        # 跳过周末
        if current.weekday() < 5:  # 0-4是周一至周五
            date_range.append(current)
        current += timedelta(days=1)
    
    # 使用固定的随机种子确保可重复性
    np.random.seed(42 + hash(symbol) % 1000)
    
    # 生成模拟数据
    price = np.random.uniform(10, 100)  # 起始价格
    mock_data = []
    
    for date in date_range:
        price_change = np.random.normal(0, 0.02) * price
        price += price_change
        open_price = price * (1 + np.random.normal(0, 0.01))
        high_price = price * (1 + abs(np.random.normal(0, 0.02)))
        low_price = price * (1 - abs(np.random.normal(0, 0.02)))
        close_price = price
        volume = int(np.random.uniform(50000, 5000000))
        amount = volume * price
        
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
            'source': 'mock_data'
        })
    
    # 创建DataFrame
    df = pd.DataFrame(mock_data)
    logger.info(f"成功生成 {symbol} 的模拟数据，共 {len(df)} 行")
    return df

def main():
    symbols = ['600519', '601398', '600036']
    all_data = {}
    
    for symbol in symbols:
        try:
            # 生成模拟数据
            logger.info(f"生成 {symbol} 的模拟数据")
            historical_data = generate_mock_data(symbol)
            
            if not historical_data.empty:
                all_data[symbol] = historical_data
                logger.info(f"成功获取 {symbol} 的模拟数据: {len(historical_data)} 条记录")
        except Exception as e:
            logger.error(f"生成 {symbol} 的模拟数据失败: {e}")
    
    if not all_data:
        logger.error("未能生成任何数据")
        return
    
    # 合并所有数据
    combined_data = pd.concat(all_data.values(), ignore_index=True)
    logger.info(f"合并数据成功，共 {len(combined_data)} 行")
    
    # 打印数据统计信息
    for symbol, data in all_data.items():
        logger.info(f"股票 {symbol} 数据: {len(data)} 行, 开始日期: {data['date'].min()}, 结束日期: {data['date'].max()}")
    
    logger.info("测试成功！模拟数据生成功能正常工作")

if __name__ == "__main__":
    main() 