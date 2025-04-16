"""
测试聚类分析功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入聚类分析相关模块
from src.analysis.clustering import ClusteringAnalyzer

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def generate_mock_data(symbols, days=90):
    """生成多个股票的模拟数据"""
    all_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        # 生成单个股票的模拟数据
        mock_data = generate_single_stock_data(symbol, start_date, end_date)
        all_data[symbol] = mock_data
        
    # 合并所有数据
    combined_data = pd.concat(all_data.values(), ignore_index=True)
    logger.info(f"成功生成模拟数据，共 {len(combined_data)} 行")
    
    return combined_data

def generate_single_stock_data(symbol, start_date, end_date):
    """生成单个股票的模拟数据"""
    # 创建日期列表
    date_range = []
    current = start_date
    while current <= end_date:
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

def run_clustering_test():
    """运行聚类分析测试"""
    try:
        # 生成模拟数据
        symbols = ['600519', '601398', '600036']
        data = generate_mock_data(symbols)
        
        # 创建聚类分析器
        analyzer = ClusteringAnalyzer(data)
        
        # 执行聚类分析
        n_clusters = 2
        logger.info(f"使用 {n_clusters} 个聚类运行KMeans聚类")
        
        clustering_result = analyzer.kmeans_clustering(
            params={
                'n_clusters': n_clusters,
                'features': ['open', 'close', 'high', 'low', 'volume', 'change_pct']
            }
        )
        
        # 输出聚类结果
        if clustering_result:
            logger.info("聚类分析成功完成！")
            
            # 打印聚类统计信息
            labels = clustering_result['labels']
            for i in range(n_clusters):
                cluster_count = (labels == i).sum()
                cluster_percent = cluster_count / len(labels) * 100
                logger.info(f"聚类 #{i+1}: {cluster_count} 个样本 ({cluster_percent:.2f}%)")
            
            if 'silhouette_score' in clustering_result:
                logger.info(f"轮廓分数: {clustering_result['silhouette_score']:.4f}")
        else:
            logger.error("聚类分析失败")
            
        return True
    except Exception as e:
        logger.error(f"测试聚类分析时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if run_clustering_test():
        logger.info("聚类分析测试成功!")
    else:
        logger.error("聚类分析测试失败!") 