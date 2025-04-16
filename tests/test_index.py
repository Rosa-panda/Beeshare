#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试获取A股指数数据
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# 获取项目根目录
def get_project_root():
    """获取项目根目录的绝对路径"""
    # 当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(current_file))
    return root_dir

# 添加项目根目录到系统路径
project_root = get_project_root()
sys.path.append(project_root)

# 创建指数数据保存目录
index_data_dir = os.path.join(project_root, "data", "index_data")
os.makedirs(index_data_dir, exist_ok=True)

# 导入自定义模块
from src.data_sources import AKShareDS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_get_index_data():
    """测试获取A股指数数据"""
    # 初始化AKShare数据源
    akshare_ds = AKShareDS()
    
    # 检查连接状态
    if not akshare_ds.is_ready:
        logger.error("AKShare数据源未就绪，请先检查连接")
        return False
    
    # 指数代码列表
    index_symbols = [
        '000001',  # 上证指数
        '399001',  # 深证成指
        '000300',  # 沪深300
        '000016',  # 上证50
        '000905',  # 中证500
    ]
    
    # 设置时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 获取近30天数据
    
    for symbol in index_symbols:
        logger.info(f"获取指数 {symbol} 的历史数据...")
        
        df = akshare_ds.get_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='daily'
        )
        
        if df.empty:
            logger.warning(f"未找到指数 {symbol} 的历史数据")
            continue
        
        logger.info(f"成功获取 {len(df)} 条 {symbol} 的历史数据")
        
        # 显示数据预览
        logger.info(f"数据预览:\n{df.head()}")
        
        # 将数据保存到指定CSV文件中
        today_str = datetime.now().strftime('%Y%m%d')
        file_name = f"index_{symbol}_{today_str}.csv"
        file_path = os.path.join(index_data_dir, file_name)
        df.to_csv(file_path, index=False)
        logger.info(f"数据已保存到 {file_path}")
        
    logger.info("指数数据获取测试完成")
    return True

if __name__ == "__main__":
    try:
        test_get_index_data()
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}")
        sys.exit(1) 