"""
基本功能测试脚本

用于测试数据源和存储功能
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.data_sources import AKShareDS
from src.storage import CSVStorage
from config.config import DATA_SOURCES, STORAGE_CONFIG

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_akshare():
    """
    测试AKShare数据源
    """
    logger.info("测试AKShare数据源")
    
    # 初始化数据源
    akshare = AKShareDS(DATA_SOURCES.get('akshare', {}))
    
    if not akshare.is_ready:
        logger.error("AKShare数据源初始化失败")
        return False
    
    # 测试股票列表 (A股)
    test_symbols = ['600036', '601398', '600519']
    
    # 测试历史数据获取
    logger.info("测试获取历史数据")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in test_symbols:
        logger.info(f"获取 {symbol} 的历史数据")
        
        df = akshare.get_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='daily'
        )
        
        if df.empty:
            logger.warning(f"未找到 {symbol} 的历史数据")
        else:
            logger.info(f"成功获取 {len(df)} 条 {symbol} 的历史数据")
            # 打印前5条数据
            logger.info(f"数据预览:\n{df.head()}")
    
    # 测试实时数据获取
    logger.info("测试获取实时数据")
    
    df = akshare.get_realtime_data(test_symbols)
    
    if df.empty:
        logger.warning("未找到实时数据")
    else:
        logger.info(f"成功获取 {len(df)} 条实时数据")
        # 打印数据
        logger.info(f"数据预览:\n{df}")
    
    # 测试股票搜索
    logger.info("测试搜索股票")
    
    search_keywords = ['银行', '科技', '医药']
    
    for keyword in search_keywords:
        logger.info(f"搜索关键词 '{keyword}'")
        
        results = akshare.search_symbols(keyword)
        
        if not results:
            logger.warning(f"未找到匹配 '{keyword}' 的股票")
        else:
            logger.info(f"找到 {len(results)} 个匹配结果")
            # 打印前5个结果
            logger.info(f"搜索结果预览:\n{results[:5]}")

    # 测试指数数据获取
    logger.info("测试获取指数数据")
    
    index_symbols = ['000001', '399001', '000300']
    
    for symbol in index_symbols:
        logger.info(f"获取指数 {symbol} 的历史数据")
        
        df = akshare.get_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='daily'
        )
        
        if df.empty:
            logger.warning(f"未找到指数 {symbol} 的历史数据")
        else:
            logger.info(f"成功获取 {len(df)} 条指数 {symbol} 的历史数据")
            # 打印前3条数据
            logger.info(f"数据预览:\n{df.head(3)}")
    
    return True

def test_csv_storage():
    """
    测试CSV存储
    """
    logger.info("测试CSV存储")
    
    # 初始化存储
    csv_storage = CSVStorage(STORAGE_CONFIG.get('csv', {}))
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'symbol': ['TEST'] * 10,
        'open': [100 + i for i in range(10)],
        'high': [110 + i for i in range(10)],
        'low': [90 + i for i in range(10)],
        'close': [105 + i for i in range(10)],
        'volume': [1000000 + i * 10000 for i in range(10)]
    })
    
    # 测试保存数据
    logger.info("测试保存数据")
    
    success = csv_storage.save_data(
        data=test_data,
        data_type='test',
        symbol='TEST',
        mode='overwrite'
    )
    
    if not success:
        logger.error("保存数据失败")
        return False
    
    logger.info("成功保存数据")
    
    # 测试加载数据
    logger.info("测试加载数据")
    
    loaded_data = csv_storage.load_data(
        data_type='test',
        symbol='TEST'
    )
    
    if loaded_data.empty:
        logger.error("加载数据失败")
        return False
    
    logger.info(f"成功加载 {len(loaded_data)} 条数据")
    logger.info(f"数据预览:\n{loaded_data.head()}")
    
    # 测试部分日期加载
    logger.info("测试部分日期加载")
    
    partial_data = csv_storage.load_data(
        data_type='test',
        symbol='TEST',
        start_date='2023-01-03',
        end_date='2023-01-07'
    )
    
    if partial_data.empty:
        logger.error("加载部分数据失败")
    else:
        logger.info(f"成功加载 {len(partial_data)} 条部分数据")
        logger.info(f"部分数据预览:\n{partial_data}")
    
    # 测试数据删除
    logger.info("测试数据删除")
    
    success = csv_storage.delete_data(
        data_type='test',
        symbol='TEST'
    )
    
    if not success:
        logger.error("删除数据失败")
    else:
        logger.info("成功删除数据")
        
        # 验证数据已删除
        exists = csv_storage.exists(
            data_type='test',
            symbol='TEST'
        )
        
        if exists:
            logger.error("数据删除验证失败，文件仍然存在")
        else:
            logger.info("数据删除验证成功，文件已不存在")
    
    return True

def run_tests():
    """
    运行所有测试
    """
    logger.info("开始运行测试")
    
    # 测试数据源
    akshare_test = test_akshare()
    
    # 测试存储
    csv_test = test_csv_storage()
    
    # 输出测试结果
    logger.info("测试结果:")
    logger.info(f"AKShare测试: {'通过' if akshare_test else '失败'}")
    logger.info(f"CSV存储测试: {'通过' if csv_test else '失败'}")
    
    return akshare_test and csv_test

if __name__ == "__main__":
    try:
        success = run_tests()
        if success:
            logger.info("所有测试通过")
            sys.exit(0)
        else:
            logger.error("部分测试失败")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}")
        sys.exit(2) 