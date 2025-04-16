#!/usr/bin/env python
"""
测试AKShareDS类中的_get_stock_data_via_fallback_api方法的功能。

该测试脚本验证了当主要API失效时，备用API的生成模拟数据能力。
测试包括验证生成的数据是否包含所有必要的列以及数据结构是否正确。

创建日期: 2023-06-30
最后修改: 2023-07-06
作者: BeeShare开发团队
"""
import sys
import os
import pandas as pd
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def test_fallback_api():
    """
    测试AKShareDS类中的_get_stock_data_via_fallback_api方法。
    
    该函数检查备用API是否能正确生成模拟数据，并验证数据的结构和列名。
    
    Returns:
        bool: 测试通过返回True，否则返回False
    
    Raises:
        ImportError: 当无法导入必要的模块时抛出
        Exception: 测试过程中出现的其他错误
    """
    try:
        # 导入相关模块
        from config.config import DATA_SOURCES
        from src.data_sources.akshare import AKShareDS
        
        # 创建AKShareDS实例
        logger.info("初始化AKShare数据源...")
        akshare_config = DATA_SOURCES.get('akshare', {})
        ds = AKShareDS(akshare_config)
        
        # 测试fallback方法，直接调用内部方法
        logger.info("直接调用_get_stock_data_via_fallback_api方法生成模拟数据...")
        df = ds._get_stock_data_via_fallback_api('600519', '2023-01-01', '2023-01-10', 'daily')
        
        if df is not None and not df.empty:
            logger.info(f"成功获取数据，行数: {len(df)}")
            logger.info(f"数据列: {df.columns.tolist()}")
            logger.info(f"数据前3行:\n{df.head(3)}")
            
            # 检查必要的列是否存在
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'change_pct']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                logger.error(f"缺少必要的列: {missing}")
                return False
            else:
                logger.info("包含所有必要的列！")
                
            # 测试成功
            print("\n测试结果: 成功！模拟数据生成并标准化正常工作")
            return True
        else:
            logger.error("获取数据失败，返回了空DataFrame")
            print("\n测试结果: 失败！返回了空DataFrame")
            return False
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n测试结果: 失败！出现异常")
        return False

if __name__ == "__main__":
    test_fallback_api() 