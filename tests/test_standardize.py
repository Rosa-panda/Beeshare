#!/usr/bin/env python
"""
测试列名标准化功能模块。

该脚本专门测试standardize_columns函数，确保其能够正确地将各种数据源的
原始列名映射为标准化的列名格式。

创建日期: 2023-06-30
最后修改: 2023-07-05
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

def test_standardize_columns():
    """
    测试列名标准化函数的功能。
    
    创建一个模拟数据集，应用standardize_columns函数，
    然后验证列名是否已成功标准化。
    
    Returns:
        bool: 测试通过返回True，否则返回False
    
    Raises:
        ImportError: 当无法导入standardize_columns函数时抛出
        Exception: 测试过程中出现的其他错误
    """
    try:
        # 导入standardize_columns函数
        from src.utils.column_mapping import standardize_columns
        
        # 创建测试数据
        test_data = {
            '日期': ['2023-01-01', '2023-01-02', '2023-01-03'],
            '开盘': [100, 101, 102],
            '收盘': [101, 102, 103],
            '最高': [102, 103, 104],
            '最低': [99, 100, 101],
            '成交量': [1000, 2000, 3000],
            '成交额': [100000, 200000, 300000]
        }
        
        df = pd.DataFrame(test_data)
        logger.info(f"原始列名: {df.columns.tolist()}")
        
        # 使用standardize_columns函数
        result_df = standardize_columns(df, source_type='AKSHARE')
        logger.info(f"标准化后的列名: {result_df.columns.tolist()}")
        
        # 验证列名是否标准化
        expected_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
        actual_columns = result_df.columns.tolist()
        
        # 检查结果是否符合预期
        is_successful = all(col in actual_columns for col in expected_columns)
        
        if is_successful:
            logger.info("测试成功！standardize_columns函数工作正常")
            return True
        else:
            logger.error(f"测试失败！预期列名: {expected_columns}, 实际列名: {actual_columns}")
            return False
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_standardize_columns() 