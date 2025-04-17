#!/usr/bin/env python
"""
测试增强版列标准化功能

该脚本测试column_mapping.py的增强功能，包括模糊匹配、
内容推断和用户自定义映射支持。

创建日期: 2024-04-13
作者: BeeShare开发团队
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

# 假定增强版函数已经实现
try:
    from src.utils.enhanced_column_mapping import (
        find_closest_column_match,
        infer_column_type,
        load_user_mappings,
        standardize_columns,
        standardize_date_format,
        StandardColumns  # 假设复用了原来的枚举
    )
except ImportError:
    # 如果还没有实现，提示信息
    logger.warning("增强版列标准化模块尚未实现，本测试将失败")
    # 为了测试目的，我们可以从原始模块导入，并创建mock函数
    from src.utils.column_mapping import StandardColumns
    
    def find_closest_column_match(column_name, target_columns, threshold=0.8):
        """模拟实现"""
        logger.warning("使用模拟的find_closest_column_match函数")
        return None
        
    def infer_column_type(df, column_name):
        """模拟实现"""
        logger.warning("使用模拟的infer_column_type函数")
        return None
        
    def load_user_mappings(mapping_file="config/column_mappings.json"):
        """模拟实现"""
        logger.warning("使用模拟的load_user_mappings函数")
        return {}
        
    def standardize_columns(df, source_type, use_fuzzy_match=False, infer_column_types=False, logger=None):
        """模拟实现"""
        logger.warning("使用模拟的standardize_columns函数")
        from src.utils.column_mapping import standardize_columns as orig_standardize
        return orig_standardize(df, source_type, logger)
        
    def standardize_date_format(df, date_column='date', inplace=False):
        """模拟实现"""
        logger.warning("使用模拟的standardize_date_format函数")
        return df


class TestEnhancedColumnMapping(unittest.TestCase):
    """测试增强版列标准化功能"""
    
    def setUp(self):
        """初始化测试数据"""
        # 创建样本数据
        self.sample_data = pd.DataFrame({
            # 标准命名的列
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'close': [101, 102, 103, 104, 105],
            
            # 需要模糊匹配的列
            'highest_price': [105, 106, 107, 108, 109],  # 应匹配到'high'
            'lowest_price': [98, 99, 100, 101, 102],     # 应匹配到'low'
            'volumeTraded': [10000, 11000, 12000, 13000, 14000],  # 应匹配到'volume'
            
            # 需要内容推断的列
            'numeric_column_1': [1500000, 1600000, 1700000, 1800000, 1900000],  # 大数值，可能是'amount'
            'numeric_column_2': [5, 10, 15, 20, 25],  # 小数值，难以推断
            'string_column': ['000001', '000002', '000003', '000004', '000005'],  # 可能是'symbol'
            
            # 特殊格式的日期列
            'timestamp': [1609459200, 1609545600, 1609632000, 1609718400, 1609804800]  # Unix时间戳
        })
        
        # 不同命名的数据源
        self.custom_source_data = pd.DataFrame({
            '日期时间': ['2023-01-01', '2023-01-02', '2023-01-03'],
            '开盘价格': [100, 101, 102],
            '收盘价格': [101, 102, 103],
            '最高价格': [102, 103, 104],
            '最低价格': [99, 100, 101],
            '成交量(手)': [1000, 2000, 3000],
            '成交金额(元)': [100000, 200000, 300000],
            '股票代码': ['000001', '000002', '000003'],
            '股票简称': ['平安银行', '万科A', '神州数码']
        })
    
    def test_find_closest_column_match(self):
        """测试模糊匹配功能"""
        test_cases = [
            # (列名, 目标列表, 期望匹配)
            ('highest_price', ['high', 'low', 'close'], 'high'),
            ('lowest_price', ['high', 'low', 'close'], 'low'),
            ('volumeTraded', ['volume', 'amount', 'turnover'], 'volume'),
            ('daily_change', ['change', 'change_pct', 'turnover'], 'change'),
            ('percent_chg', ['change', 'change_pct', 'turnover'], 'change_pct'),
            ('no_match_at_all', ['open', 'close', 'high', 'low'], None)  # 低于阈值，无匹配
        ]
        
        for col_name, targets, expected in test_cases:
            result = find_closest_column_match(col_name, targets)
            logger.info(f"列名 '{col_name}' 匹配结果: {result}, 期望: {expected}")
            self.assertEqual(result, expected, f"'{col_name}' 应匹配到 '{expected}'，但得到 '{result}'")
    
    def test_infer_column_type(self):
        """测试列类型推断功能"""
        # 测试日期类型推断
        self.assertEqual(
            infer_column_type(self.sample_data, 'date'), 
            StandardColumns.DATE,
            "应将'date'列识别为日期类型"
        )
        
        # 测试数值类型推断
        self.assertEqual(
            infer_column_type(self.sample_data, 'numeric_column_1'), 
            StandardColumns.AMOUNT,
            "应将大数值列识别为金额类型"
        )
        
        # 测试字符串类型推断
        self.assertEqual(
            infer_column_type(self.sample_data, 'string_column'), 
            StandardColumns.SYMBOL,
            "应将股票代码格式的列识别为代码类型"
        )
    
    def test_load_user_mappings(self):
        """测试加载用户自定义映射"""
        mappings = load_user_mappings()
        self.assertIsInstance(mappings, dict, "用户映射应该是字典类型")
        
        # 检查是否包含示例中的映射键
        expected_keys = ["CUSTOM_SOURCE", "CSV_EXPORT", "EXCEL_DATA", "WEB_SCRAPER"]
        for key in expected_keys:
            self.assertIn(key, mappings, f"用户映射应包含'{key}'")
    
    def test_enhanced_standardize_columns(self):
        """测试增强版standardize_columns函数"""
        # 测试精确映射
        result = standardize_columns(self.custom_source_data, 'CUSTOM_SOURCE')
        expected_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'symbol', 'name']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"结果应包含标准列'{col}'")
        
        # 测试模糊匹配
        result = standardize_columns(
            self.sample_data, 
            'A_SHARE', 
            use_fuzzy_match=True
        )
        self.assertIn('high', result.columns, "应通过模糊匹配将'highest_price'映射为'high'")
        self.assertIn('low', result.columns, "应通过模糊匹配将'lowest_price'映射为'low'")
        
        # 测试内容推断
        result = standardize_columns(
            self.sample_data, 
            'A_SHARE', 
            infer_column_types=True
        )
        self.assertIn('amount', result.columns, "应通过内容推断将'numeric_column_1'映射为'amount'")
        self.assertIn('symbol', result.columns, "应通过内容推断将'string_column'映射为'symbol'")
    
    def test_standardize_date_format(self):
        """测试日期格式标准化"""
        # 创建日期测试数据
        date_test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023/01/02', '20230103'],  # 字符串日期
            'timestamp': [1609459200, 1609545600, 1609632000]  # Unix时间戳(秒)
        })
        
        # 测试字符串日期标准化
        result = standardize_date_format(date_test_data)
        self.assertTrue(pd.api.types.is_datetime64_dtype(result['date']), 
                        "字符串日期应转换为datetime64类型")
        
        # 测试时间戳标准化
        result = standardize_date_format(date_test_data, date_column='timestamp')
        self.assertTrue(pd.api.types.is_datetime64_dtype(result['timestamp']), 
                        "时间戳应转换为datetime64类型")
        
        # 验证日期值正确
        expected_dates = pd.date_range(start='2023-01-01', periods=3)
        pd.testing.assert_index_equal(result['date'].dt.normalize().sort_values().reset_index(drop=True), 
                                      expected_dates.sort_values(), 
                                      "转换后的日期值应正确")

if __name__ == "__main__":
    unittest.main() 