"""
测试股票代码标准化功能
"""

import sys
import os
import unittest
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入相关模块
from src.data_sources.akshare import AKShareDS
from config.config import DATA_SOURCES

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class TestCodeStandardization(unittest.TestCase):
    """测试股票代码标准化功能"""
    
    def setUp(self):
        """初始化测试环境"""
        self.ds = AKShareDS(DATA_SOURCES.get('akshare', {}))
    
    def test_normalize_symbol(self):
        """测试内部的_normalize_symbol方法"""
        # 测试不同格式的股票代码
        test_cases = {
            '600519': '600519',  # 上证主板
            '000001': '000001',  # 深证主板
            '300059': '300059',  # 创业板
            '688001': '688001',  # 科创板
            '001279': '001279',  # 深证主板新股
            '1279': '001279',    # 带前导0的处理
        }
        
        for input_code, expected_code in test_cases.items():
            output_code = self.ds._normalize_symbol(input_code)
            logger.info(f"输入代码: {input_code}, 标准化后: {output_code}, 期望结果: {expected_code}")
            self.assertEqual(output_code, expected_code)
    
    def test_standardize_code(self):
        """测试公开的standardize_code方法"""
        # 测试不同格式的股票代码
        test_cases = {
            '600519': '600519',  # 上证主板
            '000001': '000001',  # 深证主板
            '300059': '300059',  # 创业板
            '688001': '688001',  # 科创板
            '001279': '001279',  # 深证主板新股
            '1279': '001279',    # 带前导0的处理
        }
        
        for input_code, expected_code in test_cases.items():
            output_code = self.ds.standardize_code(input_code)
            logger.info(f"输入代码: {input_code}, 标准化后: {output_code}, 期望结果: {expected_code}")
            self.assertEqual(output_code, expected_code)

if __name__ == "__main__":
    unittest.main() 