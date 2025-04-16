#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQLite存储模块的测试脚本

测试流程图:
```mermaid
flowchart TD
    A[开始测试] --> B[初始化SQLite存储]
    B --> C[测试股票代码数据存储与读取]
    C --> D[测试历史数据存储与读取]
    D --> E[测试实时数据存储与读取]
    E --> F[测试数据删除功能]
    F --> G[测试日期过滤功能]
    G --> H[结束测试]
```
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.sqlite_storage import SQLiteStorage
from src.storage.base import DataType

class TestSQLiteStorage(unittest.TestCase):
    """测试SQLite存储实现的类"""
    
    def setUp(self):
        """测试开始前的准备工作"""
        # 使用临时文件作为数据库
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, 'test_stockdata.db')
        
        # 初始化存储实例
        self.storage = SQLiteStorage(self.db_path)
        
        # 创建测试数据
        # 股票代码数据
        self.symbols_data = pd.DataFrame({
            'symbol': ['000001', '000002', '000003'],
            'name': ['平安银行', '万科A', '神州高铁'],
            'industry': ['银行', '房地产', '交通运输'],
            'market': ['SZ', 'SZ', 'SZ']
        })
        
        # 历史数据
        self.historical_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10).date,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100 + 10,
            'low': np.random.rand(10) * 100 - 10,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000, 10000, 10),
            'turnover': np.random.rand(10) * 10,
            'change_pct': np.random.rand(10) * 5 - 2.5
        })
        
        # 实时数据
        self.realtime_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(5)],
            'price': np.random.rand(5) * 100,
            'volume': np.random.randint(100, 1000, 5),
            'bid_price': np.random.rand(5) * 100 - 0.5,
            'ask_price': np.random.rand(5) * 100 + 0.5,
            'bid_volume': np.random.randint(50, 500, 5),
            'ask_volume': np.random.randint(50, 500, 5),
            'change_pct': np.random.rand(5) * 2 - 1
        })
        
        # 指数数据
        self.index_data = pd.DataFrame({
            'symbol': ['000001.SH'] * 10,
            'date': pd.date_range(start='2023-01-01', periods=10).date,
            'open': np.random.rand(10) * 3000 + 3000,
            'high': np.random.rand(10) * 3000 + 3050,
            'low': np.random.rand(10) * 3000 + 2950,
            'close': np.random.rand(10) * 3000 + 3000,
            'volume': np.random.randint(100000, 1000000, 10),
            'change_pct': np.random.rand(10) * 2 - 1
        })
    
    def tearDown(self):
        """测试结束后的清理工作"""
        # 删除临时目录和数据库文件
        self.temp_dir.cleanup()
    
    def test_save_and_load_symbols(self):
        """测试保存和加载股票代码数据"""
        # 保存股票代码数据
        result = self.storage.save_data(self.symbols_data, DataType.SYMBOL)
        self.assertTrue(result, "保存股票代码数据应该成功")
        
        # 加载所有股票代码数据
        loaded_data = self.storage.load_data(DataType.SYMBOL)
        self.assertFalse(loaded_data.empty, "加载的股票代码数据不应为空")
        self.assertEqual(len(loaded_data), len(self.symbols_data), "加载的数据行数应与保存的相同")
        
        # 加载特定股票代码数据
        symbol = self.symbols_data['symbol'][0]
        loaded_single = self.storage.load_data(DataType.SYMBOL, symbol)
        self.assertEqual(len(loaded_single), 1, "应该只加载一条股票代码数据")
        self.assertEqual(loaded_single['symbol'].iloc[0], symbol, "加载的股票代码应该与请求的相同")
    
    def test_save_and_load_historical_data(self):
        """测试保存和加载历史数据"""
        symbol = '000001'
        
        # 保存历史数据
        result = self.storage.save_data(self.historical_data, DataType.HISTORICAL, symbol)
        self.assertTrue(result, "保存历史数据应该成功")
        
        # 加载历史数据
        loaded_data = self.storage.load_data(DataType.HISTORICAL, symbol)
        self.assertFalse(loaded_data.empty, "加载的历史数据不应为空")
        self.assertEqual(len(loaded_data), len(self.historical_data), "加载的数据行数应与保存的相同")
        
        # 测试日期过滤
        start_date = self.historical_data['date'].iloc[2]
        end_date = self.historical_data['date'].iloc[7]
        filtered_data = self.storage.load_data(DataType.HISTORICAL, symbol, start_date, end_date)
        self.assertEqual(len(filtered_data), 6, "按日期过滤后应有6条数据")
    
    def test_save_and_load_realtime_data(self):
        """测试保存和加载实时数据"""
        symbol = '000001'
        
        # 保存实时数据
        result = self.storage.save_data(self.realtime_data, DataType.REALTIME, symbol)
        self.assertTrue(result, "保存实时数据应该成功")
        
        # 加载实时数据
        loaded_data = self.storage.load_data(DataType.REALTIME, symbol)
        self.assertFalse(loaded_data.empty, "加载的实时数据不应为空")
        self.assertEqual(len(loaded_data), len(self.realtime_data), "加载的数据行数应与保存的相同")
    
    def test_save_and_load_index_data(self):
        """测试保存和加载指数数据"""
        # 保存指数数据
        result = self.storage.save_data(self.index_data, DataType.INDEX)
        self.assertTrue(result, "保存指数数据应该成功")
        
        # 加载所有指数数据
        loaded_data = self.storage.load_data(DataType.INDEX)
        self.assertFalse(loaded_data.empty, "加载的指数数据不应为空")
        self.assertEqual(len(loaded_data), len(self.index_data), "加载的数据行数应与保存的相同")
        
        # 加载特定指数数据
        symbol = self.index_data['symbol'][0]
        loaded_single = self.storage.load_data(DataType.INDEX, symbol)
        self.assertEqual(len(loaded_single), 10, "应该加载10条指数数据")
        self.assertEqual(loaded_single['symbol'].iloc[0], symbol, "加载的指数代码应该与请求的相同")
    
    def test_delete_data(self):
        """测试删除数据"""
        symbol = '000001'
        
        # 首先保存一些数据
        self.storage.save_data(self.symbols_data, DataType.SYMBOL)
        self.storage.save_data(self.historical_data, DataType.HISTORICAL, symbol)
        
        # 测试删除特定股票的历史数据
        result = self.storage.delete_data(DataType.HISTORICAL, symbol)
        self.assertTrue(result, "删除特定股票的历史数据应该成功")
        
        # 验证数据已被删除
        loaded_data = self.storage.load_data(DataType.HISTORICAL, symbol)
        self.assertTrue(loaded_data.empty, "删除后加载的历史数据应为空")
        
        # 测试删除特定股票的代码数据
        result = self.storage.delete_data(DataType.SYMBOL, symbol)
        self.assertTrue(result, "删除特定股票的代码数据应该成功")
        
        # 验证特定股票数据已被删除，但其他股票数据还在
        loaded_all = self.storage.load_data(DataType.SYMBOL)
        self.assertEqual(len(loaded_all), len(self.symbols_data) - 1, "应该删除了一条股票代码数据")
        
        # 测试删除所有股票代码数据
        result = self.storage.delete_data(DataType.SYMBOL)
        self.assertTrue(result, "删除所有股票代码数据应该成功")
        
        # 验证所有股票代码数据已被删除
        loaded_all = self.storage.load_data(DataType.SYMBOL)
        self.assertTrue(loaded_all.empty, "删除后加载的所有股票代码数据应为空")

if __name__ == '__main__':
    unittest.main() 