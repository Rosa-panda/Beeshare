#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源使用示例

展示如何使用重构后的数据源模块。
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_source import DataSourceFactory, DataSourceError
from src.data.config import data_source_config


def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger('BeeShare')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    
    return logger


def show_available_sources():
    """显示可用的数据源"""
    logger = logging.getLogger('BeeShare.Example')
    sources = DataSourceFactory.get_available_sources()
    logger.info(f"可用的数据源: {', '.join(sources)}")
    
    return sources


def demo_get_stock_list():
    """演示获取股票列表"""
    logger = logging.getLogger('BeeShare.Example')
    logger.info("============ 获取股票列表 ============")
    
    try:
        # 获取AKShare数据源实例
        data_source = DataSourceFactory.get_data_source('akshare')
        
        # 获取股票列表
        stock_list = data_source.get_all_stock_list()
        
        logger.info(f"股票总数: {len(stock_list)}")
        logger.info(f"前5条记录:\n{stock_list.head()}")
        
    except DataSourceError as e:
        logger.error(f"获取股票列表失败: {str(e)}")
        if e.original_error:
            logger.error(f"原始错误: {str(e.original_error)}")


def demo_search_stock(keyword):
    """演示搜索股票
    
    Args:
        keyword: 搜索关键词
    """
    logger = logging.getLogger('BeeShare.Example')
    logger.info(f"============ 搜索股票: {keyword} ============")
    
    try:
        # 获取AKShare数据源实例
        data_source = DataSourceFactory.get_data_source('akshare')
        
        # 搜索股票
        result = data_source.search_symbols(keyword)
        
        if result.empty:
            logger.info(f"未找到匹配 '{keyword}' 的股票")
        else:
            logger.info(f"找到 {len(result)} 条记录")
            logger.info(f"搜索结果:\n{result}")
        
    except DataSourceError as e:
        logger.error(f"搜索股票失败: {str(e)}")
        if e.original_error:
            logger.error(f"原始错误: {str(e.original_error)}")


def demo_identify_stock(symbol):
    """演示识别股票类型
    
    Args:
        symbol: 股票代码
    """
    logger = logging.getLogger('BeeShare.Example')
    logger.info(f"============ 识别股票类型: {symbol} ============")
    
    try:
        # 获取AKShare数据源实例
        data_source = DataSourceFactory.get_data_source('akshare')
        
        # 识别股票类型
        result = data_source.identify_stock_type(symbol)
        
        logger.info(f"识别结果: {result}")
        
    except DataSourceError as e:
        logger.error(f"识别股票类型失败: {str(e)}")
        if e.original_error:
            logger.error(f"原始错误: {str(e.original_error)}")


def demo_get_historical_data(symbol, days=30):
    """演示获取历史数据
    
    Args:
        symbol: 股票代码
        days: 获取天数，默认30天
    """
    logger = logging.getLogger('BeeShare.Example')
    logger.info(f"============ 获取历史数据: {symbol}, {days}天 ============")
    
    try:
        # 获取AKShare数据源实例
        data_source = DataSourceFactory.get_data_source('akshare')
        
        # 计算日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 获取历史数据
        result = data_source.get_historical_data(symbol, start_date, end_date)
        
        if result.empty:
            logger.info(f"未找到股票 {symbol} 的历史数据")
        else:
            logger.info(f"获取到 {len(result)} 条记录")
            logger.info(f"数据预览:\n{result.head()}")
        
    except DataSourceError as e:
        logger.error(f"获取历史数据失败: {str(e)}")
        if e.original_error:
            logger.error(f"原始错误: {str(e.original_error)}")


def demo_get_realtime_data(symbols):
    """演示获取实时数据
    
    Args:
        symbols: 股票代码列表
    """
    logger = logging.getLogger('BeeShare.Example')
    logger.info(f"============ 获取实时数据: {symbols} ============")
    
    try:
        # 获取AKShare数据源实例
        data_source = DataSourceFactory.get_data_source('akshare')
        
        # 获取实时数据
        result = data_source.get_realtime_data(symbols)
        
        if result.empty:
            logger.info(f"未获取到实时数据")
        else:
            logger.info(f"获取到 {len(result)} 条记录")
            logger.info(f"数据预览:\n{result}")
        
    except DataSourceError as e:
        logger.error(f"获取实时数据失败: {str(e)}")
        if e.original_error:
            logger.error(f"原始错误: {str(e.original_error)}")


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger()
    logger.info("开始数据源示例演示")
    
    try:
        # 显示可用数据源
        show_available_sources()
        
        # 演示获取股票列表
        demo_get_stock_list()
        
        # 演示搜索股票
        demo_search_stock("银行")
        
        # 演示识别股票类型
        demo_identify_stock("600036")  # 招商银行
        demo_identify_stock("000001")  # 平安银行
        demo_identify_stock("688001")  # 华兴源创
        
        # 演示获取历史数据
        demo_get_historical_data("600036", 10)  # 获取招商银行最近10天的数据
        
        # 演示获取实时数据
        demo_get_realtime_data(["600036", "601398"])  # 获取招商银行和工商银行的实时数据
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {str(e)}")
    finally:
        logger.info("数据源示例演示结束")


if __name__ == "__main__":
    main() 