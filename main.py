"""
股票数据获取系统主程序

提供命令行接口，用于获取各种股票数据并存储

流程图:
```mermaid
graph TD
    A[开始] --> B[加载配置]
    B --> C[初始化数据源]
    C --> D[初始化存储]
    D --> E[解析命令行参数]
    E --> F{选择操作类型}
    F -->|获取历史数据| G[获取历史数据]
    F -->|获取实时数据| H[获取实时数据]
    F -->|搜索股票| I[搜索股票]
    G --> J[保存数据]
    H --> J
    J --> K[结束]
    I --> K
```
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import traceback
import sqlite3
import time
import json
import configparser

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
# 移除重复导入，只使用新版数据源
# 移除不存在的模块导入
from config.config import (
    DATA_SOURCES, 
    FETCH_CONFIG, 
    STORAGE_CONFIG, 
    MARKETS,
    LOG_CONFIG
)
from src.storage.base import DataType
from src.data_sources.akshare import AKShareDS
from src.storage.sqlite_storage import SQLiteStorage
from src.analysis.clustering import ClusteringAnalyzer
from src.utils.column_mapping import (
    StandardColumns, 
    standardize_columns,
    detect_and_log_column_issues,
    get_standard_column_list
)

# 设置日志记录
from src.utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# 导入项目功能模块
from src.api.stock_api import StockAPI
# 移除重复导入
from src.common.constants import DATA_SOURCES
from src.utils.config_manager import ConfigManager

def init_data_sources():
    """
    初始化所有启用的数据源
    
    Returns:
        dict: 数据源名称到数据源实例的映射
    """
    data_sources = {}
    
    # 初始化AKShare数据源
    if DATA_SOURCES.get('akshare', {}).get('enabled', False):
        logger.info("初始化AKShare数据源")
        akshare_config = DATA_SOURCES.get('akshare', {})
        akshare_ds = AKShareDS(akshare_config)
        if akshare_ds.is_ready:
            data_sources['akshare'] = akshare_ds
        else:
            logger.warning("AKShare数据源初始化失败，将不可用")
    
    return data_sources

def init_storage():
    """
    初始化存储
    
    Returns:
        dict: 存储名称到存储实例的映射
    """
    from src.storage.config import StorageConfig
    
    storage_instances = {}
    
    # 获取活跃的存储类型
    storage_config = StorageConfig()
    active_storage_type = storage_config.get_storage_type()
    
    # 确保活跃存储类型为sqlite
    if active_storage_type != 'sqlite':
        logger.warning(f"系统现在只支持SQLite存储，将配置从{active_storage_type}修改为sqlite")
        storage_config.set_storage_type('sqlite', migrate=False)
        active_storage_type = 'sqlite'
    
    # 初始化SQLite存储
    try:
        logger.info("初始化SQLite存储")
        from src.storage.sqlite_storage import SQLiteStorage
        sqlite_config = storage_config.get_storage_config('sqlite')
        sqlite_storage = SQLiteStorage(sqlite_config['db_path'])
        storage_instances['sqlite'] = sqlite_storage
    except Exception as e:
        logger.error(f"初始化SQLite存储失败: {e}")
    
    return storage_instances

def get_historical_data(args, data_sources, storage):
    """
    获取历史股票数据并存储
    
    Args:
        args: 命令行参数
        data_sources: 数据源实例
        storage: 存储实例
    
    Returns:
        bool: 是否成功
    """
    # 检查是否提供了股票代码
    if not args.symbol:
        logger.error("未提供股票代码，无法获取历史数据")
        return False
    
    # 确定使用的数据源
    ds_name = args.source.lower() if args.source else 'akshare'  # 默认使用AKShare
    data_source = data_sources.get(ds_name)
    
    if not data_source:
        logger.error(f"数据源 {ds_name} 不可用")
        return False
    
    # 获取历史数据
    logger.info(f"从 {ds_name} 获取 {args.symbol} 的历史数据...")
    
    # 如果未提供end_date，使用当前日期
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    logger.info(f"数据时间范围: {args.start_date} 至 {end_date}")
    
    # 转换interval格式
    interval = args.interval
    if interval == '1d':
        interval = 'daily'
    elif interval == '1wk':
        interval = 'weekly'
    elif interval == '1mo':
        interval = 'monthly'
    
    # 先尝试获取股票信息，确认股票是否存在
    try:
        stock_info = data_source.get_symbol_info(args.symbol) if hasattr(data_source, 'get_symbol_info') else {}
        if stock_info and stock_info.get('name'):
            logger.info(f"股票确认: {args.symbol} - {stock_info.get('name')}, 交易所: {stock_info.get('exchange')}")
    except Exception as e:
        logger.warning(f"获取股票信息时出错: {e}")
    
    # 格式化标准股票代码 - 使用_normalize_symbol方法替代standardize_code
    if hasattr(data_source, '_normalize_symbol'):
        std_symbol = data_source._normalize_symbol(args.symbol)
    else:
        # 如果没有_normalize_symbol方法，直接使用原始代码
        std_symbol = args.symbol
        logger.warning(f"数据源没有提供规范化股票代码的方法，使用原始代码: {args.symbol}")
    
    if not std_symbol:
        logger.warning(f"无法识别的股票代码: {args.symbol}")
        return False
    
    # 获取历史数据
    data = data_source.get_historical_data(
        symbol=std_symbol, 
        start_date=args.start_date, 
        end_date=end_date,
        interval=interval
    )
    
    if data.empty:
        logger.warning(f"未找到 {args.symbol} 的历史数据")
        
        # 尝试判断是否为新上市股票
        if stock_info and '上市日期' in str(stock_info):
            logger.info(f"提示: {args.symbol} 可能是新上市股票，上市日期为 {stock_info.get('listing_date', '未知')}")
            logger.info(f"请尝试使用晚于股票上市日期的起始日期进行查询。")
        
        logger.info(f"建议: 您可以尝试以下方法:")
        logger.info(f"1. 使用不同的日期范围")
        logger.info(f"2. 检查股票代码是否正确")
        logger.info(f"3. 尝试使用其他数据源")
        logger.info(f"4. 如果这是一支新上市的股票，可能尚未有足够的历史数据。您可以尝试获取实时数据: python main.py realtime --symbol {args.symbol}")
        
        return False
    
    logger.info(f"成功获取 {len(data)} 条历史数据")
    
    # 确定使用的存储方式
    storage_name = args.storage.lower() if args.storage else 'sqlite'  # 默认使用SQLite存储
    storage_instance = storage.get(storage_name)
    
    if not storage_instance:
        logger.error(f"存储方式 {storage_name} 不可用")
        return False
    
    # 存储数据
    logger.info(f"将数据保存到 {storage_name} 存储...")
    success = storage_instance.save_data(data=data, data_type=DataType.HISTORICAL, symbol=args.symbol)
    
    if success:
        logger.info(f"成功保存 {args.symbol} 的历史数据")
    else:
        logger.error(f"保存 {args.symbol} 的历史数据失败")
    
    return success

def get_realtime_data(args, data_sources, storage):
    """
    获取实时股票数据并存储
    
    Args:
        args: 命令行参数
        data_sources: 数据源实例
        storage: 存储实例
    
    Returns:
        bool: 是否成功
    """
    # 确定要获取的股票
    symbols = []
    
    if args.symbol:
        # 如果命令行提供了股票代码，使用它
        symbols = args.symbol.split(',')
    elif hasattr(args, 'market') and args.market:
        # 否则使用指定市场的默认股票
        market = args.market.lower()
        if market == 'cn':
            # 默认获取中国A股主要股票数据
            symbols = ['600036', '601398', '600276', '600519', '601288']
            logger.info(f"使用默认A股股票列表: {symbols}")
        else:
            logger.error(f"未知的市场代码: {args.market}")
            return False
    
    if not symbols:
        logger.error("未提供股票代码(--symbol)或市场(--market)，无法获取实时数据")
        return False
    
    # 确定使用的数据源
    ds_name = args.source if args.source else 'akshare'  # 默认使用AKShare
    data_source = data_sources.get(ds_name)
    
    if not data_source:
        logger.error(f"数据源 {ds_name} 不可用")
        return False
    
    # 获取实时数据
    logger.info(f"从 {ds_name} 获取 {len(symbols)} 只股票的实时数据...")
    
    try:
        df = data_source.get_realtime_data(symbols)
        
        if df.empty:
            logger.warning(f"未找到实时数据")
            return False
        
        logger.info(f"成功获取 {len(df)} 条实时数据")
        
        # 确定使用的存储方式
        storage_name = args.storage if args.storage else 'sqlite'  # 默认使用SQLite存储
        storage_instance = storage.get(storage_name)
        
        if not storage_instance:
            logger.error(f"存储方式 {storage_name} 不可用")
            return False
        
        # 存储数据
        logger.info(f"将数据保存到 {storage_name} 存储...")
        
        # 检查数据是否包含symbol列
        if 'symbol' in df.columns:
            success = True
            # 按照股票代码分组保存数据
            for symbol, group_data in df.groupby('symbol'):
                symbol_success = storage_instance.save_data(data=group_data, data_type=DataType.REALTIME, symbol=symbol)
                if not symbol_success:
                    success = False
                    logger.error(f"保存股票 {symbol} 的实时数据失败")
        else:
            logger.error("实时数据中没有股票代码(symbol)列，无法保存")
            success = False
        
        if success:
            logger.info(f"成功保存实时数据")
        else:
            logger.error(f"保存实时数据失败")
        
        return success
        
    except Exception as e:
        logger.error(f"获取实时数据失败: {e}")
        raise

def cleanup_realtime_tables(args, data_sources, storage):
    """清理实时数据表，用于结构改变后重建表"""
    try:
        storage_name = args.storage if args.storage else 'sqlite'  # 默认使用SQLite存储
        storage_instance = storage.get(storage_name)
        
        if not storage_instance:
            logger.error(f"未找到存储方式: {storage_name}")
            return
        
        logger.info(f"清理实时数据表...")
        
        # 处理市场参数或特定股票
        if args.market:
            if args.market.lower() == 'cn':
                # 默认清理中国A股主要股票数据表
                symbols = ['600036', '601398', '600276', '600519', '601288']
                logger.info(f"将清理默认A股股票列表的实时数据表: {symbols}")
            else:
                logger.error(f"未知的市场代码: {args.market}")
                return
        elif args.symbol:
            symbols = args.symbol.split(',')
            logger.info(f"将清理指定股票的实时数据表: {symbols}")
        else:
            logger.warning("未提供股票代码或市场，将尝试清理所有实时数据表")
            symbols = None
        
        if symbols:
            for symbol in symbols:
                success = storage_instance.delete_data(DataType.REALTIME, symbol)
                if success:
                    logger.info(f"成功删除 {symbol} 的实时数据表")
                else:
                    logger.warning(f"删除 {symbol} 的实时数据表失败或表不存在")
        else:
            # 实现清理所有实时数据表的逻辑
            if storage_name == 'sqlite':
                try:
                    # 使用SQLite存储特有的方法获取所有表
                    conn = storage_instance._get_connection()
                    # 获取所有以"realtime_"开头的表
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'realtime_%'")
                    tables = cursor.fetchall()
                    
                    if not tables:
                        logger.info("未找到任何实时数据表")
                    else:
                        count = 0
                        for (table_name,) in tables:
                            # 从表名中提取股票代码
                            symbol = table_name.replace('realtime_', '')
                            success = storage_instance.delete_data(DataType.REALTIME, symbol)
                            if success:
                                count += 1
                                logger.info(f"成功删除 {symbol} 的实时数据表")
                            else:
                                logger.warning(f"删除 {symbol} 的实时数据表失败")
                        
                        logger.info(f"共清理了 {count} 个实时数据表")
                except Exception as e:
                    logger.error(f"清理所有实时数据表时发生错误: {e}")
            else:
                logger.warning(f"清理所有实时数据表功能目前仅支持SQLite存储")
        
        logger.info("实时数据表清理完成，下次获取数据时将自动创建新表")
        
    except Exception as e:
        logger.error(f"清理实时数据表失败: {e}")
        raise

def search_symbols(args, data_sources):
    """
    搜索股票代码
    
    Args:
        args: 命令行参数
        data_sources: 数据源实例
    
    Returns:
        bool: 是否成功
    """
    # 检查是否提供了搜索关键词
    if not args.keyword:
        logger.error("未提供搜索关键词")
        return False
    
    # 确定使用的数据源
    ds_name = args.source.lower() if args.source else 'akshare'  # 默认使用AKShare
    data_source = data_sources.get(ds_name)
    
    if not data_source:
        logger.error(f"数据源 {ds_name} 不可用")
        return False
    
    # 搜索股票
    logger.info(f"使用 {ds_name} 搜索 '{args.keyword}'...")
    
    results = data_source.search_symbols(args.keyword)
    
    if not results:
        logger.warning(f"未找到匹配 '{args.keyword}' 的股票")
        return False
    
    # 显示搜索结果
    logger.info(f"找到 {len(results)} 个匹配结果:")
    
    # 将结果转换为DataFrame并打印
    df = pd.DataFrame(results)
    print(df)
    
    return True

def identify_stock(args, data_sources):
    """
    识别股票代码的详细信息（所属市场、板块等）
    
    Args:
        args: 命令行参数
        data_sources: 数据源实例
    
    Returns:
        bool: 是否成功
    """
    # 检查是否提供了股票代码
    if not args.symbol:
        logger.error("未提供股票代码，无法识别股票信息")
        return False
    
    # 确定使用的数据源
    ds_name = args.source.lower() if args.source else 'akshare'  # 默认使用AKShare
    data_source = data_sources.get(ds_name)
    
    if not data_source:
        logger.error(f"数据源 {ds_name} 不可用")
        return False
    
    # 处理多个股票代码
    symbols = args.symbol.split(',')
    results = []
    
    for symbol in symbols:
        logger.info(f"识别股票代码 {symbol} 的信息...")
        
        # 调用数据源的股票识别方法
        stock_info = data_source._identify_stock_type(symbol)
        
        # 获取更详细的股票信息（如有）
        detailed_info = data_source.get_symbol_info(symbol)
        
        # 合并基本信息和详细信息
        if detailed_info:
            for key, value in detailed_info.items():
                if key not in stock_info or not stock_info[key]:
                    stock_info[key] = value
        
        results.append(stock_info)
    
    # 显示结果
    if results:
        logger.info(f"成功识别 {len(results)} 只股票的信息:")
        
        # 将结果转换为DataFrame并打印
        df = pd.DataFrame(results)
        print(df)
        return True
    else:
        logger.warning("未能识别任何股票信息")
        return False

def get_index_data(args, data_sources, storage):
    """
    获取指数数据并存储
    
    Args:
        args: 命令行参数
        data_sources: 数据源实例
        storage: 存储实例
    
    Returns:
        bool: 是否成功
    """
    # 检查是否提供了指数代码
    if not args.symbol:
        logger.error("未提供指数代码，无法获取指数数据")
        return False
    
    # 确定使用的数据源
    ds_name = args.source.lower() if args.source else 'akshare'  # 默认使用AKShare
    data_source = data_sources.get(ds_name)
    
    if not data_source:
        logger.error(f"数据源 {ds_name} 不可用")
        return False
    
    # 获取指数数据
    logger.info(f"从 {ds_name} 获取指数 {args.symbol} 的数据...")
    
    # 如果未提供end_date，使用当前日期
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    logger.info(f"数据时间范围: {args.start_date} 至 {end_date}")
    
    # 转换interval格式
    interval = args.interval
    if interval == '1d':
        interval = 'daily'
    elif interval == '1wk':
        interval = 'weekly'
    elif interval == '1mo':
        interval = 'monthly'
    
    df = data_source.get_index_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=end_date,
        interval=interval
    )
    
    if df.empty:
        logger.warning(f"未找到指数 {args.symbol} 的数据")
        return False
    
    logger.info(f"成功获取 {len(df)} 条指数数据")
    
    # 确定使用的存储方式
    storage_name = args.storage.lower() if args.storage else 'sqlite'  # 默认使用SQLite存储
    storage_instance = storage.get(storage_name)
    
    if not storage_instance:
        logger.error(f"存储方式 {storage_name} 不可用")
        return False
    
    # 存储数据
    logger.info(f"将指数数据保存到 {storage_name} 存储...")
    success = storage_instance.save_data(
        data=df,
        data_type=DataType.INDEX,
        symbol=args.symbol
    )
    
    if success:
        logger.info(f"成功保存指数 {args.symbol} 的数据")
    else:
        logger.error(f"保存指数 {args.symbol} 的数据失败")
    
    return success

def setup_args(parser=None):
    """设置命令行参数"""
    if parser is None:
        parser = argparse.ArgumentParser(description='股票数据获取与分析工具')
        
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 历史数据
    parser_historical = subparsers.add_parser('historical', help='获取历史数据')
    parser_historical.add_argument('--symbol', type=str, required=True, help='股票代码，可用逗号分隔多个代码')
    parser_historical.add_argument('--start-date', type=str, required=True, help='开始日期，格式为YYYY-MM-DD')
    parser_historical.add_argument('--end-date', type=str, default=None, help='结束日期，格式为YYYY-MM-DD，默认为今天')
    parser_historical.add_argument('--interval', type=str, default='daily', 
                                  choices=['daily', 'weekly', 'monthly', '1d', '1wk', '1mo'], 
                                  help='数据间隔，默认为daily，支持daily/1d, weekly/1wk, monthly/1mo')
    parser_historical.add_argument('--source', type=str, default='akshare', 
                                  choices=get_available_data_sources(), help='数据源，默认为akshare')
    parser_historical.add_argument('--storage', type=str, default='sqlite', 
                                  choices=get_available_storage_methods(), help='存储方式，默认为sqlite')
    parser_historical.add_argument('--storage-mode', type=str, default='append',
                                  choices=['append', 'overwrite'], help='存储模式，默认为append')
    
    # 实时数据
    parser_realtime = subparsers.add_parser('realtime', help='获取实时数据')
    parser_realtime.add_argument('--symbol', type=str, help='股票代码，可用逗号分隔多个代码')
    parser_realtime.add_argument('--market', type=str, choices=['us', 'cn', 'hk'], help='市场代码，指定时将获取该市场默认股票数据')
    parser_realtime.add_argument('--source', type=str, default='akshare', 
                                choices=get_available_data_sources(), help='数据源，默认为akshare')
    parser_realtime.add_argument('--storage', type=str, default='sqlite', 
                                choices=get_available_storage_methods(), help='存储方式，默认为sqlite')
    parser_realtime.add_argument('--storage-mode', type=str, default='overwrite',
                                choices=['append', 'overwrite'], help='存储模式，默认为overwrite')
    
    # 股票查询
    parser_search = subparsers.add_parser('search', help='搜索股票代码')
    parser_search.add_argument('--keyword', type=str, required=True, help='搜索关键词')
    parser_search.add_argument('--source', type=str, default='akshare', 
                              choices=get_available_data_sources(), help='数据源，默认为akshare')
    
    # 股票代码识别
    parser_identify = subparsers.add_parser('identify', help='识别股票代码')
    parser_identify.add_argument('--symbol', type=str, required=True, help='股票代码，可用逗号分隔多个代码')
    parser_identify.add_argument('--source', type=str, default='akshare', 
                               choices=get_available_data_sources(), help='数据源，默认为akshare')
    
    # 指数数据
    parser_index = subparsers.add_parser('index', help='获取指数数据')
    parser_index.add_argument('--symbol', type=str, required=True, help='指数代码，可用逗号分隔多个代码')
    parser_index.add_argument('--start-date', type=str, required=True, help='开始日期，格式为YYYY-MM-DD')
    parser_index.add_argument('--end-date', type=str, default=None, help='结束日期，格式为YYYY-MM-DD，默认为今天')
    parser_index.add_argument('--interval', type=str, default='daily',
                             choices=['daily', 'weekly', 'monthly', '1d', '1wk', '1mo'],
                             help='数据间隔，默认为daily，支持daily/1d, weekly/1wk, monthly/1mo')
    parser_index.add_argument('--source', type=str, default='akshare', 
                             choices=get_available_data_sources(), help='数据源，默认为akshare')
    parser_index.add_argument('--storage', type=str, default='sqlite', 
                             choices=get_available_storage_methods(), help='存储方式，默认为sqlite')
    parser_index.add_argument('--storage-mode', type=str, default='append',
                             choices=['append', 'overwrite'], help='存储模式，默认为append')
    
    # 聚类分析命令
    parser_clustering = subparsers.add_parser('clustering', help='执行股票聚类分析')
    parser_clustering.add_argument('--symbols', '-s', required=True, help='要分析的股票代码，以逗号分隔')
    parser_clustering.add_argument('--start-date', help='开始日期，格式为YYYY-MM-DD，默认为90天前')
    parser_clustering.add_argument('--end-date', help='结束日期，格式为YYYY-MM-DD，默认为今天')
    parser_clustering.add_argument('--interval', default='1d', help='数据间隔，如1d(日)、1wk(周)、1mo(月)，默认为1d')
    parser_clustering.add_argument('--n-clusters', type=int, default=3, help='聚类数量，默认为3')
    parser_clustering.add_argument('--features', help='用于聚类的特征，以逗号分隔，默认为"open,close,high,low,volume,change_percent"')
    parser_clustering.add_argument('--find-optimal', action='store_true', help='是否自动寻找最佳聚类数量')
    parser_clustering.add_argument('--max-clusters', type=int, default=10, help='最大聚类数量，默认为10')
    parser_clustering.add_argument('--plot-type', default='clusters', 
                                choices=['clusters', 'elbow', 'feature_distribution', 'centroids', 'all'], 
                                help='可视化类型，默认为clusters')
    parser_clustering.add_argument('--output-dir', help='结果输出目录，默认为当前目录')
    parser_clustering.add_argument('--use-pca', action='store_true', help='是否使用PCA降维进行可视化')
    parser_clustering.add_argument('--plot-3d', action='store_true', help='是否绘制3D聚类图')
    parser_clustering.add_argument('--source', type=str, default='akshare', 
                                 choices=get_available_data_sources(), help='数据源，默认为akshare')
    parser_clustering.add_argument('--storage', type=str, default='sqlite',
                                 choices=get_available_storage_methods(), help='存储方式，默认为sqlite')
    
    # 存储管理命令
    parser_storage = subparsers.add_parser('storage', help='管理存储配置')
    parser_storage.add_argument('--type', type=str, choices=['sqlite'], 
                               help='设置默认存储类型')
    parser_storage.add_argument('--migrate', action='store_true', 
                               help='迁移数据到新的存储类型')
    parser_storage.add_argument('--status', action='store_true',
                               help='显示当前存储状态')
    parser_storage.add_argument('--optimize', action='store_true',
                               help='优化SQLite数据库（仅适用于SQLite存储）')
    
    # 清理实时数据表命令
    parser_cleanup = subparsers.add_parser("cleanup", help="清理实时数据表")
    parser_cleanup.add_argument("--symbol", help="股票代码，多个代码用逗号分隔")
    parser_cleanup.add_argument("--market", help="市场代码，如cn表示中国A股")
    parser_cleanup.add_argument("--storage", help="存储方式，默认为sqlite")
    
    return parser

def main():
    """主程序入口"""
    try:
        logger.info("股票数据获取系统启动")
        
        # 初始化数据源和存储
        data_sources = init_data_sources()
        storage = init_storage()
        
        # 如果没有可用的数据源或存储，退出程序
        if not data_sources:
            logger.error("没有可用的数据源，程序退出")
            return
        
        if not storage:
            logger.error("没有可用的存储方式，程序退出")
            return
        
        # 设置命令行参数
        parser = setup_args()
        args = parser.parse_args()
        
        # 如果没有指定命令，显示帮助信息
        if args.command is None:
            parser.print_help()
            return
        
        # 执行相应的命令
        if args.command == 'historical':
            get_historical_data(args, data_sources, storage)
        elif args.command == 'realtime':
            get_realtime_data(args, data_sources, storage)
        elif args.command == 'search':
            search_symbols(args, data_sources)
        elif args.command == 'index':
            get_index_data(args, data_sources, storage)
        elif args.command == 'identify':
            identify_stock(args, data_sources)
        elif args.command == 'clustering':
            # 运行聚类分析，使用命令行指定的存储类型
            storage_type = args.storage.lower()
            if storage_type in storage:
                run_clustering_analysis(data_sources['akshare'], storage[storage_type], args)
            else:
                logger.error(f"指定的存储类型 {storage_type} 不可用，请确保它已经正确初始化")
        elif args.command == 'storage':
            # 管理存储配置
            manage_storage(args)
        elif args.command == "cleanup":
            cleanup_realtime_tables(args, data_sources, storage)
        else:
            logger.error(f"未知命令: {args.command}")
            parser.print_help()
        
        logger.info("程序执行完毕")
        
    except Exception as e:
        logger.exception(f"程序执行过程中发生错误: {e}")
        sys.exit(1)

def get_available_data_sources():
    """
    获取可用的数据源列表
    
    Returns:
        list: 可用数据源名称列表
    """
    return [name for name, config in DATA_SOURCES.items() if config.get('enabled', False)]

def get_available_storage_methods():
    """
    获取可用的存储方法列表
    
    Returns:
        list: 可用存储方法名称列表
    """
    return ['sqlite']  # 系统现在只支持SQLite存储

def run_clustering_analysis(data_source, storage, args):
    """
    运行聚类分析
    
    Args:
        data_source: 数据源实例
        storage: 存储实例
        args: 命令行参数
    """
    # 导入必要的库
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    try:
        # 解析参数
        symbols = args.symbols.split(',')
        n_clusters = args.n_clusters
        start_date = args.start_date
        end_date = args.end_date
        # 修改特征处理逻辑，使用默认特征列表
        default_features = ['open', 'close', 'high', 'low', 'volume', 'change_pct']
        features = args.features.split(',') if args.features else default_features
        find_optimal = args.find_optimal
        max_clusters = args.max_clusters
        plot_type = args.plot_type if hasattr(args, 'plot_type') and args.plot_type else 'clusters'
        output_dir = args.output_dir if hasattr(args, 'output_dir') else None
        use_pca = args.use_pca if hasattr(args, 'use_pca') else False
        plot_3d = args.plot_3d if hasattr(args, 'plot_3d') else False
        
        logger.info(f"开始对股票 {symbols} 进行聚类分析")
        print(f"开始对股票 {', '.join(symbols)} 进行聚类分析...")
        
        # 获取历史数据
        all_data = {}
        
        # 设置默认日期范围
        if not start_date:
            # 默认为90天前
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            logger.info(f"未指定开始日期，使用默认值: {start_date}")
            
        if not end_date:
            # 默认为今天
            end_date = datetime.now().strftime('%Y-%m-%d')
            logger.info(f"未指定结束日期，使用默认值: {end_date}")
            
        # 确保日期不是未来日期
        today = datetime.now().strftime('%Y-%m-%d')
        if end_date > today:
            logger.warning(f"指定的结束日期 {end_date} 是未来日期，将使用今天的日期: {today}")
            end_date = today
        
        # 获取每个股票的历史数据
        for symbol in symbols:
            logger.info(f"获取 {symbol} 的历史数据")
            try:
                # 格式化标准股票代码 - 使用_normalize_symbol方法替代standardize_code
                if hasattr(data_source, '_normalize_symbol'):
                    std_symbol = data_source._normalize_symbol(symbol)
                else:
                    # 如果没有_normalize_symbol方法，直接使用原始代码
                    std_symbol = symbol
                    logger.warning(f"数据源没有提供规范化股票代码的方法，使用原始代码: {symbol}")
                
                if not std_symbol:
                    logger.warning(f"无法识别的股票代码: {symbol}")
                    continue
                
                # 获取历史数据
                data = data_source.get_historical_data(
                    symbol=std_symbol, 
                    start_date=start_date, 
                    end_date=end_date,
                    interval=args.interval
                )
                
                if data is None or data.empty:
                    logger.warning(f"获取不到 {symbol} 的历史数据")
                    continue
                    
                # 数据预处理
                # 检查是否已经包含change_pct列，如果没有则计算
                if 'change_pct' not in data.columns:
                    if 'close' in data.columns:
                        data['change_pct'] = data['close'].pct_change() * 100
                        logger.info(f"已计算 {symbol} 的change_pct列")
                    else:
                        logger.warning(f"{symbol} 数据缺少close列，无法计算change_pct")
                        continue
                
                # 删除缺失值
                data.dropna(inplace=True)
                
                # 检查是否有足够的数据进行聚类分析
                if len(data) < 10:  # 假设至少需要10条数据进行有意义的聚类
                    logger.warning(f"{symbol} 有效数据不足，仅有 {len(data)} 行")
                    continue
                
                # 验证所需的特征列是否都存在
                missing_features = [feat for feat in features if feat not in data.columns]
                if missing_features:
                    logger.warning(f"{symbol} 数据缺少以下特征列: {missing_features}")
                    
                    # 尝试生成缺失的特征
                    for feat in missing_features:
                        if feat == 'change_pct' and 'close' in data.columns:
                            data['change_pct'] = data['close'].pct_change() * 100
                            logger.info(f"已计算 {symbol} 的change_pct列")
                        elif feat == 'volume' and feat not in data.columns:
                            # 如果缺少成交量，则使用随机数据或平均值
                            data['volume'] = np.random.randint(10000, 100000, size=len(data))
                            logger.warning(f"为 {symbol} 生成了随机volume数据")
                
                # 再次检查特征是否全部存在
                missing_features = [feat for feat in features if feat not in data.columns]
                if missing_features:
                    logger.error(f"{symbol} 数据仍然缺少以下特征列: {missing_features}，跳过该股票")
                    continue
                
                # 添加到数据集
                all_data[symbol] = data
                logger.info(f"成功添加 {symbol} 的数据，共 {len(data)} 行")
                
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
                continue
        
        if not all_data:
            logger.error("没有获取到任何有效的历史数据")
            print("没有获取到任何有效的历史数据，请检查股票代码和日期范围")
            return
        
        # 合并所有数据
        combined_data = []
        for symbol, data in all_data.items():
            data = data.copy()
            data['symbol'] = symbol
            combined_data.append(data)
            
        all_stock_data = pd.concat(combined_data)
        logger.info(f"合并所有股票数据，共 {len(all_stock_data)} 行")
        
        # 创建分析器并设置数据
        logger.info("初始化聚类分析器")
        from src.analysis.clustering import ClusteringAnalyzer
        analyzer = ClusteringAnalyzer()
        analyzer.set_data(all_stock_data)
        
        # 寻找最佳聚类数
        if find_optimal:
            logger.info(f"自动寻找最佳聚类数 (最大聚类数: {max_clusters})")
            print(f"正在自动寻找最佳聚类数 (最大聚类数: {max_clusters})...")
            
            optimal_k, _ = analyzer.determine_optimal_clusters(
                max_clusters=max_clusters, 
                kmeans_params={'features': features}
            )
            
            logger.info(f"确定的最佳聚类数: {optimal_k}")
            print(f"确定的最佳聚类数: {optimal_k}")
            
            # 使用最佳聚类数
            n_clusters = optimal_k
            
            # 可视化肘部法则
            if plot_type == 'elbow' or plot_type == 'all':
                analyzer.visualize(plot_type='elbow', features=features)
        
        # 运行聚类分析
        logger.info(f"使用 {n_clusters} 个聚类运行KMeans聚类")
        print(f"正在使用 {n_clusters} 个聚类运行KMeans聚类分析...")
        
        # 确保features不为None，如果为None则使用默认值
        clustering_features = features if features is not None else default_features
        
        # 确保数据中包含所有需要的特征
        for feature in clustering_features:
            if feature not in all_stock_data.columns:
                logger.error(f"数据中缺少特征列: {feature}")
                print(f"错误: 数据中缺少特征列 '{feature}'，无法进行聚类分析")
                return
        
        clustering_result = analyzer.kmeans_clustering(
            params={
                'n_clusters': n_clusters,
                'features': clustering_features
            }
        )
        
        # 可视化聚类结果
        if plot_type == 'all':
            plot_types = ['clusters', 'feature_distribution', 'centroids']
        elif plot_type:
            plot_types = [plot_type]
        else:
            plot_types = ['clusters']
            
        for p_type in plot_types:
            print(f"正在生成 {p_type} 可视化...")
            try:
                analyzer.visualize(
                    result=clustering_result,
                    plot_type=p_type,
                    use_pca=use_pca,
                    plot_3d=plot_3d,
                    output_dir=output_dir
                )
            except Exception as e:
                logger.error(f"生成 {p_type} 可视化时出错: {e}")
        
        print("\n聚类分析完成！")
        
        # 输出聚类统计信息
        print("\n聚类结果统计信息:")
        labels = clustering_result['labels']
        for i in range(n_clusters):
            cluster_count = (labels == i).sum()
            cluster_percent = cluster_count / len(labels) * 100
            print(f"聚类 #{i+1}: {cluster_count} 个样本 ({cluster_percent:.2f}%)")
        
        if 'silhouette_score' in clustering_result:
            print(f"\n轮廓分数: {clustering_result['silhouette_score']:.4f}")
            
        return clustering_result
        
    except ImportError:
        logger.error("聚类分析功能导入失败，请确认相关依赖是否安装")
        print("聚类分析功能导入失败，请确认是否安装了sklearn和matplotlib等依赖库")
    except Exception as e:
        logger.error(f"聚类分析失败: {e}")
        print(f"聚类分析失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def manage_storage(args):
    """
    管理存储功能
    
    Args:
        args: 命令行参数
    """
    storage = init_storage()
    
    if args.status:
        try:
            logger.info("获取存储状态信息...")
            # 获取数据库文件大小
            db_path = storage.db_path if hasattr(storage, 'db_path') else "未知"
            
            if os.path.exists(db_path):
                db_size = os.path.getsize(db_path) / (1024 * 1024)  # 转换为MB
                
                # 获取表数量
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                # 获取各类表的数量
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'historical_stock_%'")
                hist_stock_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'historical_index_%'")
                hist_index_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'realtime_%'")
                realtime_count = cursor.fetchone()[0]
                
                # 输出状态信息
                print("\n====== 存储状态信息 ======")
                print(f"数据库路径: {db_path}")
                print(f"数据库大小: {db_size:.2f} MB")
                print(f"表数量: {table_count}")
                print(f"  - 股票历史数据表: {hist_stock_count}")
                print(f"  - 指数历史数据表: {hist_index_count}")
                print(f"  - 实时数据表: {realtime_count}")
                print("==========================\n")
                
                conn.close()
            else:
                print(f"数据库文件不存在: {db_path}")
                
        except Exception as e:
            logger.error(f"获取存储状态失败: {e}")
            print(f"获取存储状态失败: {e}")
    
    elif args.optimize:
        try:
            logger.info("开始优化存储...")
            print("开始优化SQLite数据库...")
            
            # 使用存储对象的优化方法
            if hasattr(storage, '_optimize_database'):
                storage._optimize_database()
                print("基本数据库优化完成")
                
            # 调用索引优化方法
            if hasattr(storage, 'optimize_indexes'):
                print("开始优化数据库索引...")
                storage.optimize_indexes()
                print("索引优化完成")
                
            print("数据库优化完成！")
            
        except Exception as e:
            logger.error(f"优化存储失败: {e}")
            print(f"优化存储失败: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"程序执行过程中发生错误: {e}")
        sys.exit(1) 