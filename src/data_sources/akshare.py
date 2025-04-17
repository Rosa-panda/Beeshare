"""
AKShare数据源实现

流程图:
```mermaid
sequenceDiagram
    participant Client
    participant AKShareDS
    participant akshare库
    
    Client->>AKShareDS: get_historical_data(symbol, start_date, end_date)
    AKShareDS->>akshare库: stock_zh_a_hist(symbol, start_date, end_date)
    akshare库-->>AKShareDS: 返回历史数据
    AKShareDS-->>Client: 返回处理后的DataFrame
    
    Client->>AKShareDS: get_realtime_data(symbols)
    AKShareDS->>akshare库: stock_zh_a_spot
    akshare库-->>AKShareDS: 返回实时数据
    AKShareDS-->>Client: 返回处理后的DataFrame
```
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import logging
import re
import time
from .base import DataSource
import numpy as np
import json
import traceback
from functools import wraps
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from src.utils.column_mapping import (
    StandardColumns, 
    standardize_columns, 
    detect_and_log_column_issues
)
from config.config import CONFIG

logger = logging.getLogger(__name__)

def run_with_timeout(func, timeout=30, retry_count=3, retry_delay=2):
    """
    在指定的超时时间内运行函数，超时则抛出异常
    
    Args:
        func: 要运行的函数
        timeout: 超时时间(秒)
        retry_count: 重试次数
        retry_delay: 重试延迟(秒)
        
    Returns:
        函数的返回值
        
    Raises:
        TimeoutError: 超时时抛出
    """
    import concurrent.futures
    import time
    
    for attempt in range(retry_count):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logging.warning(f"操作超时(>{timeout}秒)，将进行第{attempt+1}/{retry_count}次重试")
            time.sleep(retry_delay)
        except Exception as e:
            logging.warning(f"操作失败: {e}，将进行第{attempt+1}/{retry_count}次重试")
            time.sleep(retry_delay)
    
    raise TimeoutError(f"操作在{retry_count}次尝试后仍然超时或失败")

def log_function_call(level='DEBUG'):
    """记录函数调用的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            level_num = getattr(logging, level)
            if logger.isEnabledFor(level_num):
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.log(level_num, f"调用 {func.__name__}({signature})")
            try:
                result = func(*args, **kwargs)
                if logger.isEnabledFor(level_num):
                    logger.log(level_num, f"{func.__name__} 返回: {type(result)}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} 抛出异常: {e}")
                raise
        return wrapper
    return decorator

class AKShareDS(DataSource):
    """
    AKShare数据源实现
    
    使用akshare库获取中国A股市场数据
    """
    
    def __init__(self, config=None):
        """
        初始化AKShare数据源
        
        Args:
            config (dict, optional): 配置信息. Defaults to None.
        """
        super().__init__("akshare", config)
        
        # 默认配置
        self.default_config = {
            'testing': True,  # 默认开启测试模式，使得在数据获取失败时返回模拟数据
            'mock_data_seed': 42,  # 随机数种子，用于生成模拟数据
            'retry_count': 3,  # API调用重试次数
            'retry_delay': 2,  # 重试间隔（秒）
            'timeout': 30,  # API调用超时时间（秒），增加到30秒
            'allow_partial': True,  # 允许部分功能可用
        }
        
        # 合并配置
        if config is None:
            self.config = self.default_config
        else:
            self.config = {**self.default_config, **config}
        
        # 输出当前配置信息    
        logger.info(f"AKShare数据源初始化，测试模式: {self.config.get('testing', False)}")
        
        # 初始化股票列表（备选）
        self.stock_code_name_df = pd.DataFrame()
            
        # 初始化连接
        logger.info("正在测试与AKShare的连接...")
        # 设置连接超时
        logger.info(f"连接超时设置为 {self.config.get('timeout', 30)} 秒")
        self.is_ready = self.check_connection()
        
        # 如果连接失败但允许部分功能，尝试加载本地股票列表
        if not self.is_ready and self.config.get('allow_partial'):
            try:
                # 尝试加载预定义的股票列表
                from .stock_list import STOCK_LIST
                temp_df = pd.DataFrame(STOCK_LIST, columns=['code', 'name'])
                if not temp_df.empty:
                    self.stock_code_name_df = temp_df
                    logger.info(f"从本地缓存加载 {len(self.stock_code_name_df)} 只A股股票信息")
            except Exception as e:
                logger.warning(f"加载本地股票列表失败: {e}")
                
            # 声明部分功能可用
            self.partial_ready = True
            logger.warning("AKShare连接失败，但已启用部分功能")
        else:
            self.partial_ready = self.is_ready
        
        # 尝试导入所有可能需要的AKShare函数
        try:
            # 检查AKShare版本
            akshare_version = ak.__version__
            logger.info(f"AKShare 版本: {akshare_version}")
            
            # 版本检查
            try:
                from pkg_resources import parse_version
                if parse_version(akshare_version) < parse_version('1.0.0'):
                    logger.warning(f"当前AKShare版本 {akshare_version} 可能过旧，建议升级到最新版本")
            except:
                pass
            
            # A股市场函数映射
            self.market_functions = {
                'historical': ak.stock_zh_a_hist,
                'realtime': ak.stock_zh_a_spot,
                'info': ak.stock_individual_info_em,
                'index': ak.stock_zh_index_daily,  # 指数日线数据
                'sh_spot': ak.stock_sh_a_spot_em,  # 上交所A股实时行情
                'sz_spot': ak.stock_sz_a_spot_em,  # 深交所A股实时行情
            }
            
            # 检查所有关键函数是否可用
            for func_name, func in self.market_functions.items():
                if func is None:
                    logger.warning(f"函数 {func_name} 不可用")
            
            # 记录测试模式状态
            if self.config['testing']:
                logger.info("数据源已启用测试模式，在API调用失败时将使用模拟数据")
            
        except Exception as e:
            logger.error(f"初始化AKShare函数映射失败: {e}")
            self.market_functions = {}
        
        # 初始化股票编码映射（用于股票代码判断和前缀添加）
        if not self.stock_code_name_df.empty:
            pass  # 已经在前面初始化过了
        else:
            try:
                self.stock_code_name_df = ak.stock_info_a_code_name()
                logger.info(f"成功加载 {len(self.stock_code_name_df)} 只A股股票信息")
            except Exception as e:
                logger.error(f"加载A股股票列表失败: {e}")
                self.stock_code_name_df = pd.DataFrame()
    
    def check_connection(self):
        """
        检查与AKShare的连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 从配置中获取超时和重试设置
            max_timeout = self.config.get('timeout', 30)  # 增加默认超时时间至30秒
            retry_count = self.config.get('retry_count', 3) 
            retry_delay = self.config.get('retry_delay', 2)
            
            import sys
            import threading
            import time
            
            # 使用更轻量级的函数测试连接
            def get_lightweight_test():
                try:
                    # 尝试几种轻量级API调用，按优先级排序
                    try:
                        # 只获取一只股票的信息，比获取全部列表更快
                        test_data = ak.stock_individual_info_em(symbol="000001")
                        return not test_data.empty if isinstance(test_data, pd.DataFrame) else test_data is not None
                    except:
                        # 如果上面的方法失败，尝试获取股票数量信息
                        try:
                            stock_list = ak.stock_info_a_code_name()
                            return not stock_list.empty if isinstance(stock_list, pd.DataFrame) else stock_list is not None
                        except:
                            # 最后尝试获取交易日期信息
                            try:
                                trading_dates = ak.tool_trade_date_hist_sina()
                                return not trading_dates.empty if isinstance(trading_dates, pd.DataFrame) else trading_dates is not None
                            except:
                                return False
                except Exception as e:
                    logger.debug(f"轻量级测试连接失败: {e}")
                    return False
            
            # 实现重试机制
            for attempt in range(retry_count):
                if attempt > 0:
                    logger.info(f"尝试第 {attempt+1}/{retry_count} 次连接AKShare...")
                    time.sleep(retry_delay)  # 重试前等待
                
                # Windows平台使用线程和Event实现超时
                if sys.platform.startswith('win'):
                    result = [None]
                    exception = [None]
                    event = threading.Event()
                    
                    def test_connection():
                        try:
                            # 使用轻量级连接测试
                            result[0] = get_lightweight_test()
                        except Exception as e:
                            exception[0] = e
                        finally:
                            event.set()
                    
                    # 创建并启动线程
                    thread = threading.Thread(target=test_connection)
                    thread.daemon = True
                    thread.start()
                    
                    # 等待线程完成或超时，根据尝试次数动态增加超时时间
                    current_timeout = min(max_timeout, 10 + attempt * 5)  # 第一次10秒，然后逐步增加
                    logger.info(f"连接超时设置为 {current_timeout} 秒")
                    
                    if not event.wait(timeout=current_timeout):
                        logger.warning(f"连接AKShare超时 (超时设置: {current_timeout}秒)")
                        continue  # 尝试下一次重试
                    
                    # 检查是否有异常
                    if exception[0]:
                        logger.warning(f"连接AKShare出现异常: {exception[0]}")
                        continue  # 尝试下一次重试
                    
                    if result[0]:
                        logger.info("AKShare连接测试成功")
                        return True
                    else:
                        logger.warning("AKShare连接测试失败，返回空结果")
                        continue  # 尝试下一次重试
                else:
                    # 非Windows平台使用信号实现超时
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"连接AKShare超时 (超时设置: {current_timeout}秒)")
                    
                    # 设置动态超时时间
                    current_timeout = min(max_timeout, 10 + attempt * 5)
                    logger.info(f"连接超时设置为 {current_timeout} 秒")
                    
                    try:
                        # 设置超时信号
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(current_timeout)
                        
                        # 使用轻量级连接测试
                        connection_result = get_lightweight_test()
                        
                        # 关闭超时
                        signal.alarm(0)
                        
                        if connection_result:
                            logger.info("AKShare连接测试成功")
                            return True
                        else:
                            logger.warning("AKShare连接测试失败，返回空结果")
                            continue  # 尝试下一次重试
                    except TimeoutError as e:
                        logger.warning(f"{e}")
                        continue  # 尝试下一次重试
                    except Exception as e:
                        logger.warning(f"连接AKShare出现异常: {e}")
                        continue  # 尝试下一次重试
                    finally:
                        # 确保关闭超时
                        signal.alarm(0)
            
            # 所有重试都失败
            logger.error(f"经过 {retry_count} 次尝试后，连接AKShare仍然失败")
            return False
            
        except Exception as e:
            logger.error(f"连接AKShare失败: {e}")
            return False
    
    def _normalize_symbol(self, symbol):
        """
        将股票代码标准化为AKShare需要的格式
        
        Args:
            symbol (str): 原始股票代码
        
        Returns:
            str: 标准化后的股票代码
        """
        # 如果输入是None或空字符串，直接返回
        if not symbol:
            return symbol
            
        # 移除可能的后缀(.SH, .SZ, .BJ等)
        symbol = re.sub(r'\.\w+$', '', symbol)
        
        # 确保保留前导零
        # 如果是纯数字且长度小于6，则检查是否需要添加前导零
        if symbol.isdigit():
            # 深交所主板股票处理 (001、002、003开头)
            if len(symbol) == 4 and symbol[0] == '1':  # 如1279 -> 001279
                symbol = f"001{symbol[1:]}"
                logger.info(f"股票代码修正: 添加前导零 -> {symbol}")
            elif len(symbol) == 5 and symbol.startswith('01'):  # 如01279 -> 001279
                symbol = f"0{symbol}"
                logger.info(f"股票代码修正: 添加前导零 -> {symbol}")
            elif len(symbol) == 4 and symbol[0] == '2':  # 如2001 -> 002001
                symbol = f"002{symbol[1:]}"
                logger.info(f"股票代码修正: 添加前导零 -> {symbol}")
            elif len(symbol) == 4 and symbol[0] == '3':  # 如3001 -> 003001 或 300开头
                if int(symbol) < 3100:  # 小于3100认为是003开头
                    symbol = f"003{symbol[1:]}"
                else:  # 否则认为是300开头
                    symbol = f"300{symbol[1:]}"
                logger.info(f"股票代码修正: 添加前导零 -> {symbol}")
            elif len(symbol) == 3 and symbol[0] in ['0', '1', '2', '3']:  # 如123 -> 000123
                symbol = f"000{symbol}"
                logger.info(f"股票代码修正: 添加前导零 -> {symbol}")
            
            # 保证股票代码为6位
            symbol = symbol.zfill(6)
        
        # 添加强制检查，确保名称为001279的股票代码正确
        if symbol == '1279':
            symbol = '001279'
            logger.info(f"特殊处理: 将1279修正为001279")
        
        logger.info(f"标准化股票代码: 输入 {symbol} -> 输出 {symbol}")
        return symbol
    
    def _is_index(self, symbol):
        """
        判断是否为指数代码
        
        Args:
            symbol (str): 股票代码
        
        Returns:
            bool: 是否为指数
        """
        # 主要指数代码检测
        # 000001(上证指数)、399001(深证成指)、000300(沪深300)、000016(上证50)、000905(中证500)
        index_codes = ['000001', '399001', '000300', '000016', '000905']
        return symbol in index_codes
    
    def _identify_stock_type(self, symbol):
        """
        判断股票类型（主板、创业板、科创板等）
        
        Args:
            symbol (str): 股票代码
        
        Returns:
            dict: 包含股票类型信息的字典
        """
        norm_symbol = self._normalize_symbol(symbol)
        
        # 初始化返回信息
        stock_info = {
            'market': '未知',
            'exchange': '未知',
            'board': '未知',
            'prefix': '',
        }
        
        # 上海证券交易所
        if norm_symbol.startswith('6'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '上海证券交易所'
            stock_info['board'] = '主板'
            stock_info['prefix'] = 'sh'
            
        # 上海科创板
        elif norm_symbol.startswith('688'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '上海证券交易所'
            stock_info['board'] = '科创板'
            stock_info['prefix'] = 'sh'
            
        # 深圳证券交易所主板
        elif norm_symbol.startswith('000'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '深圳证券交易所'
            stock_info['board'] = '主板'
            stock_info['prefix'] = 'sz'
            
        # 深圳创业板
        elif norm_symbol.startswith('300'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '深圳证券交易所'
            stock_info['board'] = '创业板'
            stock_info['prefix'] = 'sz'
        
        # 深圳中小板（已合并入主板，保留兼容）
        elif norm_symbol.startswith('002'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '深圳证券交易所'
            stock_info['board'] = '主板(原中小板)'
            stock_info['prefix'] = 'sz'
        
        # 新增：深圳主板其他代码段（001、003）
        elif norm_symbol.startswith('001') or norm_symbol.startswith('003'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '深圳证券交易所'
            stock_info['board'] = '主板'
            stock_info['prefix'] = 'sz'
        
        # 北交所
        elif norm_symbol.startswith('4') or norm_symbol.startswith('8') and not norm_symbol.startswith('688'):
            stock_info['market'] = 'A股'
            stock_info['exchange'] = '北京证券交易所'
            stock_info['board'] = '主板'
            stock_info['prefix'] = 'bj'
        
        # 检查股票是否存在于股票列表中
        if not self.stock_code_name_df.empty:
            matched = self.stock_code_name_df[self.stock_code_name_df['code'] == norm_symbol]
            if not matched.empty:
                stock_info['name'] = matched.iloc[0]['name']
                stock_info['exists'] = True
            else:
                stock_info['exists'] = False
                logger.warning(f"股票代码 {norm_symbol} 未在A股列表中找到")
        
        logger.info(f"股票 {norm_symbol} 分析结果: {stock_info}")
        return stock_info
    
    def _get_index_symbol_with_prefix(self, symbol):
        """
        获取带前缀的指数代码
        
        Args:
            symbol (str): 原始指数代码
        
        Returns:
            str: 带前缀的指数代码
        """
        # 根据指数代码规则添加前缀
        if symbol.startswith('000') or symbol == '000905' or symbol == '000016' or symbol == '000300':
            return f"sh{symbol}"
        elif symbol.startswith('399'):
            return f"sz{symbol}"
        return symbol
        
    def _get_stock_symbol_with_prefix(self, symbol):
        """
        为股票代码添加交易所前缀
        
        Args:
            symbol (str): 原始股票代码
        
        Returns:
            str: 带前缀的股票代码
        """
        norm_symbol = self._normalize_symbol(symbol)
        stock_info = self._identify_stock_type(norm_symbol)
        
        if stock_info['prefix']:
            return f"{stock_info['prefix']}{norm_symbol}"
        
        return norm_symbol
    
    def get_historical_data(self, symbol, start_date, end_date, interval=None, force_stock=False):
        """
        获取股票或指数的历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期, 格式: YYYY-MM-DD
            end_date: 结束日期, 格式: YYYY-MM-DD
            interval: 数据间隔，支持的值：'daily'(日), 'weekly'(周), 'monthly'(月)，默认为'daily'
            force_stock: 强制作为股票处理，即使识别为指数
        
        Returns:
            DataFrame: 包含历史数据的DataFrame
        """
        # 参数检查
        if not symbol:
            logger.error("股票代码不能为空")
            return pd.DataFrame()
            
        # 标准化参数
        symbol = self._normalize_symbol(symbol) if symbol else ''
        
        # 转换interval
        if interval is None:
            interval = 'daily'
        elif interval == '1d':
            interval = 'daily'
        elif interval == '1wk':
            interval = 'weekly'
        elif interval == '1mo':
            interval = 'monthly'
        
        # 记录请求信息
        logger.info(f"获取 {symbol} 的历史数据，时间区间: {start_date} 至 {end_date}，数据间隔: {interval}")
        
        # 识别股票类型
        stock_info = self._identify_stock_type(symbol)
        logger.info(f"股票 {symbol} 类型信息: {stock_info}")
        
        # 如果是指数且不强制作为股票处理，使用指数数据接口
        if self._is_index(symbol) and not force_stock:
            logger.info(f"{symbol} 被识别为指数，使用指数数据接口")
            df = self.get_index_data(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                return df
            else:
                logger.warning(f"通过指数接口无法获取 {symbol} 数据，尝试作为股票获取")
        
        # 方法链：尝试所有可能的方法获取数据
        # 1. 首先尝试特定交易所专用接口
        try:
            # 根据交易所选择不同的方法
            if stock_info.get('exchange') == '上海证券交易所':
                logger.info(f"尝试使用 get_sh_hist 获取 {symbol} 历史数据")
                df = self.get_sh_hist(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    return df
            elif stock_info.get('exchange') == '深圳证券交易所':
                logger.info(f"尝试使用 get_sz_hist 获取 {symbol} 历史数据")
                df = self.get_sz_hist(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    return df
            # 其他交易所...
        except Exception as e:
            logger.warning(f"使用交易所专用接口获取数据失败: {e}")
        
        # 2. 其次尝试标准API
        try:
            logger.info(f"尝试使用 _get_stock_data_via_standard_api 获取 {symbol} 历史数据")
            df = self._get_stock_data_via_standard_api(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"使用标准API获取数据失败: {e}")
        
        # 3. 最后尝试备用API或模拟数据
        try:
            logger.info(f"尝试使用 _get_stock_data_via_fallback_api 获取 {symbol} 历史数据")
            df = self._get_stock_data_via_fallback_api(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"使用备用API获取数据失败: {e}")
        
        # 所有方法都失败，尝试使用测试模式
        if self.config.get('testing', False):
            logger.warning(f"所有API方法获取 {symbol} 数据失败，使用测试模式返回模拟数据")
            return self._get_mock_data(symbol, start_date, end_date, interval)
        
        # 如果测试模式未启用，返回空DataFrame
        logger.error(f"无法获取 {symbol} 的历史数据，且测试模式未启用")
        return pd.DataFrame()
    
    def _get_stock_data_via_standard_api(self, symbol, start_date, end_date, interval):
        """使用标准AKShare API获取股票数据"""
        # 清理参数
        if start_date is None:
            clean_start_date = ''
        else:
            clean_start_date = start_date.replace('-', '')
            
        if end_date is None:
            clean_end_date = ''
        else:
            clean_end_date = end_date.replace('-', '')
            
        period = self.get_period_from_interval(interval)
        
        # 处理股票代码，添加正确的前缀
        prefixed_symbol = self._get_stock_symbol_with_prefix(symbol)
        
        # 记录详细的参数信息用于调试
        logger.info(f"调用AKShare API获取股票数据, 代码: {prefixed_symbol}, 期间: {period}, 开始日期: {clean_start_date}, 结束日期: {clean_end_date}")
        
        try:
            # 获取股票信息
            stock_info = self._identify_stock_type(symbol)
            
            # 修改：根据AKShare 1.16.78版本的API要求调整参数
            # 注意：最新版可能期望使用stock_zh_a_hist_163，或东财的stock_zh_a_hist_em
            try:
                # 尝试方法1：使用东方财富网接口 - AKShare 1.16.78中推荐的方法
                logger.info(f"尝试使用东方财富网接口(em)获取 {symbol} 历史数据")
                df = ak.stock_zh_a_hist(
                    symbol=prefixed_symbol,
                    period=period,
                    start_date=clean_start_date,
                    end_date=clean_end_date,
                    adjust="qfq"  # 前复权
                )
            except Exception as e:
                logger.warning(f"东方财富网接口获取数据失败: {e}，尝试备用接口")
                try:
                    # 尝试方法2: 使用网易财经接口 (163)
                    logger.info(f"尝试使用网易财经接口(163)获取 {symbol} 历史数据")
                    df = ak.stock_zh_a_hist_163(
                        symbol=prefixed_symbol.replace("sh", "0").replace("sz", "1"),  # 网易接口使用0/1作为前缀
                        start_date=clean_start_date,
                        end_date=clean_end_date,
                        adjust="qfq"  # 前复权
                    )
                except Exception as e2:
                    logger.warning(f"网易财经接口获取数据失败: {e2}，尝试另一备用接口")
                    try:
                        # 尝试方法3: 使用腾讯财经接口
                        logger.info(f"尝试使用腾讯财经接口获取 {symbol} 历史数据")
                        df = ak.stock_zh_a_hist_tx(
                            symbol=prefixed_symbol,
                            start_date=clean_start_date,
                            end_date=clean_end_date
                        )
                    except Exception as e3:
                        logger.warning(f"腾讯财经接口获取数据失败: {e3}，尝试不使用日期参数")
                        
                        # 尝试方法4: 不使用日期参数
                        try:
                            logger.info(f"尝试不使用日期参数获取 {symbol} 历史数据")
                            df = ak.stock_zh_a_hist(
                                symbol=prefixed_symbol,
                                period=period,
                                adjust="qfq"
                            )
                        except Exception as e4:
                            logger.error(f"所有标准API方法尝试失败: {e4}")
                            raise Exception(f"无法通过标准API获取 {symbol} 的历史数据")
            
            if df is not None and not df.empty:
                logger.info(f"成功通过标准API获取 {symbol} 数据, 共 {len(df)} 行")
                
                # 记录原始列名
                logger.debug(f"原始数据列: {list(df.columns)}")
                
                # 添加metadata
                df['symbol'] = symbol
                df['source'] = 'akshare'
                
                return df
            else:
                logger.warning("标准API返回空数据集，请检查股票代码、日期范围或网络连接")
                # 如果开启了测试模式，则返回模拟数据
                if self.config.get('testing', False):
                    logger.info("使用测试模式生成模拟数据")
                    return self._get_mock_data(symbol, start_date, end_date, interval)
                
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"标准API获取数据失败: {e}")
            logger.debug(f"异常详情: {traceback.format_exc()}")
            
            # 如果开启了测试模式，则返回模拟数据
            if self.config.get('testing', False):
                logger.info("API调用失败，使用测试模式生成模拟数据")
                return self._get_mock_data(symbol, start_date, end_date, interval)
            
            raise

    def _get_stock_data_via_fallback_api(self, symbol, start_date, end_date, interval):
        """使用备用AKShare API获取股票数据，如果无法获取则生成模拟数据"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import traceback
        
        logger.warning(f"使用备用方法获取 {symbol} 历史数据")
        
        # 标准化参数
        if start_date:
            clean_start_date = start_date.replace('-', '')
        else:
            clean_start_date = None
            
        if end_date:
            clean_end_date = end_date.replace('-', '')
        else:
            clean_end_date = None
        
        # 获取股票信息
        norm_symbol = self._normalize_symbol(symbol)
        stock_info = self._identify_stock_type(norm_symbol)
        
        # 构建股票代码格式
        if stock_info.get('exchange') == '上海证券交易所':
            prefix = 'sh'
        elif stock_info.get('exchange') == '深圳证券交易所':
            prefix = 'sz'
        else:
            prefix = ''
        
        prefixed_symbol = f"{prefix}{norm_symbol}"
        
        # 备用方法1：尝试使用新浪财经接口 - 在AKShare 1.16.78中更可靠
        try:
            logger.info(f"尝试使用新浪财经接口获取数据: {norm_symbol}")
            
            # 在AKShare 1.16.78中，sina接口更名为stock_zh_a_daily
            try:
                # 新版AKShare的接口名
                df_sina = ak.stock_zh_a_daily(symbol=prefixed_symbol, adjust="qfq")
            except Exception as e1:
                logger.warning(f"新版新浪接口调用失败: {e1}，尝试备用方法")
                try:
                    # 备用接口: 分钟级别可以获取日级别汇总
                    df_sina = ak.stock_zh_a_minute(symbol=prefixed_symbol, period="daily", adjust="qfq")
                except Exception as e2:
                    logger.warning(f"分钟级别接口调用失败: {e2}，尝试旧名称接口")
                    try:
                        # 旧版接口名可能
                        df_sina = ak.stock_zh_a_hist_min_em(symbol=prefixed_symbol, start_date=clean_start_date, end_date=clean_end_date, period='daily', adjust='qfq')
                    except Exception as e3:
                        logger.warning(f"所有新浪接口调用失败: {e3}")
                        raise
                
            if df_sina is not None and not df_sina.empty:
                # 对日期范围进行过滤，因为新浪接口可能不支持直接传入日期参数
                if start_date or end_date:
                    # 确保日期列是datetime类型
                    if 'date' in df_sina.columns:
                        df_sina['date'] = pd.to_datetime(df_sina['date'])
                    elif '日期' in df_sina.columns:
                        df_sina['date'] = pd.to_datetime(df_sina['日期'])
                        df_sina = df_sina.rename(columns={'日期': 'date'})
                    
                    # 应用日期过滤
                    if start_date:
                        start_datetime = pd.to_datetime(start_date)
                        df_sina = df_sina[df_sina['date'] >= start_datetime]
                    
                    if end_date:
                        end_datetime = pd.to_datetime(end_date)
                        df_sina = df_sina[df_sina['date'] <= end_datetime]
                
                if not df_sina.empty:
                    logger.info(f"成功通过新浪财经接口获取 {symbol} 数据，共 {len(df_sina)} 行")
                    df_sina['symbol'] = symbol
                    df_sina['source'] = 'akshare_sina'
                    return df_sina
            else:
                logger.warning(f"新浪财经接口返回空数据: {norm_symbol}")
        except Exception as e:
            logger.warning(f"新浪财经接口获取数据失败: {e}")
        
        # 备用方法2：尝试使用东方财富网接口
        try:
            logger.info(f"尝试使用东方财富网接口获取数据: {norm_symbol}")
            
            # 尝试不同的东方财富接口函数
            try:
                df_east = ak.stock_zh_a_hist(symbol=prefixed_symbol, period=self.get_period_from_interval(interval), start_date=clean_start_date, end_date=clean_end_date, adjust="qfq")
            except Exception as e1:
                logger.warning(f"东方财富主接口调用失败: {e1}，尝试备用方法")
                try:
                    df_east = ak.stock_zh_a_hist_min_em(symbol=prefixed_symbol, start_date=clean_start_date, end_date=clean_end_date, period='daily', adjust='qfq')
                except Exception as e2:
                    logger.warning(f"东方财富分钟接口调用失败: {e2}，尝试另一个备用方法")
                    df_east = pd.DataFrame()  # 初始化为空DataFrame
            
            if df_east is not None and not df_east.empty:
                logger.info(f"成功通过东方财富接口获取 {symbol} 数据，共 {len(df_east)} 行")
                df_east['symbol'] = symbol
                df_east['source'] = 'akshare_east'
                return df_east
        except Exception as e:
            logger.warning(f"东方财富网接口获取数据失败: {e}")
        
        # 备用方法3：尝试获取指数数据（如果可能是指数）
        try:
            if self._is_index(symbol):
                logger.info(f"尝试作为指数获取 {symbol} 数据")
                index_df = self.get_index_data(symbol, start_date, end_date, interval)
                if index_df is not None and not index_df.empty:
                    logger.info(f"成功作为指数获取 {symbol} 数据，共 {len(index_df)} 行")
                    return index_df
        except Exception as e:
            logger.warning(f"作为指数获取数据失败: {e}")
        
        # 备用方法4：尝试使用网易财经接口
        try:
            logger.info(f"尝试使用网易财经接口获取数据: {norm_symbol}")
            # 网易接口使用不同的前缀格式:
            # 上交所股票代码前缀为0，深交所为1
            if prefix == 'sh':
                sina_prefix = '0'
            elif prefix == 'sz':
                sina_prefix = '1'
            else:
                sina_prefix = ''
            
            df_163 = ak.stock_zh_a_hist_163(symbol=f"{sina_prefix}{norm_symbol}", start_date=clean_start_date, end_date=clean_end_date, adjust="qfq")
            
            if df_163 is not None and not df_163.empty:
                logger.info(f"成功通过网易财经接口获取 {symbol} 数据，共 {len(df_163)} 行")
                df_163['symbol'] = symbol
                df_163['source'] = 'akshare_163'
                return df_163
        except Exception as e:
            logger.warning(f"网易财经接口获取数据失败: {e}")
        
        # 如果开启了测试模式，则返回模拟数据
        if self.config.get('testing', False):
            logger.warning(f"无法通过任何API获取 {symbol} 数据，使用模拟数据")
            return self._get_mock_data(symbol, start_date, end_date, interval)
        
        # 所有方法都失败，返回空DataFrame
        logger.error(f"无法获取 {symbol} 的历史数据，尝试了所有可用方法")
        return pd.DataFrame()
    
    def get_realtime_data(self, symbols):
        """
        获取实时股票数据
        
        Args:
            symbols (list): 股票代码列表
        
        Returns:
            pandas.DataFrame: 包含实时数据的DataFrame
        """
        if not symbols:
            logger.warning("未提供股票代码，无法获取实时数据")
            return pd.DataFrame()
        
        # 确保symbols是列表
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # 先获取所有A股实时行情，可能会比较慢
        logger.info("尝试获取A股所有实时行情数据...")
        try:
            # 获取所有A股实时行情
            all_stock_data = ak.stock_zh_a_spot()
            logger.info(f"成功获取A股实时行情，共 {len(all_stock_data)} 只股票")
        except Exception as e:
            logger.warning(f"获取A股全部实时行情失败: {e}")
            all_stock_data = pd.DataFrame()
        
        # 存储结果
        result_data = []
        
        # 对每只股票获取数据
        for symbol in symbols:
            # 标准化股票代码
            norm_symbol = symbol
            logger.info(f"标准化股票代码: 输入 {symbol} -> 输出 {norm_symbol}")
            
            # 获取股票类型信息
            stock_info = self._identify_stock_type(norm_symbol)
            logger.info(f"股票 {norm_symbol} 分析结果: {stock_info}")
            
            if not stock_info.get('exists', False):
                logger.warning(f"股票 {norm_symbol} 不存在，跳过获取")
                continue
            
            logger.info(f"获取 {norm_symbol} 实时数据，市场: {stock_info['exchange']}, 板块: {stock_info['board']}")
            
            # 再次检查stock_info，确保正确识别了股票
            if stock_info is None or not stock_info.get('exists', False):
                logger.warning(f"无法识别股票 {norm_symbol} 的类型，跳过")
                continue
            
            # 标准化股票代码
            norm_symbol = symbol
            logger.info(f"标准化股票代码: 输入 {symbol} -> 输出 {norm_symbol}")
            
            # 获取股票类型信息
            stock_info = self._identify_stock_type(norm_symbol)
            logger.info(f"股票 {norm_symbol} 分析结果: {stock_info}")
            
            # 方法1: 从全部行情中过滤
            if not all_stock_data.empty:
                spot_data = all_stock_data[all_stock_data['代码'] == norm_symbol]
                if not spot_data.empty:
                    row = spot_data.iloc[0]
                    
                    realtime_data = {
                        'symbol': norm_symbol,
                        'timestamp': datetime.now(),
                        'price': row.get('最新价', None),
                        'change': row.get('涨跌额', None),
                        'change_pct': row.get('涨跌幅', None),  # 使用change_pct字段名
                        'volume': row.get('成交量', None),
                        'amount': row.get('成交额', None),
                        'high': row.get('最高', None),
                        'low': row.get('最低', None),
                        'open': row.get('开盘', None),
                        'prev_close': row.get('昨收', None),
                        'bid_price': row.get('买入', None),    # 添加缺失字段
                        'ask_price': row.get('卖出', None),    # 添加缺失字段
                        'bid_volume': row.get('买量', 0),      # 添加缺失字段
                        'ask_volume': row.get('卖量', 0),      # 添加缺失字段
                        'source': self.name
                    }
                    
                    result_data.append(realtime_data)
                    logger.info(f"从全部行情中成功获取 {norm_symbol} 的实时数据")
                    continue  # 成功获取到数据，跳过后续方法
            
            # 方法2: 尝试使用单独的实时行情接口
            try:
                logger.info(f"尝试使用股票行情接口获取 {norm_symbol} 的实时数据...")
                # 注意：stock_zh_a_spot_em不接受symbol参数，需要先获取所有数据再过滤
                single_spot_data = ak.stock_zh_a_spot_em()  # 获取所有股票数据
                if not single_spot_data.empty:
                    # 过滤出当前股票的数据
                    single_spot_data = single_spot_data[single_spot_data['代码'] == norm_symbol]
                    
                    if not single_spot_data.empty:
                        row = single_spot_data.iloc[0]
                        
                        realtime_data = {
                            'symbol': norm_symbol,
                            'timestamp': datetime.now(),
                            'price': row.get('最新价', None),
                            'change': row.get('涨跌额', None),
                            'change_pct': row.get('涨跌幅', None),  # 将涨跌幅直接映射为change_pct字段
                            'volume': row.get('成交量', None),
                            'amount': row.get('成交额', None),
                            'high': row.get('最高', None),
                            'low': row.get('最低', None),
                            'open': row.get('开盘', None),
                            'prev_close': row.get('昨收', None),
                            'bid_price': row.get('买入', None),    # 添加缺失字段
                            'ask_price': row.get('卖出', None),    # 添加缺失字段
                            'bid_volume': row.get('买量', 0),      # 添加缺失字段
                            'ask_volume': row.get('卖量', 0),      # 添加缺失字段
                            'source': self.name
                        }
                        
                        result_data.append(realtime_data)
                        logger.info(f"通过单独接口成功获取 {norm_symbol} 的实时数据")
                        continue  # 成功获取到数据，跳过后续方法
            except Exception as e:
                logger.warning(f"使用单独股票行情接口获取 {norm_symbol} 数据失败: {e}")
            
            # 方法3: 尝试使用历史数据接口获取当日数据
            try:
                logger.info(f"尝试使用历史数据接口获取 {norm_symbol} 的当日数据...")
                today = datetime.now().strftime('%Y-%m-%d')
                hist_data = self.get_historical_data(
                    symbol=norm_symbol,
                    start_date=today,
                    end_date=today,
                    interval='daily'
                )
                
                if not hist_data.empty:
                    row = hist_data.iloc[0]
                    
                    realtime_data = {
                        'symbol': norm_symbol,
                        'timestamp': datetime.now(),
                        'price': row.get('close', None),
                        'change': None,  # 历史数据通常没有涨跌额
                        'change_pct': row.get('change_pct', None),  # 使用与存储模块匹配的字段名
                        'volume': row.get('volume', None),
                        'amount': row.get('amount', None),
                        'high': row.get('high', None),
                        'low': row.get('low', None),
                        'open': row.get('open', None),
                        'prev_close': None,  # 历史数据通常没有昨收价
                        'bid_price': None,    # 历史数据没有这些字段，设为null
                        'ask_price': None,
                        'bid_volume': 0,
                        'ask_volume': 0,
                        'source': self.name
                    }
                    
                    result_data.append(realtime_data)
                    logger.info(f"通过历史数据接口成功获取 {norm_symbol} 的当日数据")
                    continue  # 成功获取到数据，跳过后续方法
            except Exception as e:
                logger.warning(f"使用历史数据接口获取 {norm_symbol} 当日数据失败: {e}")
            
            # 方法4: 尝试使用股票最新行情API
            try:
                logger.info(f"尝试使用特定股票代码接口获取 {norm_symbol} 的行情数据...")
                # 使用AKShare的其他API尝试获取数据
                # 根据前缀选择不同交易所的接口
                if stock_info['prefix'] == 'sh':
                    # 上海证券交易所
                    specific_data = ak.stock_sh_a_spot_em()
                    specific_data = specific_data[specific_data['代码'] == norm_symbol]
                elif stock_info['prefix'] == 'sz':
                    # 深圳证券交易所
                    specific_data = ak.stock_sz_a_spot_em()
                    specific_data = specific_data[specific_data['代码'] == norm_symbol]
                else:
                    specific_data = pd.DataFrame()
                    
                if not specific_data.empty:
                    row = specific_data.iloc[0]
                    
                    realtime_data = {
                        'symbol': norm_symbol,
                        'timestamp': datetime.now(),
                        'price': row.get('最新价', None),
                        'change': row.get('涨跌额', None),
                        'change_pct': row.get('涨跌幅', None),  # 使用change_pct字段名替代change_percent
                        'volume': row.get('成交量', None),
                        'amount': row.get('成交额', None),
                        'high': row.get('最高', None),
                        'low': row.get('最低', None),
                        'open': row.get('开盘', None),
                        'prev_close': row.get('昨收', None),
                        'bid_price': row.get('买入', None),    # 添加缺失字段
                        'ask_price': row.get('卖出', None),    # 添加缺失字段
                        'bid_volume': row.get('买量', 0),      # 添加缺失字段
                        'ask_volume': row.get('卖量', 0),      # 添加缺失字段
                        'source': self.name
                    }
                    
                    result_data.append(realtime_data)
                    logger.info(f"通过特定交易所接口成功获取 {norm_symbol} 的行情数据")
                    continue  # 成功获取到数据，跳过后续方法
            except Exception as e:
                logger.warning(f"使用特定交易所接口获取 {norm_symbol} 数据失败: {e}")
            
            # 如果所有方法都失败，并且启用了测试模式，则生成模拟数据
            if self.config.get('testing', False):
                logger.warning(f"无法获取 {norm_symbol} 的实时数据，生成模拟数据")
                
                # 生成一些合理的模拟数据
                import random
                
                base_price = random.uniform(10, 100)
                change_pct_value = random.uniform(-0.05, 0.05)
                change = base_price * change_pct_value
                
                realtime_data = {
                    'symbol': norm_symbol,
                    'timestamp': datetime.now(),
                    'price': round(base_price, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct_value * 100, 2),  # 使用change_pct字段名替代change_percent
                    'volume': random.randint(10000, 1000000),
                    'amount': random.randint(1000000, 100000000),
                    'high': round(base_price * (1 + random.uniform(0, 0.02)), 2),
                    'low': round(base_price * (1 - random.uniform(0, 0.02)), 2),
                    'open': round(base_price * (1 - change_pct_value), 2),
                    'prev_close': round(base_price - change, 2),
                    'bid_price': round(base_price * 0.999, 2),    # 买入价略低于当前价
                    'ask_price': round(base_price * 1.001, 2),    # 卖出价略高于当前价
                    'bid_volume': random.randint(100, 10000),     # 随机买入量
                    'ask_volume': random.randint(100, 10000),     # 随机卖出量
                    'source': f"{self.name}_mock"
                }
                
                result_data.append(realtime_data)
                continue
            
            # 如果所有方法都失败，记录警告
            logger.warning(f"无法获取 {norm_symbol} 的实时数据，跳过")
        
        # 转换为DataFrame
        if result_data:
            result_df = pd.DataFrame(result_data)
            
            # 确保包含所有必要字段
            required_fields = ['symbol', 'timestamp', 'price', 'change', 'change_pct', 
                              'volume', 'amount', 'high', 'low', 'open', 'prev_close',
                              'bid_price', 'ask_price', 'bid_volume', 'ask_volume']
            
            for field in required_fields:
                if field not in result_df.columns:
                    if field in ['bid_volume', 'ask_volume']:
                        result_df[field] = 0  # 对于交易量字段，默认为0
                    else:
                        result_df[field] = None  # 对于其他字段，默认为空
            
            # 输出详细调试信息
            logger.info(f"结果DataFrame的列：{list(result_df.columns)}")
            for col in result_df.columns:
                logger.info(f"列 {col} 的前3个值: {result_df[col].head(3).tolist()}")
            
            logger.info(f"成功获取 {len(result_df)} 只股票的实时数据")
            return result_df
        else:
            logger.warning("未能获取任何实时数据")
            return pd.DataFrame()
    
    def search_symbols(self, keyword):
        """
        搜索股票代码
        
        Args:
            keyword (str): 搜索关键词
        
        Returns:
            list: 匹配的股票代码和名称列表
        """
        try:
            # 使用已加载的股票列表或重新获取
            if self.stock_code_name_df.empty:
                stock_list = ak.stock_info_a_code_name()
            else:
                stock_list = self.stock_code_name_df
            
            # 过滤匹配项
            matched = stock_list[
                (stock_list['code'].str.contains(keyword)) | 
                (stock_list['name'].str.contains(keyword))
            ]
            
            results = []
            for _, row in matched.iterrows():
                # 获取股票类型信息
                stock_info = self._identify_stock_type(row['code'])
                
                results.append({
                    'symbol': row['code'],
                    'name': row['name'],
                    'market': stock_info['market'],
                    'exchange': stock_info['exchange'],
                    'board': stock_info['board'],
                    'source': self.name
                })
            
            return results
        except Exception as e:
            logger.error(f"搜索股票失败: {e}")
            return []
    
    def get_symbol_info(self, symbol):
        """
        获取股票的详细信息
        
        Args:
            symbol (str): 股票代码
        
        Returns:
            dict: 股票详细信息
        """
        # 检查数据源状态
        if not (self.is_ready or self.partial_ready):
            logger.error("AKShare数据源未就绪，请先检查连接")
            
            # 如果启用了测试模式，尝试从本地股票列表中获取信息
            if self.config.get('testing', True):
                # 从stock_list中获取
                try:
                    from .stock_list import STOCK_DICT
                    norm_symbol = self._normalize_symbol(symbol)
                    if norm_symbol in STOCK_DICT:
                        # 创建基本信息
                        stock_type_info = self._identify_stock_type(norm_symbol)
                        
                        # 添加名称
                        stock_type_info['name'] = STOCK_DICT[norm_symbol]
                        stock_type_info['symbol'] = norm_symbol
                        stock_type_info['source'] = self.name
                        
                        logger.info(f"从本地列表获取到股票 {norm_symbol} 的信息: {STOCK_DICT[norm_symbol]}")
                        return stock_type_info
                except Exception as e:
                    logger.warning(f"从本地列表获取股票信息失败: {e}")
            
            return {}
        
        # 标准化股票代码
        norm_symbol = self._normalize_symbol(symbol)
        
        try:
            # 获取股票类型信息
            stock_type_info = self._identify_stock_type(norm_symbol)
            
            # 添加交易所前缀
            stock_symbol = self._get_stock_symbol_with_prefix(norm_symbol)
            
            # 创建默认的股票信息字典
            symbol_info = {
                'symbol': norm_symbol,
                'name': '',  # 需要从其他API获取
                'market': stock_type_info['market'],
                'exchange': stock_type_info['exchange'],
                'board': stock_type_info['board'],
                'industry': '',
                'listing_date': '',
                'source': self.name
            }
            
            # 如果只支持部分功能，可能无法从API获取详细信息
            if not self.is_ready and self.partial_ready:
                # 尝试从本地股票列表获取名称
                try:
                    from .stock_list import STOCK_DICT
                    if norm_symbol in STOCK_DICT:
                        symbol_info['name'] = STOCK_DICT[norm_symbol]
                        logger.info(f"从本地列表获取到股票 {norm_symbol} 的名称: {STOCK_DICT[norm_symbol]}")
                except Exception as e:
                    logger.warning(f"从本地列表获取股票名称失败: {e}")
                
                # 直接返回有限的信息
                return symbol_info
            
            try:
                # 获取A股详细信息
                info = self.market_functions['info'](symbol=stock_symbol)
                
                # 确保info是有效的DataFrame
                if isinstance(info, pd.DataFrame) and not info.empty:
                    # 解析返回的信息表格
                    for _, row in info.iterrows():
                        if '项目' in info.columns and '值' in info.columns:
                            item_name = row.get('项目', '')
                            item_value = row.get('值', '')
                            
                            if '证券简称' in item_name:
                                symbol_info['name'] = item_value
                            elif '行业' in item_name:
                                symbol_info['industry'] = item_value
                            elif '上市日期' in item_name:
                                symbol_info['listing_date'] = item_value
                else:
                    logger.warning(f"获取{symbol}的详细信息返回了空数据或非DataFrame数据")
            except Exception as inner_e:
                logger.warning(f"解析{symbol}的详细信息时发生错误: {inner_e}")
            
            # 如果没有获取到名称，尝试从股票列表获取
            if not symbol_info['name']:
                try:
                    if hasattr(self, 'stock_code_name_df') and not getattr(self, 'stock_code_name_df', pd.DataFrame()).empty:
                        stock_list = self.stock_code_name_df
                    else:
                        stock_list = ak.stock_info_a_code_name()
                        
                    if isinstance(stock_list, pd.DataFrame) and not stock_list.empty:
                        matched = stock_list[stock_list['code'] == norm_symbol]
                        if not matched.empty:
                            symbol_info['name'] = matched.iloc[0]['name']
                except Exception as name_e:
                    logger.warning(f"获取{symbol}的名称时发生错误: {name_e}")
                    
                # 最后尝试从本地列表获取
                if not symbol_info['name']:
                    try:
                        from .stock_list import STOCK_DICT
                        if norm_symbol in STOCK_DICT:
                            symbol_info['name'] = STOCK_DICT[norm_symbol]
                            logger.info(f"从本地列表获取到股票 {norm_symbol} 的名称: {STOCK_DICT[norm_symbol]}")
                    except Exception as e:
                        logger.warning(f"从本地列表获取股票名称失败: {e}")
            
            return symbol_info
            
        except Exception as e:
            logger.error(f"获取{symbol}的详细信息失败: {e}")
            
            # 如果出错且启用了测试模式，返回基本信息
            if self.config.get('testing', True):
                # 尝试从本地股票列表获取信息
                try:
                    from .stock_list import STOCK_DICT
                    if norm_symbol in STOCK_DICT:
                        stock_type_info = self._identify_stock_type(norm_symbol)
                        stock_type_info['name'] = STOCK_DICT[norm_symbol]
                        stock_type_info['symbol'] = norm_symbol
                        stock_type_info['source'] = self.name
                        logger.info(f"从本地列表获取到股票 {norm_symbol} 的信息: {STOCK_DICT[norm_symbol]}")
                        return stock_type_info
                except Exception as loc_e:
                    logger.warning(f"从本地列表获取股票信息失败: {loc_e}")
            
            return {}
            
    def get_index_data(self, symbol, start_date=None, end_date=None, interval='daily'):
        """
        获取指数数据（如上证指数、深证成指等）
        
        Args:
            symbol (str): 指数代码，如'000001'表示上证指数
            start_date (str, optional): 开始日期，格式YYYY-MM-DD. Defaults to None.
            end_date (str, optional): 结束日期，格式YYYY-MM-DD. Defaults to None.
            interval (str, optional): 数据间隔，支持'daily', 'weekly', 'monthly'. Defaults to 'daily'.
        
        Returns:
            pandas.DataFrame: 包含指数历史数据的DataFrame
        """
        if not self.is_ready:
            logger.error("AKShare数据源未就绪，请先检查连接")
            return pd.DataFrame()
            
        logger.info(f"获取指数 {symbol} 的历史数据")
        
        try:
            # 直接调用get_historical_data，它内部会检测指数代码并使用正确的接口
            return self.get_historical_data(symbol, start_date, end_date, interval)
        except Exception as e:
            logger.error(f"获取指数 {symbol} 的数据失败: {e}")
            return pd.DataFrame()
    
    def check_date_format(self, date_str):
        """
        检查日期格式是否为YYYY-MM-DD
        
        Args:
            date_str (str): 日期字符串
            
        Returns:
            bool: 日期格式是否正确
        """
        # 使用正则表达式检查日期格式是否为YYYY-MM-DD
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        if not date_pattern.match(date_str):
            return False
            
        # 进一步检查日期是否有效
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
            
    def get_interval_value(self, interval):
        """
        获取间隔值，将用户输入的间隔值转换为AKShare需要的格式
        
        Args:
            interval (str): 用户输入的间隔值
            
        Returns:
            str: AKShare需要的间隔值格式
        """
        # 如果未指定间隔，使用默认值'daily'
        if not interval:
            return 'daily'
            
        # 定义间隔映射表
        interval_map = {
            # 日频数据
            'd': 'daily', 'day': 'daily', 'daily': 'daily', '1d': 'daily',
            # 周频数据
            'w': 'weekly', 'week': 'weekly', 'weekly': 'weekly', '1w': 'weekly', '1wk': 'weekly',
            # 月频数据
            'm': 'monthly', 'month': 'monthly', 'monthly': 'monthly', '1m': 'monthly', '1mo': 'monthly'
        }
        
        # 转换为小写并去除空格
        interval = interval.lower().strip()
        
        # 返回映射的值，如果没有匹配项则使用默认值'daily'
        return interval_map.get(interval, 'daily')
    
    def get_period_from_interval(self, interval):
        """
        根据间隔值获取AKShare所需的period参数
        
        Args:
            interval (str): 间隔值
            
        Returns:
            str: period参数值
        """
        # 将interval转换为AKShare所需的period参数
        interval_to_period = {
            'daily': 'daily',  # 日频数据
            'weekly': 'weekly',  # 周频数据
            'monthly': 'monthly'  # 月频数据
        }
        
        return interval_to_period.get(interval, 'daily')
        
    def add_market_prefix(self, symbol, stock_info):
        """
        为股票代码添加市场前缀
        
        Args:
            symbol (str): 股票代码
            stock_info (dict): 股票信息
            
        Returns:
            str: 带市场前缀的股票代码
        """
        # 确保股票代码是6位
        symbol = symbol.zfill(6)
        
        # 根据股票信息添加市场前缀
        if stock_info and 'prefix' in stock_info and stock_info['prefix']:
            return f"{stock_info['prefix']}{symbol}"
            
        # 如果没有前缀信息，根据股票代码首位判断
        if symbol.startswith(('0', '3')):
            return f"sz{symbol}"  # 深圳市场
        elif symbol.startswith(('6', '9')):
            return f"sh{symbol}"  # 上海市场
        elif symbol.startswith(('4', '8')) and not symbol.startswith('688'):
            return f"bj{symbol}"  # 北京市场
            
        # 默认返回原始代码
        return symbol
    
    def get_sh_hist(self, symbol, start_date, end_date, interval):
        """
        获取上海证券交易所股票历史数据
        
        Args:
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 数据间隔
            
        Returns:
            pandas.DataFrame: 历史数据
        """
        logging.info(f"使用上交所专用接口获取 {symbol} 历史数据")
        
        # 清理参数
        clean_start_date = start_date.replace('-', '')
        clean_end_date = end_date.replace('-', '')
        period = self.get_period_from_interval(interval)
        
        # 添加前缀
        prefixed_symbol = f"sh{symbol}"
        
        try:
            # 尝试使用AKShare API获取数据
            hist_data = ak.stock_zh_a_hist(
                symbol=prefixed_symbol, 
                period=period,
                start_date=clean_start_date,
                end_date=clean_end_date,
                adjust="qfq"
            )
            
            if hist_data is not None and not hist_data.empty:
                hist_data['symbol'] = symbol
                hist_data['source'] = 'akshare'
                return hist_data
            else:
                logging.warning(f"上交所接口未能获取到 {symbol} 历史数据")
                return None
        except Exception as e:
            logging.warning(f"使用上交所接口获取 {symbol} 数据时出错: {e}")
            return None
    
    def get_sz_hist(self, symbol, start_date, end_date, interval):
        """
        获取深圳证券交易所股票历史数据
        
        Args:
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 数据间隔
            
        Returns:
            pandas.DataFrame: 历史数据
        """
        logging.info(f"使用深交所专用接口获取 {symbol} 历史数据")
        
        # 清理参数
        clean_start_date = start_date.replace('-', '')
        clean_end_date = end_date.replace('-', '')
        period = self.get_period_from_interval(interval)
        
        # 添加前缀
        prefixed_symbol = f"sz{symbol}"
        
        try:
            # 尝试使用AKShare API获取数据
            hist_data = ak.stock_zh_a_hist(
                symbol=prefixed_symbol, 
                period=period,
                start_date=clean_start_date,
                end_date=clean_end_date,
                adjust="qfq"
            )
            
            if hist_data is not None and not hist_data.empty:
                hist_data['symbol'] = symbol
                hist_data['source'] = 'akshare'
                return hist_data
            else:
                logging.warning(f"深交所接口未能获取到 {symbol} 历史数据")
                return None
        except Exception as e:
            logging.warning(f"使用深交所接口获取 {symbol} 数据时出错: {e}")
            return None
    
    def get_kcb_hist(self, symbol, start_date, end_date, interval):
        """
        获取科创板股票历史数据
        
        Args:
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 数据间隔
            
        Returns:
            pandas.DataFrame: 历史数据
        """
        logging.info(f"使用科创板专用接口获取 {symbol} 历史数据")
        
        # 清理参数
        clean_start_date = start_date.replace('-', '')
        clean_end_date = end_date.replace('-', '')
        period = self.get_period_from_interval(interval)
        
        # 添加前缀
        prefixed_symbol = f"sh{symbol}"
        
        try:
            # 尝试使用AKShare API获取数据
            hist_data = ak.stock_zh_a_hist(
                symbol=prefixed_symbol, 
                period=period,
                start_date=clean_start_date,
                end_date=clean_end_date,
                adjust="qfq"
            )
            
            if hist_data is not None and not hist_data.empty:
                hist_data['symbol'] = symbol
                hist_data['source'] = 'akshare'
                return hist_data
            else:
                logging.warning(f"科创板接口未能获取到 {symbol} 历史数据")
                return None
        except Exception as e:
            logging.warning(f"使用科创板接口获取 {symbol} 数据时出错: {e}")
            return None
    
    def get_cyb_hist(self, symbol, start_date, end_date, interval):
        """
        获取创业板股票历史数据
        
        Args:
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 数据间隔
            
        Returns:
            pandas.DataFrame: 历史数据
        """
        logging.info(f"使用创业板专用接口获取 {symbol} 历史数据")
        
        # 清理参数
        clean_start_date = start_date.replace('-', '')
        clean_end_date = end_date.replace('-', '')
        period = self.get_period_from_interval(interval)
        
        # 添加前缀
        prefixed_symbol = f"sz{symbol}"
        
        try:
            # 尝试使用AKShare API获取数据
            hist_data = ak.stock_zh_a_hist(
                symbol=prefixed_symbol, 
                period=period,
                start_date=clean_start_date,
                end_date=clean_end_date,
                adjust="qfq"
            )
            
            if hist_data is not None and not hist_data.empty:
                hist_data['symbol'] = symbol
                hist_data['source'] = 'akshare'
                return hist_data
            else:
                logging.warning(f"创业板接口未能获取到 {symbol} 历史数据")
                return None
        except Exception as e:
            logging.warning(f"使用创业板接口获取 {symbol} 数据时出错: {e}")
            return None
    
    def get_bj_hist(self, symbol, start_date, end_date, interval):
        """
        获取北交所股票历史数据
        
        Args:
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 数据间隔
            
        Returns:
            pandas.DataFrame: 历史数据
        """
        logging.info(f"使用北交所专用接口获取 {symbol} 历史数据")
        
        # 清理参数
        clean_start_date = start_date.replace('-', '')
        clean_end_date = end_date.replace('-', '')
        period = self.get_period_from_interval(interval)
        
        # 添加前缀
        prefixed_symbol = f"bj{symbol}"
        
        try:
            # 尝试使用AKShare API获取数据
            hist_data = ak.stock_zh_a_hist(
                symbol=prefixed_symbol, 
                period=period,
                start_date=clean_start_date,
                end_date=clean_end_date,
                adjust="qfq"
            )
            
            if hist_data is not None and not hist_data.empty:
                hist_data['symbol'] = symbol
                hist_data['source'] = 'akshare'
                return hist_data
            else:
                logging.warning(f"北交所接口未能获取到 {symbol} 历史数据")
                return None
        except Exception as e:
            logging.warning(f"使用北交所接口获取 {symbol} 数据时出错: {e}")
            return None
    
    def standardize_code(self, symbol):
        """
        将股票代码标准化为系统需要的格式（公开方法）
        
        Args:
            symbol (str): 原始股票代码
        
        Returns:
            str: 标准化后的股票代码
        """
        # 调用内部的标准化方法
        return self._normalize_symbol(symbol) 
    
    def _get_mock_data(self, symbol, start_date, end_date, interval=None):
        """
        生成模拟股票数据，仅用于测试目的
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔，支持的值：'daily'(日), 'weekly'(周), 'monthly'(月)，默认为'daily'
            
        Returns:
            DataFrame: 包含模拟历史数据的DataFrame
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.warning(f"生成 {symbol} 的模拟数据")
        
        # 生成日期范围
        if start_date is None:
            start = datetime.now() - timedelta(days=90)
        else:
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d')
            except:
                start = datetime.now() - timedelta(days=90)
        
        if end_date is None:
            end = datetime.now()
        else:
            try:
                end = datetime.strptime(end_date, '%Y-%m-%d')
            except:
                end = datetime.now()
        
        # 创建日期列表
        date_range = []
        current = start
        
        # 根据间隔调整日期生成逻辑
        if interval == 'weekly' or interval == '1wk':
            # 对于周数据，每7天一个点
            while current <= end:
                if current.weekday() == 0:  # 只取周一的数据
                    date_range.append(current)
                current += timedelta(days=1)
        elif interval == 'monthly' or interval == '1mo':
            # 对于月数据，每月1号一个点
            while current <= end:
                if current.day == 1:  # 只取每月1号的数据
                    date_range.append(current)
                current += timedelta(days=1)
        else:
            # 对于日数据，跳过周末
            while current <= end:
                if current.weekday() < 5:  # 0-4是周一至周五
                    date_range.append(current)
                current += timedelta(days=1)
        
        # 使用固定的随机种子确保可重复性
        np.random.seed(self.config.get('mock_data_seed', 42) + hash(symbol) % 1000)
        
        # 获取基本股票信息，用于模拟更真实的价格
        stock_info = self._identify_stock_type(symbol)
        # 模拟一个基础价格，根据股票代码生成具有一定规律的价格
        base_price = (ord(symbol[0]) * 10 + int(symbol[-2:]) % 100) % 90 + 10  # 10-100之间的价格
        
        # 生成模拟数据
        price = base_price  # 使用计算的基础价格
        mock_data = []
        
        for date in date_range:
            # 使用正弦函数添加一些周期性变化
            cycle_factor = np.sin(len(mock_data) / 10) * 0.01  # 添加轻微的周期性
            
            price_change = (np.random.normal(0, 0.02) + cycle_factor) * price  # 每日价格变化，正态分布
            price += price_change
            open_price = price * (1 + np.random.normal(0, 0.01))
            high_price = max(price, open_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(price, open_price) * (1 - abs(np.random.normal(0, 0.01)))
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
        
        # 标准化数据
        try:
            from src.utils.column_mapping import standardize_columns
            standardized_df = standardize_columns(df, source_type='AKSHARE')
            
            # 检查是否返回了空的DataFrame
            if standardized_df is None or standardized_df.empty:
                logger.warning(f"standardize_columns返回了空数据，回退使用原始数据")
                standardized_df = df
            
            # 检查是否缺少原本应该有的关键列
            if 'date' not in standardized_df.columns and 'date' in df.columns:
                logger.warning(f"标准化后丢失了date列，恢复原始列")
                standardized_df['date'] = df['date']
            
            if 'close' not in standardized_df.columns and 'close' in df.columns:
                logger.warning(f"标准化后丢失了close列，恢复原始列")
                standardized_df['close'] = df['close']
                
            return standardized_df
        except Exception as e:
            logger.warning(f"标准化列名失败: {e}，将使用原始数据")
            return df