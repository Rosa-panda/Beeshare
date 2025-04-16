"""
AKShare数据源实现模块

基于AKShare库实现DataSourceInterface接口，提供A股数据获取功能。
"""

import logging
import time
from typing import Union, List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from src.data.data_source.data_source_interface import DataSourceInterface, DataSourceError


class AKShareDataSource(DataSourceInterface):
    """AKShare数据源实现类
    
    使用AKShare库实现数据源接口，提供A股数据获取功能。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化AKShare数据源
        
        Args:
            config: 配置参数，可选
        """
        self.logger = logging.getLogger('BeeShare.AKShareDataSource')
        self.config = config or {}
        self.stock_list = None
        self.stock_list_last_update = None
        self.initialized = False
        self.stock_list_cache_time = self.config.get('stock_list_cache_time', 86400)  # 默认缓存1天
        self.retry_count = self.config.get('retry_count', 3)  # 默认重试3次
        self.retry_delay = self.config.get('retry_delay', 1)  # 默认延迟1秒
        
        # 初始化数据源
        if self.config.get('auto_initialize', True):
            self.initialize(self.config)
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化数据源
        
        Args:
            config: 配置参数，可选
            
        Returns:
            bool: 初始化是否成功
            
        Raises:
            DataSourceError: 初始化失败时抛出
        """
        if config:
            self.config.update(config)
            
        if not AKSHARE_AVAILABLE:
            error_msg = "AKShare库未安装，请执行 'pip install akshare' 安装"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", error_code=1001)
        
        try:
            # 检查AKShare版本
            version = ak.__version__
            self.logger.info(f"AKShare版本: {version}")
            
            # 尝试获取股票列表，验证AKShare可用性
            self._update_stock_list()
            self.initialized = True
            self.logger.info("AKShare数据源初始化成功")
            return True
        except Exception as e:
            error_msg = f"AKShare数据源初始化失败: {str(e)}"
            self.logger.error(error_msg)
            self.initialized = False
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=1002, original_error=e)
    
    def is_available(self) -> bool:
        """检查数据源是否可用
        
        Returns:
            bool: 数据源是否可用
        """
        if not AKSHARE_AVAILABLE:
            return False
        
        if not self.initialized:
            try:
                return self.initialize()
            except:
                return False
        
        return True
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                           interval: str = '1d') -> pd.DataFrame:
        """获取历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            interval: 数据间隔，默认日频'1d'
            
        Returns:
            DataFrame: 包含历史数据的DataFrame对象
            
        Raises:
            DataSourceError: 获取数据失败时抛出
        """
        self._ensure_initialized()
        
        # 处理股票代码格式
        processed_symbol = self._process_stock_symbol(symbol)
        
        # 根据interval转换为AKShare的参数
        period_map = {
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly'
        }
        period = period_map.get(interval, 'daily')
        
        try:
            for attempt in range(self.retry_count):
                try:
                    # 使用AKShare获取历史数据
                    if period == 'daily':
                        df = ak.stock_zh_a_hist(symbol=processed_symbol, 
                                              start_date=start_date.replace('-', ''), 
                                              end_date=end_date.replace('-', ''),
                                              adjust="qfq")
                    elif period == 'weekly':
                        df = ak.stock_zh_a_hist_weekly(symbol=processed_symbol, 
                                                     start_date=start_date.replace('-', ''), 
                                                     end_date=end_date.replace('-', ''),
                                                     adjust="qfq")
                    elif period == 'monthly':
                        df = ak.stock_zh_a_hist_monthly(symbol=processed_symbol, 
                                                      start_date=start_date.replace('-', ''), 
                                                      end_date=end_date.replace('-', ''),
                                                      adjust="qfq")
                    else:
                        raise ValueError(f"不支持的数据间隔: {interval}")
                    
                    # 标准化列名
                    column_map = {
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'amount',
                        '振幅': 'amplitude',
                        '涨跌幅': 'change_pct',
                        '涨跌额': 'change',
                        '换手率': 'turnover_rate'
                    }
                    
                    # 重命名列
                    df = df.rename(columns=column_map)
                    
                    # 确保日期格式正确
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # 添加股票代码列
                    df['symbol'] = symbol
                    
                    # 添加数据源标识
                    df['source'] = 'akshare'
                    
                    return df
                
                except Exception as e:
                    self.logger.warning(f"获取历史数据尝试 {attempt+1}/{self.retry_count} 失败: {str(e)}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
        except Exception as e:
            error_msg = f"获取股票 {symbol} 的历史数据失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2001, original_error=e)
    
    def get_realtime_data(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """获取实时数据
        
        Args:
            symbols: 股票代码或股票代码列表
            
        Returns:
            DataFrame: 包含实时数据的DataFrame对象
            
        Raises:
            DataSourceError: 获取数据失败时抛出
        """
        self._ensure_initialized()
        
        # 将单个股票代码转换为列表
        if isinstance(symbols, str):
            symbols = [symbols]
        
        try:
            # 使用AKShare获取A股实时行情
            df = ak.stock_zh_a_spot()
            
            # 过滤需要的股票
            processed_symbols = [self._process_stock_symbol(symbol) for symbol in symbols]
            result_df = df[df['代码'].isin(processed_symbols)].copy()
            
            if result_df.empty and symbols:
                self.logger.warning(f"未找到股票代码 {symbols} 的实时数据")
            
            # 标准化列名
            column_map = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'previous_close',
                '换手率': 'turnover_rate',
                '市盈率-动态': 'pe_ttm',
                '市净率': 'pb'
            }
            
            # 重命名列
            result_df = result_df.rename(columns=column_map)
            
            # 添加时间戳
            result_df['timestamp'] = datetime.now()
            
            # 添加数据源标识
            result_df['source'] = 'akshare'
            
            return result_df
            
        except Exception as e:
            error_msg = f"获取实时数据失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2002, original_error=e)
    
    def search_symbols(self, keyword: str) -> pd.DataFrame:
        """搜索股票代码
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            DataFrame: 包含搜索结果的DataFrame对象
            
        Raises:
            DataSourceError: 搜索失败时抛出
        """
        self._ensure_initialized()
        
        try:
            # 获取股票列表
            stock_list = self._get_stock_list()
            
            # 在代码和名称中搜索关键词
            result = stock_list[(stock_list['代码'].str.contains(keyword) | 
                               stock_list['名称'].str.contains(keyword))]
            
            # 标准化列名
            column_map = {
                '代码': 'symbol',
                '名称': 'name',
                '所属行业': 'industry',
                '上市时间': 'list_date'
            }
            
            # 重命名列
            result = result.rename(columns=column_map)
            
            # 添加数据源标识
            result['source'] = 'akshare'
            
            return result
            
        except Exception as e:
            error_msg = f"搜索股票失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2003, original_error=e)
    
    def get_index_data(self, symbol: str, start_date: str, end_date: str,
                     interval: str = '1d') -> pd.DataFrame:
        """获取指数数据
        
        Args:
            symbol: 指数代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            interval: 数据间隔，默认日频'1d'
            
        Returns:
            DataFrame: 包含指数数据的DataFrame对象
            
        Raises:
            DataSourceError: 获取数据失败时抛出
        """
        self._ensure_initialized()
        
        # 根据interval转换为AKShare的参数
        period_map = {
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly'
        }
        period = period_map.get(interval, 'daily')
        
        try:
            for attempt in range(self.retry_count):
                try:
                    # 根据不同周期获取指数数据
                    if period == 'daily':
                        df = ak.stock_zh_index_daily(symbol=symbol)
                    elif period == 'weekly':
                        df = ak.stock_zh_index_weekly(symbol=symbol)
                    elif period == 'monthly':
                        df = ak.stock_zh_index_monthly(symbol=symbol)
                    else:
                        raise ValueError(f"不支持的数据间隔: {interval}")
                    
                    # 过滤日期范围
                    df['date'] = pd.to_datetime(df['date'])
                    start_date_dt = pd.to_datetime(start_date)
                    end_date_dt = pd.to_datetime(end_date)
                    df = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
                    
                    # 添加指数代码列
                    df['symbol'] = symbol
                    
                    # 添加数据源标识
                    df['source'] = 'akshare'
                    
                    return df
                
                except Exception as e:
                    self.logger.warning(f"获取指数数据尝试 {attempt+1}/{self.retry_count} 失败: {str(e)}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
        except Exception as e:
            error_msg = f"获取指数 {symbol} 的数据失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2004, original_error=e)
    
    def identify_stock_type(self, symbol: str) -> Dict[str, Any]:
        """识别股票类型
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 包含股票类型信息的字典
            
        Raises:
            DataSourceError: 识别失败时抛出
        """
        self._ensure_initialized()
        
        try:
            # 获取股票市场信息
            result = {}
            
            # 处理股票代码
            symbol = symbol.strip().upper()
            if symbol.startswith(('SH', 'SZ', 'BJ')):
                symbol = symbol[2:]
            
            # 股票市场和板块识别规则
            if symbol.startswith('60'):
                result['market'] = 'SHSE'
                result['exchange'] = '上交所'
                result['board'] = '主板'
                result['prefix'] = 'sh'
            elif symbol.startswith('688'):
                result['market'] = 'SHSE'
                result['exchange'] = '上交所'
                result['board'] = '科创板'
                result['prefix'] = 'sh'
            elif symbol.startswith('000') or symbol.startswith('001'):
                result['market'] = 'SZSE'
                result['exchange'] = '深交所'
                result['board'] = '主板'
                result['prefix'] = 'sz'
            elif symbol.startswith('002'):
                result['market'] = 'SZSE'
                result['exchange'] = '深交所'
                result['board'] = '中小板'
                result['prefix'] = 'sz'
            elif symbol.startswith('003'):
                result['market'] = 'SZSE'
                result['exchange'] = '深交所'
                result['board'] = '主板'
                result['prefix'] = 'sz'
            elif symbol.startswith('300'):
                result['market'] = 'SZSE'
                result['exchange'] = '深交所'
                result['board'] = '创业板'
                result['prefix'] = 'sz'
            elif symbol.startswith(('4', '8')) and not symbol.startswith('688'):
                result['market'] = 'BJSE'
                result['exchange'] = '北交所'
                result['board'] = '主板'
                result['prefix'] = 'bj'
            else:
                result['market'] = 'UNKNOWN'
                result['exchange'] = '未知'
                result['board'] = '未知'
                result['prefix'] = 'unknown'
            
            # 完整代码
            result['symbol'] = symbol
            result['full_symbol'] = f"{result['prefix']}{symbol}"
            
            return result
            
        except Exception as e:
            error_msg = f"识别股票 {symbol} 类型失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2005, original_error=e)
    
    def get_all_stock_list(self) -> pd.DataFrame:
        """获取所有股票列表
        
        Returns:
            DataFrame: 包含所有股票信息的DataFrame对象
            
        Raises:
            DataSourceError: 获取失败时抛出
        """
        self._ensure_initialized()
        
        try:
            # 获取股票列表
            stock_list = self._get_stock_list()
            
            # 标准化列名
            column_map = {
                '代码': 'symbol',
                '名称': 'name',
                '所属行业': 'industry',
                '上市时间': 'list_date'
            }
            
            # 重命名列
            result = stock_list.rename(columns=column_map)
            
            # 添加数据源标识
            result['source'] = 'akshare'
            
            return result
            
        except Exception as e:
            error_msg = "获取股票列表失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2006, original_error=e)
    
    def _ensure_initialized(self):
        """确保数据源已初始化
        
        Raises:
            DataSourceError: 数据源未初始化时抛出
        """
        if not self.initialized:
            error_msg = "AKShare数据源未初始化"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", error_code=1003)
    
    def _get_stock_list(self) -> pd.DataFrame:
        """获取股票列表，使用缓存优化性能
        
        Returns:
            DataFrame: 股票列表
            
        Raises:
            DataSourceError: 获取失败时抛出
        """
        current_time = time.time()
        
        # 检查缓存是否有效
        if (self.stock_list is None or 
            self.stock_list_last_update is None or 
            (current_time - self.stock_list_last_update) > self.stock_list_cache_time):
            self._update_stock_list()
        
        return self.stock_list
    
    def _update_stock_list(self):
        """更新股票列表缓存
        
        Raises:
            DataSourceError: 更新失败时抛出
        """
        try:
            # 获取A股股票列表
            self.stock_list = ak.stock_info_a_code_name()
            self.stock_list_last_update = time.time()
            self.logger.info(f"更新股票列表成功，共 {len(self.stock_list)} 条记录")
        except Exception as e:
            error_msg = f"更新股票列表失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceError(error_msg, source="AKShareDataSource", 
                                 error_code=2007, original_error=e)
    
    def _process_stock_symbol(self, symbol: str) -> str:
        """处理股票代码，确保格式正确
        
        Args:
            symbol: 原始股票代码
            
        Returns:
            str: 处理后的股票代码
        """
        symbol = symbol.strip().upper()
        
        # 移除可能的前缀
        if symbol.startswith(('SH', 'SZ', 'BJ')):
            symbol = symbol[2:]
        
        return symbol 