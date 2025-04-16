"""
数据源模块

这个模块负责从各种数据源获取股票数据，包括A股、美股等市场。
支持多种数据源，如AKShare、Tushare等。
"""

import pandas as pd
import akshare as ak
import logging
from abc import ABC, abstractmethod
from src.utils.column_mapping import standardize_columns, detect_and_log_column_issues, StandardColumns

# 创建模块级日志记录器
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """数据源基类，定义通用数据获取接口"""
    
    @abstractmethod
    def get_stock_daily(self, symbol, start_date=None, end_date=None):
        """
        获取股票日线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 股票日线数据
        """
        pass
    
    @abstractmethod
    def get_stock_list(self):
        """
        获取股票列表
        
        Returns:
            DataFrame: 股票基本信息列表
        """
        pass
    
    def validate_data(self, df, required_columns=None):
        """
        验证数据是否有效
        
        Args:
            df: 输入数据框
            required_columns: 必需的列列表
            
        Returns:
            bool: 数据是否有效
        """
        if df is None or df.empty:
            logger.warning(f"数据源返回空数据")
            return False
            
        # 默认检查日期、开盘价、收盘价和成交量
        if required_columns is None:
            required_columns = [
                StandardColumns.DATE,
                StandardColumns.OPEN,
                StandardColumns.CLOSE,
                StandardColumns.VOLUME
            ]
            
        # 检查是否有缺失列
        missing_columns = detect_and_log_column_issues(df, required_columns, logger)
        
        if missing_columns:
            logger.warning(f"数据源返回的数据缺少必需列: {missing_columns}")
            return False
            
        return True


class AKShareDataSource(DataSource):
    """基于AKShare的数据源实现"""
    
    def get_stock_daily(self, symbol, start_date=None, end_date=None):
        """
        从AKShare获取A股日线数据
        
        Args:
            symbol: 股票代码，格式为"sh600000"
            start_date: 开始日期，格式为"YYYYMMDD"
            end_date: 结束日期，格式为"YYYYMMDD"
            
        Returns:
            DataFrame: 标准化的股票日线数据
        """
        logger.info(f"从AKShare获取股票{symbol}的日线数据")
        
        try:
            # 从AKShare获取数据
            df = ak.stock_zh_a_hist(
                symbol=symbol.replace("sh", "").replace("sz", ""),
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            logger.debug(f"获取到原始列：{df.columns.tolist()}")
            
            # 标准化列名
            df = standardize_columns(df, verbose=True)
            
            # 确保包含股票代码列
            if StandardColumns.SYMBOL.value not in df.columns:
                df[StandardColumns.SYMBOL.value] = symbol
                
            # 验证数据
            if not self.validate_data(df):
                logger.warning(f"股票{symbol}的数据验证失败")
                return None
                
            logger.info(f"成功获取股票{symbol}的日线数据，共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取股票{symbol}的日线数据失败: {e}")
            return None
    
    def get_stock_list(self):
        """
        获取A股股票列表
        
        Returns:
            DataFrame: 标准化的A股股票列表
        """
        logger.info("从AKShare获取A股股票列表")
        
        try:
            # 获取沪市股票
            df_sh = ak.stock_info_sh_name_code()
            df_sh['market'] = 'sh'
            
            # 获取深市股票
            df_sz = ak.stock_info_sz_name_code()
            df_sz['market'] = 'sz'
            
            # 合并
            df = pd.concat([df_sh, df_sz])
            
            # 标准化列名
            df = standardize_columns(df, verbose=True)
            
            # 创建完整的股票代码
            if 'market' in df.columns and StandardColumns.CODE.value in df.columns:
                df[StandardColumns.SYMBOL.value] = df['market'] + df[StandardColumns.CODE.value]
            
            logger.info(f"成功获取A股股票列表，共{len(df)}只股票")
            return df
            
        except Exception as e:
            logger.error(f"获取A股股票列表失败: {e}")
            return None


class TushareDataSource(DataSource):
    """基于Tushare的数据源实现"""
    
    def __init__(self, token=None):
        """
        初始化Tushare数据源
        
        Args:
            token: Tushare的API token
        """
        self.token = token
        # 这里可以添加Tushare的初始化代码
        logger.info("初始化Tushare数据源")
    
    def get_stock_daily(self, symbol, start_date=None, end_date=None):
        """
        从Tushare获取A股日线数据
        
        Args:
            symbol: 股票代码，格式为"sh600000"
            start_date: 开始日期，格式为"YYYYMMDD"
            end_date: 结束日期，格式为"YYYYMMDD"
            
        Returns:
            DataFrame: 标准化的股票日线数据
        """
        logger.info(f"从Tushare获取股票{symbol}的日线数据")
        # 目前仅占位，实际实现需根据Tushare API进行开发
        logger.warning("Tushare数据源尚未实现")
        return None
    
    def get_stock_list(self):
        """
        获取A股股票列表
        
        Returns:
            DataFrame: 标准化的A股股票列表
        """
        logger.info("从Tushare获取A股股票列表")
        # 目前仅占位，实际实现需根据Tushare API进行开发
        logger.warning("Tushare数据源尚未实现")
        return None


def create_data_source(source_type="akshare", **kwargs):
    """
    创建数据源实例
    
    Args:
        source_type: 数据源类型，支持"akshare"和"tushare"
        **kwargs: 数据源所需的其他参数
        
    Returns:
        DataSource: 数据源实例
    """
    logger.info(f"创建{source_type}数据源")
    
    if source_type.lower() == "akshare":
        return AKShareDataSource()
    elif source_type.lower() == "tushare":
        token = kwargs.get("token")
        return TushareDataSource(token=token)
    else:
        logger.error(f"不支持的数据源类型: {source_type}")
        raise ValueError(f"不支持的数据源类型: {source_type}") 