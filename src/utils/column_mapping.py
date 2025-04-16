"""
列名标准化系统

定义股票数据的标准列名和别名映射，以及列名验证功能。
支持多种数据源的字段名映射到统一的内部标准。
"""

import enum
import logging
import pandas as pd
from typing import List, Dict, Set, Optional, Union, Any


class StandardColumns(enum.Enum):
    """标准列名枚举"""
    
    # 基础行情数据
    DATE = "date"                # 日期
    OPEN = "open"                # 开盘价
    HIGH = "high"                # 最高价
    LOW = "low"                  # 最低价
    CLOSE = "close"              # 收盘价
    VOLUME = "volume"            # 成交量
    AMOUNT = "amount"            # 成交额
    CHANGE = "change"            # 涨跌幅
    CHANGE_PCT = "change_pct"    # 涨跌幅百分比
    TURNOVER = "turnover"        # 换手率
    
    # 股票基本信息
    SYMBOL = "symbol"            # 股票代码
    NAME = "name"                # 股票名称
    MARKET = "market"            # 市场（沪市/深市/港股/美股等）
    INDUSTRY = "industry"        # 所属行业
    
    # 指标数据
    PE = "pe"                    # 市盈率
    PB = "pb"                    # 市净率
    PS = "ps"                    # 市销率
    PCF = "pcf"                  # 市现率
    
    # 财务数据
    MARKET_CAP = "market_cap"    # 市值
    CIRC_MARKET_CAP = "circ_market_cap"  # 流通市值
    TOTAL_ASSETS = "total_assets"        # 总资产
    TOTAL_LIAB = "total_liab"            # 总负债
    NET_PROFIT = "net_profit"            # 净利润
    REVENUE = "revenue"                  # 营业收入
    
    # 其他
    ADJ_FACTOR = "adj_factor"    # 复权因子
    SOURCE = "source"            # 来源


# 中文列名到英文列名的映射（兼容旧版本）
CN_TO_EN_COLUMN_MAPPING = {
    "日期": "date",
    "开盘": "open",
    "开盘价": "open",
    "收盘": "close",
    "收盘价": "close",
    "最高": "high",
    "最高价": "high",
    "最低": "low",
    "最低价": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "换手率": "turnover",
    "代码": "symbol",
    "名称": "name",
    "市场类型": "market",
    "行业": "industry",
    "市盈率": "pe",
    "市净率": "pb",
    "市值": "market_cap",
    "流通市值": "circ_market_cap"
}


# 各数据源字段到标准字段的映射
SOURCE_COLUMN_MAPPINGS = {
    # A股（通用）
    "A_SHARE": {
        "日期": StandardColumns.DATE,
        "开盘": StandardColumns.OPEN,
        "开盘价": StandardColumns.OPEN,
        "最高": StandardColumns.HIGH,
        "最高价": StandardColumns.HIGH,
        "最低": StandardColumns.LOW,
        "最低价": StandardColumns.LOW,
        "收盘": StandardColumns.CLOSE,
        "收盘价": StandardColumns.CLOSE,
        "成交量": StandardColumns.VOLUME,
        "成交额": StandardColumns.AMOUNT,
        "涨跌幅": StandardColumns.CHANGE_PCT,
        "涨跌额": StandardColumns.CHANGE,
        "换手率": StandardColumns.TURNOVER,
        "代码": StandardColumns.SYMBOL,
        "名称": StandardColumns.NAME,
        "市场": StandardColumns.MARKET,
        "行业": StandardColumns.INDUSTRY,
        "市盈率": StandardColumns.PE,
        "市净率": StandardColumns.PB,
        "市值": StandardColumns.MARKET_CAP,
        "流通市值": StandardColumns.CIRC_MARKET_CAP,
        "总资产": StandardColumns.TOTAL_ASSETS,
        "总负债": StandardColumns.TOTAL_LIAB,
        "净利润": StandardColumns.NET_PROFIT,
        "营业收入": StandardColumns.REVENUE,
        "复权因子": StandardColumns.ADJ_FACTOR,
    },
    
    # AKShare的字段映射
    "AKSHARE": {
        # 中文列名映射
        "日期": StandardColumns.DATE,
        "开盘": StandardColumns.OPEN,
        "高": StandardColumns.HIGH,
        "最高": StandardColumns.HIGH,
        "低": StandardColumns.LOW,
        "最低": StandardColumns.LOW,
        "收盘": StandardColumns.CLOSE,
        "成交量": StandardColumns.VOLUME,
        "成交额": StandardColumns.AMOUNT,
        "振幅": StandardColumns.CHANGE_PCT,
        "涨跌幅": StandardColumns.CHANGE_PCT,
        "涨跌额": StandardColumns.CHANGE,
        "换手率": StandardColumns.TURNOVER,
        "代码": StandardColumns.SYMBOL,
        "名称": StandardColumns.NAME,
        "市场类型": StandardColumns.MARKET,
        
        # 增加英文列名映射，用于支持模拟数据和其他可能的英文数据源
        "date": StandardColumns.DATE,
        "open": StandardColumns.OPEN,
        "high": StandardColumns.HIGH,
        "low": StandardColumns.LOW,
        "close": StandardColumns.CLOSE,
        "volume": StandardColumns.VOLUME,
        "amount": StandardColumns.AMOUNT,
        "change": StandardColumns.CHANGE,
        "change_pct": StandardColumns.CHANGE_PCT,
        "turnover": StandardColumns.TURNOVER,
        "symbol": StandardColumns.SYMBOL,
        "name": StandardColumns.NAME,
        "market": StandardColumns.MARKET,
        "source": StandardColumns.SOURCE,
    },
    
    # Tushare字段映射
    "TUSHARE": {
        "trade_date": StandardColumns.DATE,
        "open": StandardColumns.OPEN,
        "high": StandardColumns.HIGH,
        "low": StandardColumns.LOW,
        "close": StandardColumns.CLOSE,
        "vol": StandardColumns.VOLUME,
        "amount": StandardColumns.AMOUNT,
        "pct_chg": StandardColumns.CHANGE_PCT,
        "change": StandardColumns.CHANGE,
        "turnover_rate": StandardColumns.TURNOVER,
        "ts_code": StandardColumns.SYMBOL,
        "name": StandardColumns.NAME,
        "market": StandardColumns.MARKET,
        "industry": StandardColumns.INDUSTRY,
        "pe": StandardColumns.PE,
        "pb": StandardColumns.PB,
        "ps": StandardColumns.PS,
        "total_mv": StandardColumns.MARKET_CAP,
        "circ_mv": StandardColumns.CIRC_MARKET_CAP,
        "total_assets": StandardColumns.TOTAL_ASSETS,
        "total_liab": StandardColumns.TOTAL_LIAB,
        "net_profit": StandardColumns.NET_PROFIT,
        "revenue": StandardColumns.REVENUE,
        "adj_factor": StandardColumns.ADJ_FACTOR,
    },
    
    # 美股字段映射
    "US_STOCK": {
        "date": StandardColumns.DATE,
        "open": StandardColumns.OPEN,
        "high": StandardColumns.HIGH,
        "low": StandardColumns.LOW,
        "close": StandardColumns.CLOSE,
        "volume": StandardColumns.VOLUME,
        "amount": StandardColumns.AMOUNT,
        "change": StandardColumns.CHANGE,
        "change_pct": StandardColumns.CHANGE_PCT,
        "symbol": StandardColumns.SYMBOL,
        "name": StandardColumns.NAME,
        "market": StandardColumns.MARKET,
        "industry": StandardColumns.INDUSTRY,
        "pe_ratio": StandardColumns.PE,
        "pb_ratio": StandardColumns.PB,
        "ps_ratio": StandardColumns.PS,
        "market_cap": StandardColumns.MARKET_CAP,
    },
}


# 存储用映射字典
STORAGE_TO_STANDARD_MAPPING = {
    "date": StandardColumns.DATE.value,
    "open": StandardColumns.OPEN.value,
    "high": StandardColumns.HIGH.value,
    "low": StandardColumns.LOW.value, 
    "close": StandardColumns.CLOSE.value,
    "volume": StandardColumns.VOLUME.value,
    "amount": StandardColumns.AMOUNT.value,
    "change": StandardColumns.CHANGE.value,
    "change_pct": StandardColumns.CHANGE_PCT.value,
    "turnover": StandardColumns.TURNOVER.value,
    "symbol": StandardColumns.SYMBOL.value,
    "name": StandardColumns.NAME.value,
    "market": StandardColumns.MARKET.value,
    "industry": StandardColumns.INDUSTRY.value
}

# 标准到存储的映射
STANDARD_TO_STORAGE_MAPPING = {v: k for k, v in STORAGE_TO_STANDARD_MAPPING.items()}


def standardize_columns(df: pd.DataFrame, source_type: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    将数据源的列名标准化为内部标准列名
    
    Args:
        df: 输入数据框
        source_type: 数据源类型，如'AKSHARE', 'TUSHARE', 'A_SHARE', 'US_STOCK'等
        logger: 可选的日志记录器
        
    Returns:
        DataFrame: 列名标准化后的数据框
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if df is None or df.empty:
        logger.warning("无法标准化空数据框")
        return df
        
    if source_type not in SOURCE_COLUMN_MAPPINGS:
        logger.warning(f"未知的数据源类型: {source_type}，无法进行列名标准化")
        return df
        
    mapping = SOURCE_COLUMN_MAPPINGS[source_type]
    renamed_columns = {}
    
    for col in df.columns:
        if col in mapping:
            std_col = mapping[col]
            # 如果是StandardColumns枚举类型，获取它的值
            if isinstance(std_col, StandardColumns):
                std_col = std_col.value
            renamed_columns[col] = std_col
            logger.debug(f"列名映射: {col} -> {std_col}")
            
    # 如果没有列被映射，发出警告
    if not renamed_columns:
        logger.warning(f"数据源 {source_type} 没有任何列可以映射到标准列名")
        return df  # 返回原始DataFrame，而不是空DataFrame
        
    # 记录映射信息
    logger.info(f"数据源 {source_type} 映射了 {len(renamed_columns)}/{len(df.columns)} 列到标准列名")
    
    # 重命名列
    result = df.rename(columns=renamed_columns)
    
    # 确保没有丢失原始DataFrame中的数据
    if result.empty and not df.empty:
        logger.warning("列名标准化导致数据丢失，返回原始数据")
        return df
        
    return result


def detect_and_log_column_issues(
    df: pd.DataFrame, 
    required_columns: List[Union[str, StandardColumns]], 
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    检测数据框中缺失的必需列，并记录日志
    
    Args:
        df: 输入数据框
        required_columns: 必需列的列表，可以是字符串或StandardColumns枚举
        logger: 可选的日志记录器
        
    Returns:
        List[str]: 缺失列的列表
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # 获取数据框的列集合
    actual_columns = set(df.columns)
    
    # 将所有列名转换为字符串形式
    required_cols = []
    for col in required_columns:
        if isinstance(col, StandardColumns):
            required_cols.append(col.value)
        else:
            required_cols.append(col)
    
    # 检查缺失列
    missing_columns = [col for col in required_cols if col not in actual_columns]
    
    if missing_columns:
        logger.warning(f"数据缺少必需列: {missing_columns}")
    else:
        logger.debug("数据包含所有必需列")
        
    return missing_columns


def suggest_column_mappings(
    df: pd.DataFrame, 
    target_columns: List[Union[str, StandardColumns]], 
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    为不匹配的列名建议可能的映射
    
    Args:
        df: 输入数据框
        target_columns: 目标列列表
        logger: 可选的日志记录器
        
    Returns:
        Dict[str, str]: 建议的列名映射
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # 转换StandardColumns枚举为字符串值
    target_cols = []
    for col in target_columns:
        if isinstance(col, StandardColumns):
            target_cols.append(col.value)
        else:
            target_cols.append(col)
            
    # 获取数据框的列集合
    actual_columns = set(df.columns)
    
    # 获取所有标准列名的可能别名
    all_mappings = {}
    for source_type, mappings in SOURCE_COLUMN_MAPPINGS.items():
        for source_col, std_col in mappings.items():
            if isinstance(std_col, StandardColumns):
                std_col_str = std_col.value
                if std_col_str not in all_mappings:
                    all_mappings[std_col_str] = set()
                all_mappings[std_col_str].add(source_col)
    
    # 为缺失的列建议可能的映射
    suggestions = {}
    for col in target_cols:
        if col not in actual_columns:
            # 检查数据框中是否有任何该标准列的已知别名
            if col in all_mappings:
                possible_matches = all_mappings[col].intersection(actual_columns)
                if possible_matches:
                    suggestions[col] = list(possible_matches)[0]  # 取第一个匹配项
                    logger.info(f"为缺失列 {col} 建议映射: {suggestions[col]}")
    
    return suggestions


def get_standard_column_list(category: Optional[str] = None) -> List[str]:
    """
    获取标准列名列表
    
    Args:
        category: 可选的类别过滤器，如'基础行情'，'股票信息'等
        
    Returns:
        List[str]: 标准列名列表
    """
    all_columns = [col.value for col in StandardColumns]
    
    if category is None:
        return all_columns
        
    # 基于类别过滤列
    category_filters = {
        "基础行情": [
            StandardColumns.DATE.value,
            StandardColumns.OPEN.value,
            StandardColumns.HIGH.value,
            StandardColumns.LOW.value,
            StandardColumns.CLOSE.value,
            StandardColumns.VOLUME.value,
            StandardColumns.AMOUNT.value,
            StandardColumns.CHANGE.value,
            StandardColumns.CHANGE_PCT.value,
            StandardColumns.TURNOVER.value,
        ],
        "股票信息": [
            StandardColumns.SYMBOL.value,
            StandardColumns.NAME.value,
            StandardColumns.MARKET.value,
            StandardColumns.INDUSTRY.value,
        ],
        "估值指标": [
            StandardColumns.PE.value,
            StandardColumns.PB.value,
            StandardColumns.PS.value,
            StandardColumns.PCF.value,
        ],
        "财务数据": [
            StandardColumns.MARKET_CAP.value,
            StandardColumns.CIRC_MARKET_CAP.value,
            StandardColumns.TOTAL_ASSETS.value,
            StandardColumns.TOTAL_LIAB.value,
            StandardColumns.NET_PROFIT.value,
            StandardColumns.REVENUE.value,
        ],
    }
    
    if category in category_filters:
        return category_filters[category]
    else:
        return all_columns 


def apply_reverse_mapping_for_analysis(df, source_type="STORAGE"):
    """
    应用反向映射，从标准列名转换为特定分析工具需要的列名
    
    Args:
        df: 输入数据框
        source_type: 目标映射类型，默认为"STORAGE"
    
    Returns:
        DataFrame: 列名转换后的数据框
    """
    if source_type == "STORAGE":
        # 使用 STANDARD_TO_STORAGE_MAPPING
        rename_map = {k: v for k, v in STANDARD_TO_STORAGE_MAPPING.items() if k in df.columns}
    else:
        # 构建从标准列名到源列名的映射
        reverse_map = {}
        if source_type in SOURCE_COLUMN_MAPPINGS:
            for src_col, std_col in SOURCE_COLUMN_MAPPINGS[source_type].items():
                if isinstance(std_col, StandardColumns):
                    reverse_map[std_col.value] = src_col
                else:
                    reverse_map[std_col] = src_col
                    
        rename_map = {k: v for k, v in reverse_map.items() if k in df.columns}
    
    if rename_map:
        return df.rename(columns=rename_map)
    return df.copy() 