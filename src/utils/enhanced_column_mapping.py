"""
增强版列标准化系统

在原有column_mapping.py基础上增强的功能，提供更智能的列名标准化能力，
包括模糊匹配、数据内容推断和用户自定义映射支持。

创建日期: 2024-04-13
作者: BeeShare开发团队
"""

import os
import json
import logging
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Union, Any, Set

# 从原始模块导入所需组件
from .column_mapping import (
    StandardColumns, 
    SOURCE_COLUMN_MAPPINGS, 
    STORAGE_TO_STANDARD_MAPPING,
    STANDARD_TO_STORAGE_MAPPING
)

logger = logging.getLogger(__name__)


def find_closest_column_match(column_name: str, target_columns: List[str], threshold: float = 0.8) -> Optional[str]:
    """
    查找最相似的列名匹配
    
    Args:
        column_name: 待匹配的列名
        target_columns: 目标列名列表
        threshold: 相似度阈值(0-1之间)
        
    Returns:
        最匹配的列名或None
    """
    best_match = None
    best_ratio = 0
    
    # 将输入列名转为小写便于比较
    cn_lower = column_name.lower()
    
    # 首先尝试精确匹配
    for target in target_columns:
        if target.lower() == cn_lower:
            return target
    
    # 然后尝试部分匹配和相似度匹配
    for target in target_columns:
        # 部分包含关系
        target_lower = target.lower()
        if target_lower in cn_lower or cn_lower in target_lower:
            ratio = 0.85  # 给予部分包含关系较高的初始相似度
        else:
            # 使用SequenceMatcher计算字符串相似度
            ratio = SequenceMatcher(None, cn_lower, target_lower).ratio()
        
        # 提高某些关键词的匹配权重
        keywords = {
            'open': ['开盘', '开盘价', 'start', 'begin'],
            'high': ['最高', '最高价', 'max', 'maximum', 'highest'],
            'low': ['最低', '最低价', 'min', 'minimum', 'lowest'],
            'close': ['收盘', '收盘价', 'end', 'finish'],
            'volume': ['成交量', '交易量', 'vol', 'amount'],
            'change': ['涨跌', '变化', 'chg'],
            'date': ['日期', '时间', 'time', 'day'],
            'symbol': ['代码', '股票代码', 'code', 'ticker']
        }
        
        # 检查关键词匹配
        for key, words in keywords.items():
            if key.lower() == target_lower:
                for word in words:
                    if word.lower() in cn_lower:
                        ratio += 0.1  # 包含关键词，提高相似度
                        break
        
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = target
    
    return best_match


def infer_column_type(df: pd.DataFrame, column_name: str) -> Optional[StandardColumns]:
    """
    根据列的数据特征推断可能的列类型
    
    Args:
        df: 数据框
        column_name: 列名
        
    Returns:
        推断的列类型
    """
    if column_name not in df.columns:
        logger.warning(f"列 '{column_name}' 不存在于数据框中")
        return None
    
    col_data = df[column_name]
    col_name_lower = column_name.lower()
    
    # 根据列名关键词进行初步判断
    name_based_inferences = {
        'date': ['date', 'time', 'day', '日期', '时间'],
        'open': ['open', 'start', 'begin', '开盘', '开盘价'],
        'high': ['high', 'max', 'maximum', 'highest', '最高', '最高价'],
        'low': ['low', 'min', 'minimum', 'lowest', '最低', '最低价'],
        'close': ['close', 'end', 'finish', '收盘', '收盘价'],
        'volume': ['volume', 'vol', '成交量', '交易量'],
        'amount': ['amount', 'value', 'turnover', '成交额', '金额'],
        'change': ['change', 'chg', 'diff', '涨跌额', '变动'],
        'change_pct': ['change_pct', 'pct', 'percent', '涨跌幅', '百分比', '%'],
        'symbol': ['symbol', 'code', 'ticker', '代码', '股票代码'],
        'name': ['name', 'stock_name', '名称', '股票名称', '简称']
    }
    
    for std_col, keywords in name_based_inferences.items():
        for keyword in keywords:
            if keyword in col_name_lower:
                # 匹配到关键词，返回相应枚举
                logger.debug(f"列 '{column_name}' 通过名称关键词 '{keyword}' 推断为 {std_col}")
                return getattr(StandardColumns, std_col.upper())
    
    # 日期类型判断
    if pd.api.types.is_datetime64_any_dtype(col_data):
        logger.debug(f"列 '{column_name}' 通过数据类型推断为日期")
        return StandardColumns.DATE
    
    # 数值类型判断
    if pd.api.types.is_numeric_dtype(col_data):
        # 检查是否为百分比值
        if 0 <= col_data.mean() <= 100:
            # 可能是百分比列
            if 'rate' in col_name_lower or 'ratio' in col_name_lower:
                logger.debug(f"列 '{column_name}' 通过数值特征推断为换手率")
                return StandardColumns.TURNOVER
            elif 'change' in col_name_lower or 'pct' in col_name_lower or '%' in col_name_lower:
                logger.debug(f"列 '{column_name}' 通过数值特征推断为涨跌幅")
                return StandardColumns.CHANGE_PCT
        
        # 根据数值范围特征推断
        mean_value = col_data.mean()
        if mean_value > 1_000_000:
            logger.debug(f"列 '{column_name}' 通过数值范围推断为成交额")
            return StandardColumns.AMOUNT
        elif mean_value > 10_000:
            logger.debug(f"列 '{column_name}' 通过数值范围推断为成交量")
            return StandardColumns.VOLUME
        elif 100 <= mean_value <= 10000:
            logger.debug(f"列 '{column_name}' 通过数值范围推断为价格")
            # 进一步区分不同价格列
            if '开' in col_name_lower or 'start' in col_name_lower:
                return StandardColumns.OPEN
            elif '收' in col_name_lower or 'end' in col_name_lower:
                return StandardColumns.CLOSE
            elif '高' in col_name_lower or 'max' in col_name_lower:
                return StandardColumns.HIGH
            elif '低' in col_name_lower or 'min' in col_name_lower:
                return StandardColumns.LOW
            else:
                return StandardColumns.CLOSE  # 默认为收盘价
    
    # 字符串类型判断
    if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
        # 尝试提取样本进行判断
        sample = col_data.dropna().head(10)
        
        # 股票代码判断 - 通常是6位数字或字母数字组合
        if any(len(str(x)) == 6 and str(x).isdigit() for x in sample):
            logger.debug(f"列 '{column_name}' 通过内容模式推断为股票代码")
            return StandardColumns.SYMBOL
            
        # 股票名称判断 - 通常是短字符串且唯一值较少
        if len(col_data.unique()) < 100 and sample.str.len().mean() < 10:
            logger.debug(f"列 '{column_name}' 通过内容特征推断为股票名称")
            return StandardColumns.NAME
    
    # 时间戳判断(整数类型)
    if pd.api.types.is_integer_dtype(col_data):
        # Unix时间戳通常是10位(秒)或13位(毫秒)数字
        if (col_data > 1000000000).all() and (col_data < 10000000000000).all():
            logger.debug(f"列 '{column_name}' 通过数值特征推断为时间戳(日期)")
            return StandardColumns.DATE
    
    logger.debug(f"无法推断列 '{column_name}' 的类型")
    return None


def load_user_mappings(mapping_file: str = "config/column_mappings.json") -> Dict[str, Dict[str, str]]:
    """
    从配置文件加载用户自定义的列映射
    
    Args:
        mapping_file: 映射配置文件路径
        
    Returns:
        加载的映射字典
    """
    if not os.path.exists(mapping_file):
        logger.warning(f"用户映射文件 {mapping_file} 不存在")
        return {}
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            raw_mappings = json.load(f)
        
        # 过滤掉非映射键(如"说明"等)
        user_mappings = {}
        standard_values = set(col.value for col in StandardColumns)
        
        for source, mappings in raw_mappings.items():
            # 跳过非数据源键
            if source.startswith("_") or source == "说明":
                continue
                
            user_mappings[source] = {}
            for col, std_col in mappings.items():
                # 验证标准列名是否有效
                if std_col in standard_values:
                    user_mappings[source][col] = std_col
                else:
                    # 尝试将字符串转换为枚举
                    try:
                        enum_value = getattr(StandardColumns, std_col.upper())
                        user_mappings[source][col] = enum_value
                    except (AttributeError, KeyError):
                        logger.warning(f"用户配置中的标准列名 '{std_col}' 不是有效的StandardColumns值")
        
        if user_mappings:
            logger.info(f"从 {mapping_file} 成功加载了 {len(user_mappings)} 个数据源的自定义映射")
        else:
            logger.warning(f"从 {mapping_file} 加载的用户映射为空")
            
        return user_mappings
    except Exception as e:
        logger.error(f"加载用户映射失败: {e}")
        return {}


def standardize_columns(
    df: pd.DataFrame, 
    source_type: str, 
    use_fuzzy_match: bool = False,
    infer_column_types: bool = False,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    增强版列名标准化函数
    
    Args:
        df: 输入数据框
        source_type: 数据源类型
        use_fuzzy_match: 是否使用模糊匹配
        infer_column_types: 是否推断列类型
        logger: 可选的日志记录器
        
    Returns:
        标准化后的数据框
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if df is None or df.empty:
        logger.warning("无法标准化空数据框")
        return df
    
    # 步骤0: 准备标准列名集合(用于后续查询)
    standard_column_values = [c.value for c in StandardColumns]
    
    # 步骤1: 获取映射字典
    mapping = {}
    
    # 首先尝试通过内置映射获取
    if source_type in SOURCE_COLUMN_MAPPINGS:
        mapping = SOURCE_COLUMN_MAPPINGS[source_type]
        logger.debug(f"使用内置映射 '{source_type}'")
    else:
        # 尝试加载用户自定义映射
        user_mappings = load_user_mappings()
        if source_type in user_mappings:
            mapping = user_mappings[source_type]
            logger.debug(f"使用用户自定义映射 '{source_type}'")
        else:
            logger.warning(f"未知的数据源类型: {source_type}，将尝试智能匹配")
    
    renamed_columns = {}
    
    # 步骤2: 应用精确映射
    for col in df.columns:
        if col in mapping:
            std_col = mapping[col]
            # 如果是StandardColumns枚举类型，获取它的值
            if isinstance(std_col, StandardColumns):
                std_col = std_col.value
            renamed_columns[col] = std_col
            logger.debug(f"精确列名映射: {col} -> {std_col}")
    
    # 步骤3: 应用模糊匹配（如果启用）
    if use_fuzzy_match:
        unmapped_columns = [col for col in df.columns if col not in renamed_columns]
        
        for col in unmapped_columns:
            # 首先尝试匹配映射键
            if mapping:
                key_match = find_closest_column_match(col, list(mapping.keys()))
                if key_match and key_match != col:  # 避免自匹配
                    std_col = mapping[key_match]
                    if isinstance(std_col, StandardColumns):
                        std_col = std_col.value
                    renamed_columns[col] = std_col
                    logger.debug(f"模糊键匹配: {col} -> {std_col} (通过 {key_match})")
                    continue
            
            # 然后尝试直接匹配标准列名
            std_match = find_closest_column_match(col, standard_column_values)
            if std_match:
                renamed_columns[col] = std_match
                logger.debug(f"模糊值匹配: {col} -> {std_match}")
    
    # 步骤4: 根据数据内容推断（如果启用）
    if infer_column_types:
        still_unmapped = [col for col in df.columns if col not in renamed_columns]
        for col in still_unmapped:
            inferred_type = infer_column_type(df, col)
            if inferred_type:
                std_col = inferred_type.value if isinstance(inferred_type, StandardColumns) else inferred_type
                renamed_columns[col] = std_col
                logger.debug(f"内容推断映射: {col} -> {std_col}")
    
    # 步骤5: 检查和记录映射结果
    if not renamed_columns:
        logger.warning(f"数据源 {source_type} 没有任何列可以映射到标准列名")
        return df  # 返回原始DataFrame，而不是空DataFrame
        
    # 记录映射信息
    logger.info(f"数据源 {source_type} 映射了 {len(renamed_columns)}/{len(df.columns)} 列到标准列名")
    
    # 重命名列
    result = df.rename(columns=renamed_columns)
    
    # 验证结果
    if result.empty and not df.empty:
        logger.warning("列名标准化导致数据丢失，返回原始数据")
        return df
        
    # 记录未映射的列名
    unmapped = [col for col in df.columns if col not in renamed_columns]
    if unmapped:
        logger.debug(f"以下列未映射: {unmapped}")
    
    # 返回标准化后的DataFrame
    return result


def standardize_date_format(df: pd.DataFrame, date_column: str = 'date', inplace: bool = False) -> pd.DataFrame:
    """
    标准化日期列的格式
    
    Args:
        df: 数据框
        date_column: 日期列名（应已标准化为'date'）
        inplace: 是否原位修改
        
    Returns:
        标准化日期格式后的数据框
    """
    result = df if inplace else df.copy()
    
    if date_column not in result.columns:
        logger.warning(f"列 '{date_column}' 不存在于数据框中")
        return result
    
    # 尝试将日期列转换为datetime格式
    try:
        # 先检测原始格式
        if pd.api.types.is_datetime64_any_dtype(result[date_column]):
            # 已经是日期类型，确保格式一致
            result[date_column] = pd.to_datetime(result[date_column])
            logger.debug(f"列 '{date_column}' 已经是日期类型，保持不变")
        elif pd.api.types.is_string_dtype(result[date_column]) or pd.api.types.is_object_dtype(result[date_column]):
            # 字符串格式，尝试自动解析
            original_values = result[date_column].copy()
            result[date_column] = pd.to_datetime(result[date_column], errors='coerce')
            
            # 检查是否有无法解析的值
            if result[date_column].isna().any() and not original_values.isna().any():
                # 尝试不同的日期格式
                date_formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m/%d/%Y', '%Y年%m月%d日']
                for fmt in date_formats:
                    try:
                        result[date_column] = pd.to_datetime(original_values, format=fmt, errors='coerce')
                        if not result[date_column].isna().any():
                            logger.debug(f"使用格式 '{fmt}' 成功解析日期")
                            break
                    except:
                        continue
            
            logger.debug(f"将字符串列 '{date_column}' 转换为日期类型")
        elif pd.api.types.is_integer_dtype(result[date_column]):
            # 可能是整数时间戳
            max_val = result[date_column].max()
            if max_val > 10000000000:  # 毫秒级时间戳(13位)
                result[date_column] = pd.to_datetime(result[date_column], unit='ms')
                logger.debug(f"将毫秒时间戳列 '{date_column}' 转换为日期类型")
            else:  # 秒级时间戳(10位)
                result[date_column] = pd.to_datetime(result[date_column], unit='s')
                logger.debug(f"将秒级时间戳列 '{date_column}' 转换为日期类型")
        else:
            logger.warning(f"无法识别列 '{date_column}' 的日期格式")
    except Exception as e:
        logger.warning(f"转换日期格式失败: {e}")
    
    return result


def apply_reverse_mapping_for_analysis(df: pd.DataFrame, source_type: str = "STORAGE") -> pd.DataFrame:
    """
    应用反向映射，从标准列名转换为特定分析工具需要的列名
    
    Args:
        df: 输入数据框
        source_type: 目标映射类型，默认为"STORAGE"
    
    Returns:
        DataFrame: 列名转换后的数据框
    """
    if df is None or df.empty:
        return df
        
    if source_type == "STORAGE":
        # 使用 STANDARD_TO_STORAGE_MAPPING
        rename_map = {k: v for k, v in STANDARD_TO_STORAGE_MAPPING.items() if k in df.columns}
    else:
        # 尝试从用户自定义映射中查找
        user_mappings = load_user_mappings()
        if source_type in user_mappings:
            # 构建反向映射
            source_map = user_mappings[source_type]
            reverse_map = {}
            for src_col, std_col in source_map.items():
                if isinstance(std_col, StandardColumns):
                    reverse_map[std_col.value] = src_col
                else:
                    reverse_map[std_col] = src_col
                    
            rename_map = {k: v for k, v in reverse_map.items() if k in df.columns}
        else:
            # 从内置映射中构建反向映射
            try:
                # 构建从标准列名到源列名的映射
                reverse_map = {}
                if source_type in SOURCE_COLUMN_MAPPINGS:
                    for src_col, std_col in SOURCE_COLUMN_MAPPINGS[source_type].items():
                        if isinstance(std_col, StandardColumns):
                            reverse_map[std_col.value] = src_col
                        else:
                            reverse_map[std_col] = src_col
                            
                rename_map = {k: v for k, v in reverse_map.items() if k in df.columns}
            except Exception as e:
                logger.warning(f"构建反向映射失败: {e}")
                rename_map = {}
    
    if rename_map:
        logger.debug(f"应用反向映射: {rename_map}")
        return df.rename(columns=rename_map)
    
    logger.debug("未找到适用的反向映射，返回原始数据框")
    return df.copy()


def get_all_supported_sources() -> List[str]:
    """
    获取所有支持的数据源类型
    
    Returns:
        List[str]: 所有支持的数据源类型列表
    """
    # 内置数据源
    built_in_sources = list(SOURCE_COLUMN_MAPPINGS.keys())
    
    # 用户自定义数据源
    try:
        user_mappings = load_user_mappings()
        user_sources = list(user_mappings.keys())
    except:
        user_sources = []
    
    # 合并并去重
    all_sources = list(set(built_in_sources + user_sources))
    return sorted(all_sources) 