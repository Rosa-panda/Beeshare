#!/usr/bin/env python
"""
增强版列标准化功能演示脚本

该脚本展示如何使用enhanced_column_mapping模块中的功能
进行智能列名标准化，包括模糊匹配、内容推断和日期格式化。

创建日期: 2024-04-13
作者: BeeShare开发团队
"""

import sys
import os
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

# 导入增强版列标准化模块
try:
    from src.utils.enhanced_column_mapping import (
        standardize_columns,
        standardize_date_format,
        find_closest_column_match,
        infer_column_type,
        get_all_supported_sources
    )
except ImportError:
    logger.error("无法导入增强版列标准化模块，请确保已实现该模块")
    sys.exit(1)


def create_sample_data():
    """创建示例数据"""
    # 创建基本样本数据
    data = {
        # 普通命名的列
        '日期': pd.date_range(start='2023-01-01', periods=5).strftime('%Y-%m-%d').tolist(),
        '开盘价': [100, 101, 102, 103, 104],
        '收盘价': [101, 102, 103, 104, 105],
        '最高价': [105, 106, 107, 108, 109],
        '最低价': [98, 99, 100, 101, 102],
        '成交量': [10000, 11000, 12000, 13000, 14000],
        '成交金额': [1000000, 1100000, 1200000, 1300000, 1400000],
        
        # 非标准列名
        '代码编号': ['000001', '000002', '000003', '000004', '000005'],
        '股票简称': ['平安银行', '万科A', '神州数码', '上海机场', '贵州茅台'],
        '涨跌百分比': [1.2, -0.5, 0.8, -1.0, 2.5],
        
        # 难以识别的列名
        'col_a': [5, 6, 7, 8, 9],   # 无明显特征
        'highest': [106, 107, 108, 109, 110],  # 应与'high'匹配
        'volume_traded': [11000, 12000, 13000, 14000, 15000],  # 应与'volume'匹配
    }
    
    return pd.DataFrame(data)


def demo_basic_mapping():
    """演示基本的列名映射功能"""
    print("\n=== 基本列名映射演示 ===")
    
    # 创建样本数据
    df = create_sample_data()
    print("原始数据列名:")
    print(df.columns.tolist())
    
    # 使用标准映射
    print("\n使用A_SHARE标准映射:")
    result = standardize_columns(df, 'A_SHARE')
    print("标准化后的列名:")
    print(result.columns.tolist())
    print(f"成功映射 {len(set(result.columns) & set(['date', 'open', 'high', 'low', 'close', 'volume', 'amount']))}/7 个核心列")


def demo_fuzzy_matching():
    """演示模糊匹配功能"""
    print("\n=== 模糊匹配功能演示 ===")
    
    # 创建样本数据
    df = create_sample_data()
    
    # 使用精确映射 + 模糊匹配
    print("\n使用精确映射 + 模糊匹配:")
    result = standardize_columns(df, 'A_SHARE', use_fuzzy_match=True)
    print("标准化后的列名:")
    print(result.columns.tolist())
    print("核心列映射结果:")
    core_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'symbol', 'name', 'change_pct']
    for col in core_columns:
        print(f"  {col}: {'✓' if col in result.columns else '✗'}")
    
    # 测试单个列的模糊匹配
    test_columns = ['highest', 'volume_traded', 'opening_price', 'lowest_val', 'closing']
    print("\n单个列名的模糊匹配测试:")
    for col in test_columns:
        match = find_closest_column_match(col, ['high', 'low', 'open', 'close', 'volume', 'amount'])
        print(f"  '{col}' -> '{match}'")


def demo_content_inference():
    """演示内容推断功能"""
    print("\n=== 内容推断功能演示 ===")
    
    # 创建特殊样本数据
    df = pd.DataFrame({
        'unix_time': [1609459200, 1609545600, 1609632000, 1609718400, 1609804800],  # 秒级时间戳
        'large_numbers': [1500000, 1600000, 1700000, 1800000, 1900000],  # 大数值，可能是成交额
        'stock_ids': ['000001', '000002', '000003', '000004', '000005'],  # 股票代码格式
        'percentage': [2.5, 1.8, -0.5, 3.2, -1.1],  # 百分比数值
        'price_values': [10.5, 11.2, 10.8, 11.5, 12.0]  # 价格范围的数值
    })
    
    print("原始数据列名:")
    print(df.columns.tolist())
    
    # 单独测试每个列的推断
    print("\n各列的类型推断结果:")
    for col in df.columns:
        inferred = infer_column_type(df, col)
        inferred_name = inferred.value if inferred else "未知"
        print(f"  '{col}' -> {inferred_name}")
    
    # 使用内容推断进行标准化
    print("\n使用内容推断标准化:")
    result = standardize_columns(df, 'UNKNOWN_SOURCE', infer_column_types=True)
    print("标准化后的列名:")
    print(result.columns.tolist())


def demo_date_format():
    """演示日期格式标准化功能"""
    print("\n=== 日期格式标准化演示 ===")
    
    # 创建包含不同格式日期的数据
    df = pd.DataFrame({
        'iso_dates': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'slash_dates': ['2023/01/01', '2023/01/02', '2023/01/03'],
        'compact_dates': ['20230101', '20230102', '20230103'],
        'chinese_dates': ['2023年01月01日', '2023年01月02日', '2023年01月03日'],
        'unix_seconds': [1672531200, 1672617600, 1672704000],  # 2023-01-01, 02, 03 (秒)
        'unix_millis': [1672531200000, 1672617600000, 1672704000000]  # 毫秒级
    })
    
    print("原始数据:")
    print(df.head(1))
    
    # 标准化各种日期格式
    print("\n标准化后的日期:")
    for col in df.columns:
        # 先重命名为date列
        temp_df = df.rename(columns={col: 'date'})
        # 然后标准化
        result = standardize_date_format(temp_df)
        # 检查结果
        if pd.api.types.is_datetime64_dtype(result['date']):
            print(f"  '{col}' -> {result['date'].dt.strftime('%Y-%m-%d').tolist()[0:3]}")
        else:
            print(f"  '{col}' -> 转换失败")


def demo_custom_source():
    """演示自定义数据源功能"""
    print("\n=== 自定义数据源演示 ===")
    
    # 创建符合自定义源格式的数据
    df = pd.DataFrame({
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
    
    print("自定义源数据的列名:")
    print(df.columns.tolist())
    
    # 获取所有支持的数据源
    sources = get_all_supported_sources()
    print("\n系统支持的所有数据源:")
    print(sources)
    
    # 使用自定义数据源
    if 'CUSTOM_SOURCE' in sources:
        print("\n使用CUSTOM_SOURCE标准化:")
        result = standardize_columns(df, 'CUSTOM_SOURCE')
        print("标准化后的列名:")
        print(result.columns.tolist())
    else:
        print("\n未找到CUSTOM_SOURCE配置，请确认config/column_mappings.json文件存在")


def demo_full_pipeline():
    """演示完整的列标准化流程"""
    print("\n=== 完整列标准化流程演示 ===")
    
    # 创建一个复杂的混合数据集
    complex_data = pd.DataFrame({
        '交易日期': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'StartPrice': [100, 101, 102],  # 非标准名称，需要模糊匹配
        'EndPrice': [101, 102, 103],    # 非标准名称，需要模糊匹配
        'Highest': [105, 106, 107],     # 需要模糊匹配
        'Lowest': [99, 100, 101],       # 需要模糊匹配
        'vol': [10000, 20000, 30000],  # 标准缩写
        'turnover': [1000000, 2000000, 3000000],  # 可能与amount混淆
        'code': ['000001', '000002', '000003'],  # 需要内容推断
        'percentChange': [1.5, -0.8, 2.1],  # 需要内容推断
        'random_data': [42, 43, 44]  # 无法映射的数据
    })
    
    print("原始复杂数据:")
    print(complex_data.head(1))
    
    # 1. 应用完整的标准化流程
    print("\n步骤1: 列名标准化 (使用所有智能特性)")
    result = standardize_columns(
        complex_data, 
        'A_SHARE',  # 尝试使用A股映射
        use_fuzzy_match=True,
        infer_column_types=True
    )
    
    print("标准化后的列名:")
    print(result.columns.tolist())
    
    # 2. 日期格式标准化
    print("\n步骤2: 日期格式标准化")
    if 'date' in result.columns:
        result = standardize_date_format(result)
        print(f"日期列类型: {result['date'].dtype}")
        print(f"日期样本: {result['date'].head(3)}")
    else:
        print("未找到标准化的日期列")
    
    # 3. 最终结果展示
    print("\n最终处理结果:")
    display_cols = [col for col in ['date', 'open', 'close', 'high', 'low', 'volume', 'amount'] if col in result.columns]
    print(result[display_cols].head(3))


if __name__ == "__main__":
    print("=== 增强版列标准化功能演示 ===")
    
    # 顺序运行各个演示函数
    demo_basic_mapping()
    demo_fuzzy_matching()
    demo_content_inference()
    demo_date_format()
    demo_custom_source()
    demo_full_pipeline()
    
    print("\n演示完成!") 