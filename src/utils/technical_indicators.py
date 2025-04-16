#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
技术指标计算工具

提供常用技术分析指标的计算函数，包括：
- 动量指标：RSI、MACD、CCI等
- 波动性指标：布林带、ATR等
- 成交量指标：OBV、MFI等
- 趋势指标：ADX、Aroon等
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators(df, include_all=False):
    """
    为股票数据添加技术指标
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含 open, high, low, close, volume 列
        include_all (bool, optional): 是否计算所有技术指标. Defaults to False.
        
    Returns:
        pandas.DataFrame: 添加了技术指标的数据
    """
    result_df = df.copy()
    
    # 检查必要的列是否存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    
    if missing_columns:
        logger.warning(f"以下列缺失，部分指标可能无法计算: {missing_columns}")
    
    # 添加动量指标
    result_df = add_momentum_indicators(result_df)
    
    # 添加波动性指标
    result_df = add_volatility_indicators(result_df)
    
    # 添加成交量指标
    result_df = add_volume_indicators(result_df)
    
    # 添加趋势指标
    result_df = add_trend_indicators(result_df)
    
    # 如果需要，添加更多指标
    if include_all:
        result_df = add_additional_indicators(result_df)
    
    # 删除计算指标产生的NaN值
    result_df = result_df.dropna()
    
    return result_df

def add_momentum_indicators(df):
    """
    添加动量指标
    
    Args:
        df (pandas.DataFrame): 股票数据
        
    Returns:
        pandas.DataFrame: 添加了动量指标的数据
    """
    result_df = df.copy()
    
    # RSI - 相对强弱指数
    if 'close' in result_df.columns:
        result_df = calculate_rsi(result_df)
    
    # MACD - 指数平滑异同移动平均线
    if 'close' in result_df.columns:
        result_df = calculate_macd(result_df)
    
    # CCI - 顺势指标
    if all(col in result_df.columns for col in ['high', 'low', 'close']):
        result_df = calculate_cci(result_df)
    
    return result_df

def add_volatility_indicators(df):
    """
    添加波动性指标
    
    Args:
        df (pandas.DataFrame): 股票数据
        
    Returns:
        pandas.DataFrame: 添加了波动性指标的数据
    """
    result_df = df.copy()
    
    # 布林带
    if 'close' in result_df.columns:
        result_df = calculate_bollinger_bands(result_df)
    
    # ATR - 真实波动幅度均值
    if all(col in result_df.columns for col in ['high', 'low', 'close']):
        result_df = calculate_atr(result_df)
    
    return result_df

def add_volume_indicators(df):
    """
    添加成交量指标
    
    Args:
        df (pandas.DataFrame): 股票数据
        
    Returns:
        pandas.DataFrame: 添加了成交量指标的数据
    """
    result_df = df.copy()
    
    # OBV - 能量潮
    if all(col in result_df.columns for col in ['close', 'volume']):
        result_df = calculate_obv(result_df)
    
    # MFI - 资金流量指标
    if all(col in result_df.columns for col in ['high', 'low', 'close', 'volume']):
        result_df = calculate_mfi(result_df)
    
    return result_df

def add_trend_indicators(df):
    """
    添加趋势指标
    
    Args:
        df (pandas.DataFrame): 股票数据
        
    Returns:
        pandas.DataFrame: 添加了趋势指标的数据
    """
    result_df = df.copy()
    
    # ADX - 平均方向指数
    if all(col in result_df.columns for col in ['high', 'low', 'close']):
        result_df = calculate_adx(result_df)
    
    # Aroon - 阿隆指标
    if all(col in result_df.columns for col in ['high', 'low']):
        result_df = calculate_aroon(result_df)
    
    return result_df

def add_additional_indicators(df):
    """
    添加其他指标
    
    Args:
        df (pandas.DataFrame): 股票数据
        
    Returns:
        pandas.DataFrame: 添加了其他指标的数据
    """
    result_df = df.copy()
    
    # 可以在这里添加更多的指标
    
    return result_df

# ------------------- 具体指标计算函数 -------------------

def calculate_rsi(df, periods=14):
    """
    计算RSI - 相对强弱指数
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含close列
        periods (int, optional): 计算周期. Defaults to 14.
        
    Returns:
        pandas.DataFrame: 添加了RSI的数据
    """
    result_df = df.copy()
    
    # 计算价格变化
    delta = result_df['close'].diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 计算平均上涨和下跌
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    # 计算相对强度
    rs = avg_gain / avg_loss
    
    # 计算RSI
    result_df['rsi'] = 100 - (100 / (1 + rs))
    
    return result_df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD - 指数平滑异同移动平均线
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含close列
        fast_period (int, optional): 快线周期. Defaults to 12.
        slow_period (int, optional): 慢线周期. Defaults to 26.
        signal_period (int, optional): 信号线周期. Defaults to 9.
        
    Returns:
        pandas.DataFrame: 添加了MACD相关指标的数据
    """
    result_df = df.copy()
    
    # 计算快线和慢线的指数移动平均
    fast_ema = result_df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = result_df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # 计算MACD线
    result_df['macd'] = fast_ema - slow_ema
    
    # 计算信号线
    result_df['macd_signal'] = result_df['macd'].ewm(span=signal_period, adjust=False).mean()
    
    # 计算MACD柱状图
    result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
    
    return result_df

def calculate_cci(df, periods=20):
    """
    计算CCI - 顺势指标
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含high, low, close列
        periods (int, optional): 计算周期. Defaults to 20.
        
    Returns:
        pandas.DataFrame: 添加了CCI的数据
    """
    result_df = df.copy()
    
    # 计算典型价格 (high + low + close) / 3
    tp = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # 计算典型价格的简单移动平均
    tp_ma = tp.rolling(window=periods).mean()
    
    # 计算典型价格的平均偏差
    tp_md = tp.rolling(window=periods).apply(lambda x: np.fabs(x - x.mean()).mean())
    
    # 计算CCI
    result_df['cci'] = (tp - tp_ma) / (0.015 * tp_md)
    
    return result_df

def calculate_bollinger_bands(df, periods=20, deviations=2):
    """
    计算布林带
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含close列
        periods (int, optional): 计算周期. Defaults to 20.
        deviations (int, optional): 标准差倍数. Defaults to 2.
        
    Returns:
        pandas.DataFrame: 添加了布林带指标的数据
    """
    result_df = df.copy()
    
    # 计算中轨（简单移动平均线）
    result_df['bb_middle'] = result_df['close'].rolling(window=periods).mean()
    
    # 计算标准差
    result_df['bb_std'] = result_df['close'].rolling(window=periods).std()
    
    # 计算上轨和下轨
    result_df['bb_upper'] = result_df['bb_middle'] + (result_df['bb_std'] * deviations)
    result_df['bb_lower'] = result_df['bb_middle'] - (result_df['bb_std'] * deviations)
    
    # 计算带宽
    result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
    
    # 删除临时列
    result_df = result_df.drop('bb_std', axis=1)
    
    return result_df

def calculate_atr(df, periods=14):
    """
    计算ATR - 真实波动幅度均值
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含high, low, close列
        periods (int, optional): 计算周期. Defaults to 14.
        
    Returns:
        pandas.DataFrame: 添加了ATR的数据
    """
    result_df = df.copy()
    
    # 计算真实波动幅度 (TR)
    high_low = result_df['high'] - result_df['low']
    high_close_prev = (result_df['high'] - result_df['close'].shift(1)).abs()
    low_close_prev = (result_df['low'] - result_df['close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # 计算ATR
    result_df['atr'] = tr.rolling(window=periods).mean()
    
    return result_df

def calculate_obv(df):
    """
    计算OBV - 能量潮
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含close, volume列
        
    Returns:
        pandas.DataFrame: 添加了OBV的数据
    """
    result_df = df.copy()
    
    # 计算价格变化方向
    price_change = result_df['close'].diff()
    
    # 初始化OBV
    result_df['obv'] = 0
    
    # 根据价格变化方向计算OBV
    result_df.loc[price_change > 0, 'obv'] = result_df['volume']
    result_df.loc[price_change < 0, 'obv'] = -result_df['volume']
    result_df.loc[price_change == 0, 'obv'] = 0
    
    # 计算累积OBV
    result_df['obv'] = result_df['obv'].cumsum()
    
    return result_df

def calculate_mfi(df, periods=14):
    """
    计算MFI - 资金流量指标
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含high, low, close, volume列
        periods (int, optional): 计算周期. Defaults to 14.
        
    Returns:
        pandas.DataFrame: 添加了MFI的数据
    """
    result_df = df.copy()
    
    # 计算典型价格
    tp = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # 计算资金流
    money_flow = tp * result_df['volume']
    
    # 确定价格变化方向
    price_change = tp.diff()
    
    # 初始化正向和负向资金流
    result_df['positive_flow'] = 0
    result_df['negative_flow'] = 0
    
    # 根据价格变化方向计算资金流
    result_df.loc[price_change > 0, 'positive_flow'] = money_flow
    result_df.loc[price_change < 0, 'negative_flow'] = money_flow
    
    # 计算周期内的正向和负向资金流总和
    positive_mf = result_df['positive_flow'].rolling(window=periods).sum()
    negative_mf = result_df['negative_flow'].rolling(window=periods).sum()
    
    # 计算资金流比率
    mf_ratio = positive_mf / negative_mf
    
    # 计算MFI
    result_df['mfi'] = 100 - (100 / (1 + mf_ratio))
    
    # 删除临时列
    result_df = result_df.drop(['positive_flow', 'negative_flow'], axis=1)
    
    return result_df

def calculate_adx(df, periods=14):
    """
    计算ADX - 平均方向指数
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含high, low, close列
        periods (int, optional): 计算周期. Defaults to 14.
        
    Returns:
        pandas.DataFrame: 添加了ADX的数据
    """
    result_df = df.copy()
    
    # 计算方向性指标所需的值
    result_df['tr'] = 0
    result_df['plus_dm'] = 0
    result_df['minus_dm'] = 0
    
    # 计算真实波动幅度 (TR)
    for i in range(1, len(result_df)):
        high_low = result_df['high'].iloc[i] - result_df['low'].iloc[i]
        high_close_prev = abs(result_df['high'].iloc[i] - result_df['close'].iloc[i-1])
        low_close_prev = abs(result_df['low'].iloc[i] - result_df['close'].iloc[i-1])
        
        result_df['tr'].iloc[i] = max(high_low, high_close_prev, low_close_prev)
    
    # 计算+DM和-DM
    for i in range(1, len(result_df)):
        up_move = result_df['high'].iloc[i] - result_df['high'].iloc[i-1]
        down_move = result_df['low'].iloc[i-1] - result_df['low'].iloc[i]
        
        if up_move > down_move and up_move > 0:
            result_df['plus_dm'].iloc[i] = up_move
        else:
            result_df['plus_dm'].iloc[i] = 0
            
        if down_move > up_move and down_move > 0:
            result_df['minus_dm'].iloc[i] = down_move
        else:
            result_df['minus_dm'].iloc[i] = 0
    
    # 计算指数平均
    result_df['tr_periods'] = result_df['tr'].rolling(window=periods).sum()
    result_df['plus_dm_periods'] = result_df['plus_dm'].rolling(window=periods).sum()
    result_df['minus_dm_periods'] = result_df['minus_dm'].rolling(window=periods).sum()
    
    # 计算+DI和-DI
    result_df['plus_di'] = 100 * (result_df['plus_dm_periods'] / result_df['tr_periods'])
    result_df['minus_di'] = 100 * (result_df['minus_dm_periods'] / result_df['tr_periods'])
    
    # 计算DX
    result_df['dx'] = 100 * (abs(result_df['plus_di'] - result_df['minus_di']) / 
                           (result_df['plus_di'] + result_df['minus_di']))
    
    # 计算ADX
    result_df['adx'] = result_df['dx'].rolling(window=periods).mean()
    
    # 删除临时列
    result_df = result_df.drop(['tr', 'plus_dm', 'minus_dm', 'tr_periods', 
                             'plus_dm_periods', 'minus_dm_periods', 'plus_di', 
                             'minus_di', 'dx'], axis=1)
    
    return result_df

def calculate_aroon(df, periods=25):
    """
    计算Aroon指标 - 阿隆指标
    
    Args:
        df (pandas.DataFrame): 股票数据，需要包含high, low列
        periods (int, optional): 计算周期. Defaults to 25.
        
    Returns:
        pandas.DataFrame: 添加了Aroon指标的数据
    """
    result_df = df.copy()
    
    # 初始化Aroon上升和下降指标
    result_df['aroon_up'] = 0
    result_df['aroon_down'] = 0
    
    # 逐行计算Aroon指标
    for i in range(periods, len(result_df)):
        # 获取当前周期的数据子集
        period_data = result_df.iloc[i-periods+1:i+1]
        
        # 计算高点和低点的位置
        high_idx = period_data['high'].idxmax()
        low_idx = period_data['low'].idxmin()
        
        # 计算从高点/低点到当前位置的周期数
        periods_since_high = i - high_idx
        periods_since_low = i - low_idx
        
        # 计算Aroon指标
        result_df.loc[result_df.index[i], 'aroon_up'] = ((periods - periods_since_high) / periods) * 100
        result_df.loc[result_df.index[i], 'aroon_down'] = ((periods - periods_since_low) / periods) * 100
    
    # 计算Aroon震荡指标
    result_df['aroon_osc'] = result_df['aroon_up'] - result_df['aroon_down']
    
    return result_df 