"""
股票数据分析模块

提供各种分析工具，用于分析股票数据，包括技术分析和统计分析
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any, Optional
import talib
from enum import Enum

# 导入列名标准化模块
from src.utils.column_mapping import (
    StandardColumns, 
    detect_and_log_column_issues, 
    standardize_columns,
    suggest_column_mappings
)

logger = logging.getLogger(__name__)

class Analyzer(ABC):
    """
    分析器基类
    
    所有具体分析器都应该继承这个基类并实现分析方法
    """
    
    def __init__(self, data=None):
        """
        初始化分析器
        
        Args:
            data (pandas.DataFrame, optional): 分析数据. Defaults to None.
        """
        self._data = data
        self._standardized = False  # 标记数据是否已经标准化
        
    @property
    def data(self):
        """获取分析数据"""
        return self._data
    
    @data.setter
    def data(self, data):
        """设置分析数据"""
        self._data = data
        self._standardized = False  # 重置标准化标记
        
    def set_data(self, data):
        """
        设置分析数据
        
        Args:
            data (pandas.DataFrame): 分析数据
        """
        self.data = data
    
    def standardize_data(self, source_type="auto"):
        """
        标准化数据列名
        
        Args:
            source_type (str): 数据源类型，如果为auto则自动检测
            
        Returns:
            bool: 是否标准化成功
        """
        if self._data is None or self._standardized:
            return False
            
        try:
            self._data = standardize_columns(self._data, source_type)
            self._standardized = True
            return True
        except Exception as e:
            logger.error(f"标准化数据列名失败: {e}")
            return False
    
    def validate_input(self, required_columns=None):
        """
        验证输入数据
        
        Args:
            required_columns (list, optional): 所需的数据列. 
                                              如果为None，则根据具体分析器需求决定.
        
        Returns:
            bool: 数据是否有效
        """
        # 如果没有数据，返回False
        if self._data is None or self._data.empty:
            logger.error("没有提供数据或数据为空")
            return False
        
        # 如果数据未标准化，尝试标准化
        if not self._standardized:
            self.standardize_data()
        
        # 转换required_columns为StandardColumns枚举
        if required_columns is not None:
            std_columns = []
            for col in required_columns:
                if isinstance(col, str):
                    # 尝试将字符串转换为枚举
                    try:
                        std_columns.append(StandardColumns[col.upper()])
                    except KeyError:
                        std_columns.append(col)  # 如果不是枚举，保持原样
                else:
                    std_columns.append(col)
            
            # 检测并记录列问题
            issues = detect_and_log_column_issues(self._data, std_columns)
            if issues:
                # 尝试获取建议的列映射
                suggestions = suggest_column_mappings(self._data, std_columns)
                if suggestions:
                    logger.info(f"发现可能的列映射: {suggestions}")
                    # 应用建议的列映射
                    for std_col, src_col in suggestions.items():
                        if src_col in self._data.columns:
                            self._data[std_col] = self._data[src_col]
                            logger.info(f"自动映射列 '{src_col}' 到 '{std_col}'")
                
                # 再次检查是否还有问题
                issues = detect_and_log_column_issues(self._data, std_columns)
                if issues:
                    return False
        
        return True
        
    @abstractmethod
    def analyze(self, **kwargs):
        """
        执行分析
        
        需要由子类实现具体分析逻辑
        
        Returns:
            pandas.DataFrame: 分析结果
        """
        raise NotImplementedError("子类必须实现analyze方法")
    
    def visualize(self, **kwargs):
        """
        可视化分析结果
        
        默认实现为空，子类可以根据需要重写此方法
        
        Returns:
            matplotlib.Figure: 可视化结果
        """
        logger.warning("visualize方法尚未在此分析器中实现")
        return None


class MovingAverageAnalyzer(Analyzer):
    """
    移动平均分析器
    """
    
    def validate_input(self):
        """验证输入数据是否包含所需的列"""
        return super().validate_input([StandardColumns.DATE, StandardColumns.CLOSE])
    
    def analyze(self, short_window=5, long_window=20, **kwargs):
        """
        计算移动平均
        
        Args:
            short_window (int): 短期窗口大小
            long_window (int): 长期窗口大小
            
        Returns:
            pandas.DataFrame: 带有移动平均结果的数据框
        """
        try:
            logger.info(f"开始计算移动平均，短期窗口={short_window}，长期窗口={long_window}")
            
            if not self.validate_input():
                logger.error("输入数据验证失败，无法计算移动平均")
                return None
            
            # 确保数据按日期排序
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            df = self._data.sort_values(by=date_col).copy()
            
            # 计算移动平均
            df['MA_short'] = df[close_col].rolling(window=short_window).mean()
            df['MA_long'] = df[close_col].rolling(window=long_window).mean()
            
            # 生成交叉信号：1为金叉(买入)，-1为死叉(卖出)，0为无信号
            df['signal'] = 0
            df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
            df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1
            
            # 检测信号变化
            df['signal_change'] = df['signal'].diff().ne(0).astype(int)
            
            logger.info("移动平均计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算移动平均时发生错误: {e}")
            return None
    
    def visualize(self, result=None, title="移动平均分析", **kwargs):
        """
        可视化移动平均结果
        
        Args:
            result: 分析结果数据框，如果为None则使用analyze方法计算
            title: 图表标题
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        try:
            if result is None:
                logger.info("未提供结果数据，将先执行分析")
                result = self.analyze(**kwargs)
                
            if result is None or result.empty:
                logger.error("无有效数据进行可视化")
                return None
                
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制收盘价
            ax.plot(result[date_col], result[close_col], label='收盘价', alpha=0.7)
            
            # 绘制移动平均线
            ax.plot(result[date_col], result['MA_short'], label=f'短期MA({kwargs.get("short_window", 5)})', linewidth=1.5)
            ax.plot(result[date_col], result['MA_long'], label=f'长期MA({kwargs.get("long_window", 20)})', linewidth=1.5)
            
            # 标记买入卖出点
            buy_signals = result[(result['signal_change'] == 1) & (result['signal'] == 1)]
            sell_signals = result[(result['signal_change'] == 1) & (result['signal'] == -1)]
            
            ax.scatter(buy_signals[date_col], buy_signals[close_col], color='green', marker='^', s=100, label='买入信号')
            ax.scatter(sell_signals[date_col], sell_signals[close_col], color='red', marker='v', s=100, label='卖出信号')
            
            # 设置图形属性
            ax.set_title(title)
            ax.set_xlabel('日期')
            ax.set_ylabel('价格')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"可视化移动平均时发生错误: {e}")
            return None


class RsiAnalyzer(Analyzer):
    """
    RSI(相对强弱指标)分析器
    """
    
    def validate_input(self):
        """验证输入数据是否包含所需的列"""
        return super().validate_input([StandardColumns.DATE, StandardColumns.CLOSE])
    
    def analyze(self, period=14, overbought=70, oversold=30, **kwargs):
        """
        计算RSI指标
        
        Args:
            period (int): RSI计算周期
            overbought (int): 超买阈值
            oversold (int): 超卖阈值
            
        Returns:
            pandas.DataFrame: 带有RSI结果的数据框
        """
        try:
            logger.info(f"开始计算RSI，周期={period}")
            
            if not self.validate_input():
                logger.error("输入数据验证失败，无法计算RSI")
                return None
            
            # 确保数据按日期排序
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            df = self._data.sort_values(by=date_col).copy()
            
            # 计算RSI - 使用TA-Lib
            df['RSI'] = talib.RSI(df[close_col].values, timeperiod=period)
            
            # 生成交易信号: 1为买入(超卖回升)，-1为卖出(超买回落)，0为无信号
            df['signal'] = 0
            
            # 识别超买和超卖区域
            df['overbought'] = df['RSI'] > overbought
            df['oversold'] = df['RSI'] < oversold
            
            # 从超卖区域向上突破为买入信号
            oversold_breakout = (df['RSI'] > oversold) & (df['RSI'].shift(1) <= oversold)
            df.loc[oversold_breakout, 'signal'] = 1
            
            # 从超买区域向下突破为卖出信号
            overbought_breakdown = (df['RSI'] < overbought) & (df['RSI'].shift(1) >= overbought)
            df.loc[overbought_breakdown, 'signal'] = -1
            
            logger.info("RSI计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算RSI时发生错误: {e}")
            return None
    
    def visualize(self, result=None, title="RSI分析", **kwargs):
        """
        可视化RSI结果
        
        Args:
            result: 分析结果数据框，如果为None则使用analyze方法计算
            title: 图表标题
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        try:
            if result is None:
                logger.info("未提供结果数据，将先执行分析")
                result = self.analyze(**kwargs)
                
            if result is None or result.empty:
                logger.error("无有效数据进行可视化")
                return None
            
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            
            period = kwargs.get('period', 14)
            overbought = kwargs.get('overbought', 70)
            oversold = kwargs.get('oversold', 30)
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制收盘价
            ax1.plot(result[date_col], result[close_col], label='收盘价', alpha=0.7)
            
            # 标记买入卖出点
            buy_signals = result[result['signal'] == 1]
            sell_signals = result[result['signal'] == -1]
            
            ax1.scatter(buy_signals[date_col], buy_signals[close_col], color='green', marker='^', s=100, label='买入信号')
            ax1.scatter(sell_signals[date_col], sell_signals[close_col], color='red', marker='v', s=100, label='卖出信号')
            
            # 设置第一个子图属性
            ax1.set_title(title)
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制RSI
            ax2.plot(result[date_col], result['RSI'], color='purple', label=f'RSI({period})')
            
            # 绘制超买超卖线
            ax2.axhline(y=overbought, color='r', linestyle='--', alpha=0.5, label=f'超买({overbought})')
            ax2.axhline(y=oversold, color='g', linestyle='--', alpha=0.5, label=f'超卖({oversold})')
            
            # 设置RSI子图属性
            ax2.set_xlabel('日期')
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"可视化RSI时发生错误: {e}")
            return None


class MacdAnalyzer(Analyzer):
    """
    MACD(指数平滑异同移动平均线)分析器
    """
    
    def validate_input(self):
        """验证输入数据是否包含所需的列"""
        return super().validate_input([StandardColumns.DATE, StandardColumns.CLOSE])
    
    def analyze(self, fast_period=12, slow_period=26, signal_period=9, **kwargs):
        """
        计算MACD指标
        
        Args:
            fast_period (int): 快线周期
            slow_period (int): 慢线周期
            signal_period (int): 信号线周期
            
        Returns:
            pandas.DataFrame: 带有MACD结果的数据框
        """
        try:
            logger.info(f"开始计算MACD，快线周期={fast_period}，慢线周期={slow_period}，信号线周期={signal_period}")
            
            if not self.validate_input():
                logger.error("输入数据验证失败，无法计算MACD")
                return None
            
            # 确保数据按日期排序
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            df = self._data.sort_values(by=date_col).copy()
            
            # 计算MACD - 使用TA-Lib
            macd, macd_signal, macd_hist = talib.MACD(
                df[close_col].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
            
            # 生成交易信号: 1为买入(MACD上穿信号线)，-1为卖出(MACD下穿信号线)，0为无信号
            df['signal'] = 0
            
            # MACD上穿信号线为买入信号
            crossover = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
            df.loc[crossover, 'signal'] = 1
            
            # MACD下穿信号线为卖出信号
            crossunder = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
            df.loc[crossunder, 'signal'] = -1
            
            logger.info("MACD计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算MACD时发生错误: {e}")
            return None
    
    def visualize(self, result=None, title="MACD分析", **kwargs):
        """
        可视化MACD结果
        
        Args:
            result: 分析结果数据框，如果为None则使用analyze方法计算
            title: 图表标题
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        try:
            if result is None:
                logger.info("未提供结果数据，将先执行分析")
                result = self.analyze(**kwargs)
                
            if result is None or result.empty:
                logger.error("无有效数据进行可视化")
                return None
            
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制收盘价
            ax1.plot(result[date_col], result[close_col], label='收盘价', alpha=0.7)
            
            # 标记买入卖出点
            buy_signals = result[result['signal'] == 1]
            sell_signals = result[result['signal'] == -1]
            
            ax1.scatter(buy_signals[date_col], buy_signals[close_col], color='green', marker='^', s=100, label='买入信号')
            ax1.scatter(sell_signals[date_col], sell_signals[close_col], color='red', marker='v', s=100, label='卖出信号')
            
            # 设置第一个子图属性
            ax1.set_title(title)
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 为MACD柱状图创建颜色列表
            colors = ['red' if val < 0 else 'green' for val in result['MACD_hist']]
            
            # 绘制MACD柱状图
            ax2.bar(result[date_col], result['MACD_hist'], color=colors, alpha=0.5, label='MACD直方图')
            
            # 绘制MACD线和信号线
            ax2.plot(result[date_col], result['MACD'], color='blue', label='MACD')
            ax2.plot(result[date_col], result['MACD_signal'], color='red', label='信号线')
            
            # 设置MACD子图属性
            ax2.set_xlabel('日期')
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"可视化MACD时发生错误: {e}")
            return None


class KdjAnalyzer(Analyzer):
    """
    KDJ(随机指标)分析器
    """
    
    def validate_input(self):
        """验证输入数据是否包含所需的列"""
        return super().validate_input([
            StandardColumns.DATE, 
            StandardColumns.HIGH, 
            StandardColumns.LOW, 
            StandardColumns.CLOSE
        ])
    
    def analyze(self, n=9, m1=3, m2=3, **kwargs):
        """
        计算KDJ指标
        
        Args:
            n (int): RSV计算周期
            m1 (int): K值平滑因子
            m2 (int): D值平滑因子
            
        Returns:
            pandas.DataFrame: 带有KDJ结果的数据框
        """
        try:
            logger.info(f"开始计算KDJ，n={n}，m1={m1}，m2={m2}")
            
            if not self.validate_input():
                logger.error("输入数据验证失败，无法计算KDJ")
                return None
            
            # 确保数据按日期排序
            date_col = StandardColumns.DATE.value
            high_col = StandardColumns.HIGH.value
            low_col = StandardColumns.LOW.value
            close_col = StandardColumns.CLOSE.value
            
            df = self._data.sort_values(by=date_col).copy()
            
            # 计算KDJ
            low_list = df[low_col].rolling(window=n).min()
            high_list = df[high_col].rolling(window=n).max()
            
            # 计算RSV
            rsv = (df[close_col] - low_list) / (high_list - low_list) * 100
            
            # 计算K值
            df['K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
            
            # 计算D值
            df['D'] = df['K'].ewm(alpha=1/m2, adjust=False).mean()
            
            # 计算J值
            df['J'] = 3 * df['K'] - 2 * df['D']
            
            # 生成交易信号
            df['signal'] = 0
            
            # K线上穿D线为买入信号
            crossover = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
            df.loc[crossover, 'signal'] = 1
            
            # K线下穿D线为卖出信号
            crossunder = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
            df.loc[crossunder, 'signal'] = -1
            
            logger.info("KDJ计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算KDJ时发生错误: {e}")
            return None
    
    def visualize(self, result=None, title="KDJ分析", **kwargs):
        """
        可视化KDJ结果
        
        Args:
            result: 分析结果数据框，如果为None则使用analyze方法计算
            title: 图表标题
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        try:
            if result is None:
                logger.info("未提供结果数据，将先执行分析")
                result = self.analyze(**kwargs)
                
            if result is None or result.empty:
                logger.error("无有效数据进行可视化")
                return None
            
            date_col = StandardColumns.DATE.value
            close_col = StandardColumns.CLOSE.value
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制收盘价
            ax1.plot(result[date_col], result[close_col], label='收盘价', alpha=0.7)
            
            # 标记买入卖出点
            buy_signals = result[result['signal'] == 1]
            sell_signals = result[result['signal'] == -1]
            
            ax1.scatter(buy_signals[date_col], buy_signals[close_col], color='green', marker='^', s=100, label='买入信号')
            ax1.scatter(sell_signals[date_col], sell_signals[close_col], color='red', marker='v', s=100, label='卖出信号')
            
            # 设置第一个子图属性
            ax1.set_title(title)
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制KDJ
            ax2.plot(result[date_col], result['K'], color='blue', label='K')
            ax2.plot(result[date_col], result['D'], color='orange', label='D')
            ax2.plot(result[date_col], result['J'], color='purple', label='J', alpha=0.6)
            
            # 绘制超买超卖线
            ax2.axhline(y=80, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=20, color='g', linestyle='--', alpha=0.3)
            
            # 设置KDJ子图属性
            ax2.set_xlabel('日期')
            ax2.set_ylabel('KDJ')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"可视化KDJ时发生错误: {e}")
            return None 