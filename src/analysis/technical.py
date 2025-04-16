"""
技术分析模块

提供股票技术分析功能，包括移动平均线、MACD、RSI、KDJ、布林带等指标的计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import Analyzer
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer(Analyzer):
    """技术分析器，提供常用技术指标计算"""
    
    def __init__(self, data=None):
        """
        初始化技术分析器
        
        Args:
            data (pandas.DataFrame, optional): 分析数据. Defaults to None.
        """
        super().__init__(data)
        
    def analyze(self, methods=None, params=None):
        """
        执行技术分析
        
        Args:
            methods (list, optional): 要执行的分析方法列表. Defaults to None.
            params (dict, optional): 分析参数字典. Defaults to None.
            
        Returns:
            pandas.DataFrame: 包含分析结果的DataFrame
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可供分析")
            return pd.DataFrame()
            
        # 如果未指定方法，默认计算所有指标
        if methods is None:
            methods = ['ma', 'ema', 'macd', 'rsi', 'kdj', 'boll']
            
        # 如果未指定参数，使用默认参数
        if params is None:
            params = {}
            
        # 复制数据，避免修改原始数据
        result = self.data.copy()
        
        # 执行指定的分析方法
        for method in methods:
            if method == 'ma':
                periods = params.get('ma_periods', [5, 10, 20, 60, 120, 250])
                for period in periods:
                    result = self.calculate_ma(result, period)
            elif method == 'ema':
                periods = params.get('ema_periods', [5, 10, 20, 60])
                for period in periods:
                    result = self.calculate_ema(result, period)
            elif method == 'macd':
                fast = params.get('macd_fast', 12)
                slow = params.get('macd_slow', 26)
                signal = params.get('macd_signal', 9)
                result = self.calculate_macd(result, fast, slow, signal)
            elif method == 'rsi':
                period = params.get('rsi_period', 14)
                result = self.calculate_rsi(result, period)
            elif method == 'kdj':
                k_period = params.get('k_period', 9)
                d_period = params.get('d_period', 3)
                j_period = params.get('j_period', 3)
                result = self.calculate_kdj(result, k_period, d_period, j_period)
            elif method == 'boll':
                period = params.get('boll_period', 20)
                std_dev = params.get('boll_std_dev', 2)
                result = self.calculate_boll(result, period, std_dev)
            else:
                logger.warning(f"未知的分析方法: {method}")
                
        return result
    
    def calculate_ma(self, data, period=20):
        """
        计算移动平均线
        
        Args:
            data (pandas.DataFrame): 股票数据
            period (int, optional): 周期. Defaults to 20.
            
        Returns:
            pandas.DataFrame: 添加了移动平均线的数据
        """
        col_name = f'ma{period}'
        data[col_name] = data['close'].rolling(window=period).mean()
        return data
        
    def calculate_ema(self, data, period=20):
        """
        计算指数移动平均线
        
        Args:
            data (pandas.DataFrame): 股票数据
            period (int, optional): 周期. Defaults to 20.
            
        Returns:
            pandas.DataFrame: 添加了指数移动平均线的数据
        """
        col_name = f'ema{period}'
        data[col_name] = data['close'].ewm(span=period, adjust=False).mean()
        return data
        
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        
        Args:
            data (pandas.DataFrame): 股票数据
            fast_period (int, optional): 快线周期. Defaults to 12.
            slow_period (int, optional): 慢线周期. Defaults to 26.
            signal_period (int, optional): 信号线周期. Defaults to 9.
            
        Returns:
            pandas.DataFrame: 添加了MACD指标的数据
        """
        # 计算快线和慢线的EMA
        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF(MACD线)
        data['macd_dif'] = ema_fast - ema_slow
        
        # 计算DEA(信号线)
        data['macd_dea'] = data['macd_dif'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        data['macd_hist'] = (data['macd_dif'] - data['macd_dea']) * 2
        
        return data
        
    def calculate_rsi(self, data, period=14):
        """
        计算RSI指标
        
        Args:
            data (pandas.DataFrame): 股票数据
            period (int, optional): 周期. Defaults to 14.
            
        Returns:
            pandas.DataFrame: 添加了RSI指标的数据
        """
        # 计算价格变化
        delta = data['close'].diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        data[f'rsi{period}'] = 100 - (100 / (1 + rs))
        
        return data
        
    def calculate_kdj(self, data, k_period=9, d_period=3, j_period=3):
        """
        计算KDJ指标
        
        Args:
            data (pandas.DataFrame): 股票数据
            k_period (int, optional): K值周期. Defaults to 9.
            d_period (int, optional): D值周期. Defaults to 3.
            j_period (int, optional): J值周期. Defaults to 3.
            
        Returns:
            pandas.DataFrame: 添加了KDJ指标的数据
        """
        # 计算最低价和最高价
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        # 计算RSV
        rsv = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # 计算K值
        data['kdj_k'] = rsv.ewm(alpha=1/d_period, adjust=False).mean()
        
        # 计算D值
        data['kdj_d'] = data['kdj_k'].ewm(alpha=1/j_period, adjust=False).mean()
        
        # 计算J值
        data['kdj_j'] = 3 * data['kdj_k'] - 2 * data['kdj_d']
        
        return data
        
    def calculate_boll(self, data, period=20, std_dev=2):
        """
        计算布林带指标
        
        Args:
            data (pandas.DataFrame): 股票数据
            period (int, optional): 周期. Defaults to 20.
            std_dev (int, optional): 标准差倍数. Defaults to 2.
            
        Returns:
            pandas.DataFrame: 添加了布林带指标的数据
        """
        # 计算中轨线(SMA)
        data['boll_mid'] = data['close'].rolling(window=period).mean()
        
        # 计算标准差
        data['boll_std'] = data['close'].rolling(window=period).std()
        
        # 计算上轨线
        data['boll_upper'] = data['boll_mid'] + (data['boll_std'] * std_dev)
        
        # 计算下轨线
        data['boll_lower'] = data['boll_mid'] - (data['boll_std'] * std_dev)
        
        return data
        
    def calculate_trend(self, data, short_period=5, long_period=20):
        """
        判断股票趋势
        
        Args:
            data (pandas.DataFrame): 股票数据
            short_period (int, optional): 短期均线周期. Defaults to 5.
            long_period (int, optional): 长期均线周期. Defaults to 20.
            
        Returns:
            pandas.DataFrame: 添加了趋势判断的数据
        """
        # 确保已计算短期和长期均线
        data = self.calculate_ma(data, short_period)
        data = self.calculate_ma(data, long_period)
        
        # 判断趋势
        data['trend'] = 'sideways'  # 默认为横盘
        
        # 上涨趋势：短期均线在长期均线之上
        data.loc[data[f'ma{short_period}'] > data[f'ma{long_period}'], 'trend'] = 'uptrend'
        
        # 下跌趋势：短期均线在长期均线之下
        data.loc[data[f'ma{short_period}'] < data[f'ma{long_period}'], 'trend'] = 'downtrend'
        
        return data
        
    def calculate_support_resistance(self, data, window=20):
        """
        计算支撑位和阻力位
        
        Args:
            data (pandas.DataFrame): 股票数据
            window (int, optional): 窗口大小. Defaults to 20.
            
        Returns:
            pandas.DataFrame: 添加了支撑位和阻力位的数据
        """
        # 初始化支撑位和阻力位列
        data['support'] = np.nan
        data['resistance'] = np.nan
        
        # 滑动窗口计算局部最低和最高点
        for i in range(window, len(data)):
            # 取当前窗口的数据
            window_data = data.iloc[i-window:i]
            
            # 找到局部最低点
            if data.iloc[i-window//2]['low'] == window_data['low'].min():
                data.loc[data.index[i-window//2], 'support'] = data.iloc[i-window//2]['low']
                
            # 找到局部最高点
            if data.iloc[i-window//2]['high'] == window_data['high'].max():
                data.loc[data.index[i-window//2], 'resistance'] = data.iloc[i-window//2]['high']
                
        return data
    
    def visualize(self, data=None, indicators=None, start_date=None, end_date=None, title=None):
        """
        可视化技术分析结果
        
        Args:
            data (pandas.DataFrame, optional): 要可视化的数据. Defaults to None.
            indicators (list, optional): 要可视化的指标列表. Defaults to None.
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        # 使用类中的数据或传入的数据
        df = data if data is not None else self.data
        
        if df is None or df.empty:
            logger.error("没有数据可供可视化")
            return None
            
        # 过滤日期范围
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # 如果未指定指标，使用所有可用的技术指标
        if indicators is None:
            # 查找所有技术指标列
            ma_cols = [col for col in df.columns if col.startswith('ma')]
            ema_cols = [col for col in df.columns if col.startswith('ema')]
            boll_cols = [col for col in df.columns if col.startswith('boll')]
            macd_cols = [col for col in df.columns if col.startswith('macd')]
            rsi_cols = [col for col in df.columns if col.startswith('rsi')]
            kdj_cols = [col for col in df.columns if col.startswith('kdj')]
            
            # 组合所有指标
            indicators = ['close'] + ma_cols + ema_cols
            
            # 如果有布林带指标，添加到指标列表
            if any(boll_cols):
                indicators.extend(boll_cols)
                
            # MACD, RSI, KDJ需要单独的子图表
            has_macd = any(macd_cols)
            has_rsi = any(rsi_cols)
            has_kdj = any(kdj_cols)
            
            # 设置图表数量
            n_subplots = 1
            if has_macd: n_subplots += 1
            if has_rsi: n_subplots += 1
            if has_kdj: n_subplots += 1
            
            # 创建图表
            fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 8 * n_subplots), sharex=True)
            
            # 如果只有一个子图，将axes转换为列表便于索引
            if n_subplots == 1:
                axes = [axes]
                
            # 绘制价格和均线
            ax = axes[0]
            ax.plot(df.index, df['close'], label='收盘价')
            
            for col in ma_cols:
                ax.plot(df.index, df[col], label=col)
                
            for col in ema_cols:
                ax.plot(df.index, df[col], label=col, linestyle='--')
                
            # 如果有布林带指标，绘制布林带
            if any(boll_cols):
                ax.plot(df.index, df['boll_mid'], label='布林中轨', color='g')
                ax.plot(df.index, df['boll_upper'], label='布林上轨', color='r')
                ax.plot(df.index, df['boll_lower'], label='布林下轨', color='b')
                ax.fill_between(df.index, df['boll_upper'], df['boll_lower'], alpha=0.1, color='g')
                
            ax.set_title(title or f"{df['symbol'].iloc[0] if 'symbol' in df.columns else ''} 价格和移动平均线")
            ax.set_ylabel('价格')
            ax.grid(True)
            ax.legend()
            
            # 绘制MACD
            subplot_idx = 1
            if has_macd:
                ax = axes[subplot_idx]
                ax.plot(df.index, df['macd_dif'], label='MACD线', color='b')
                ax.plot(df.index, df['macd_dea'], label='信号线', color='r')
                ax.bar(df.index, df['macd_hist'], label='柱状图', color=['g' if x >= 0 else 'r' for x in df['macd_hist']])
                ax.set_title('MACD')
                ax.set_ylabel('值')
                ax.grid(True)
                ax.legend()
                subplot_idx += 1
                
            # 绘制RSI
            if has_rsi:
                ax = axes[subplot_idx]
                for col in rsi_cols:
                    ax.plot(df.index, df[col], label=col)
                ax.axhline(70, color='r', linestyle='--')
                ax.axhline(30, color='g', linestyle='--')
                ax.set_title('RSI')
                ax.set_ylabel('值')
                ax.grid(True)
                ax.legend()
                subplot_idx += 1
                
            # 绘制KDJ
            if has_kdj:
                ax = axes[subplot_idx]
                ax.plot(df.index, df['kdj_k'], label='K线', color='b')
                ax.plot(df.index, df['kdj_d'], label='D线', color='y')
                ax.plot(df.index, df['kdj_j'], label='J线', color='m')
                ax.axhline(80, color='r', linestyle='--')
                ax.axhline(20, color='g', linestyle='--')
                ax.set_title('KDJ')
                ax.set_ylabel('值')
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            return fig
        else:
            # 如果指定了指标，只绘制指定的指标
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for indicator in indicators:
                if indicator in df.columns:
                    ax.plot(df.index, df[indicator], label=indicator)
                else:
                    logger.warning(f"指标 {indicator} 在数据中不存在")
            
            ax.set_title(title or f"{df['symbol'].iloc[0] if 'symbol' in df.columns else ''} 技术指标")
            ax.set_xlabel('日期')
            ax.set_ylabel('值')
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            return fig
