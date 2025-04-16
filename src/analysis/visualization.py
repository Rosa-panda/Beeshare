"""
可视化模块

提供股票数据可视化功能，包括K线图、交易量图、技术指标图等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import seaborn as sns
from datetime import datetime, timedelta
import logging
from . import Analyzer

try:
    import mplfinance as mpf
    MPLFINANCE_INSTALLED = True
except ImportError:
    MPLFINANCE_INSTALLED = False
    logging.warning("mplfinance库未安装，某些高级可视化功能将不可用。可以通过pip install mplfinance安装。")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False
    logging.warning("plotly库未安装，交互式可视化功能将不可用。可以通过pip install plotly安装。")

logger = logging.getLogger(__name__)

class StockVisualizer(Analyzer):
    """股票数据可视化工具，提供各种可视化功能"""
    
    def __init__(self, data=None):
        """
        初始化可视化工具
        
        Args:
            data (pandas.DataFrame, optional): 股票数据，需要包含OHLC和成交量数据. Defaults to None.
        """
        super().__init__(data)
        
    def plot_candlestick(self, start_date=None, end_date=None, title=None, volume=True, figsize=(12, 8)):
        """
        绘制K线图
        
        Args:
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            volume (bool, optional): 是否显示成交量. Defaults to True.
            figsize (tuple, optional): 图表大小. Defaults to (12, 8).
            
        Returns:
            matplotlib.Figure: K线图
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可供可视化")
            return None
            
        # 确保数据包含OHLC
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            logger.error("数据缺少OHLC列")
            return None
            
        # 如果要显示成交量，确保数据包含成交量列
        if volume and 'volume' not in self.data.columns:
            logger.warning("数据缺少volume列，将不显示成交量")
            volume = False
            
        # 过滤日期范围
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if df.empty:
            logger.error("过滤后没有数据可供可视化")
            return None
            
        if MPLFINANCE_INSTALLED:
            # 使用mplfinance绘制K线图
            # 需要将日期索引转换为datetime类型
            df.index = pd.to_datetime(df.index)
            
            # 设置mplfinance样式
            mc = mpf.make_marketcolors(
                up='red', down='green',
                edge='inherit',
                wick='black',
                volume='inherit'
            )
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='--',
                y_on_right=False
            )
            
            # 创建图表
            kwargs = {'type': 'candle', 'style': s, 'figsize': figsize}
            
            if volume:
                kwargs['volume'] = True
                
            if title:
                kwargs['title'] = title
            elif 'symbol' in df.columns:
                kwargs['title'] = f"{df['symbol'].iloc[0]} K线图"
                
            # 绘制K线图
            fig, axes = mpf.plot(df, **kwargs, returnfig=True)
            
            return fig
        else:
            # 如果没有安装mplfinance，使用matplotlib绘制简单的K线图
            fig = plt.figure(figsize=figsize)
            
            # 设置网格
            if volume:
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1], sharex=ax1)
            else:
                ax1 = plt.subplot(1, 1, 1)
                ax2 = None
                
            # 绘制K线图
            # 将日期索引转换为matplotlib可用的格式
            df['date_int'] = mdates.date2num(df.index.to_pydatetime())
            
            # 准备OHLC数据
            ohlc = df[['date_int', 'open', 'high', 'low', 'close']].values
            
            # 绘制K线
            candlestick_ohlc(ax1, ohlc, width=0.6, colorup='red', colordown='green')
            
            # 设置x轴格式
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(date_format)
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            
            # 设置标题
            if title:
                ax1.set_title(title)
            elif 'symbol' in df.columns:
                ax1.set_title(f"{df['symbol'].iloc[0]} K线图")
                
            # 设置网格线
            ax1.grid(True)
            
            # 如果需要显示成交量，绘制成交量柱状图
            if volume and ax2 is not None:
                # 根据收盘价是否上涨设置颜色
                colors = ['red' if close >= open else 'green' for open, close in zip(df['open'], df['close'])]
                ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
                ax2.set_ylabel('成交量')
                ax2.grid(True)
                
            plt.tight_layout()
            return fig
            
    def plot_interactive_candlestick(self, start_date=None, end_date=None, title=None, indicators=None):
        """
        绘制交互式K线图（使用plotly）
        
        Args:
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            indicators (list, optional): 要显示的技术指标列表. Defaults to None.
            
        Returns:
            plotly.graph_objects.Figure: 交互式K线图
        """
        if not PLOTLY_INSTALLED:
            logger.error("plotly库未安装，无法创建交互式图表")
            return None
            
        if self.data is None or self.data.empty:
            logger.error("没有数据可供可视化")
            return None
            
        # 确保数据包含OHLC
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            logger.error("数据缺少OHLC列")
            return None
            
        # 过滤日期范围
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if df.empty:
            logger.error("过滤后没有数据可供可视化")
            return None
            
        # 确定是否显示成交量和技术指标
        show_volume = 'volume' in df.columns
        
        # 如果未指定技术指标，查找可用的技术指标列
        if indicators is None:
            indicators = []
            # 查找常见技术指标列
            for col in df.columns:
                if col.startswith('ma') or col.startswith('ema') or col.startswith('boll'):
                    indicators.append(col)
                    
        # 设置子图数量
        n_subplots = 1 + show_volume + (len(indicators) > 0)
        
        # 确定子图高度比例
        if n_subplots == 1:
            row_heights = [1]
            row_specs = [[{"type": "xy"}]]
        elif n_subplots == 2:
            if show_volume:
                row_heights = [0.8, 0.2]
                row_specs = [[{"type": "xy"}], [{"type": "xy"}]]
            else:  # 只有K线和指标
                row_heights = [0.7, 0.3]
                row_specs = [[{"type": "xy"}], [{"type": "xy"}]]
        else:  # n_subplots == 3
            row_heights = [0.6, 0.2, 0.2]
            row_specs = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]]
            
        # 创建子图
        fig = make_subplots(
            rows=n_subplots, 
            cols=1, 
            shared_xaxes=True, 
            row_heights=row_heights,
            row_specs=row_specs,
            vertical_spacing=0.02
        )
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',
                decreasing_line_color='green'
            ),
            row=1, col=1
        )
        
        # 添加技术指标
        subplot_idx = 2
        if len(indicators) > 0:
            for indicator in indicators:
                if indicator in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[indicator],
                            mode='lines',
                            name=indicator
                        ),
                        row=1, col=1
                    )
                    
        # 添加成交量
        if show_volume:
            # 为上涨和下跌创建不同颜色的成交量柱
            colors = ['red' if close >= open else 'green' for open, close in zip(df['open'], df['close'])]
            
            for i, (date, volume) in enumerate(zip(df.index, df['volume'])):
                fig.add_trace(
                    go.Bar(
                        x=[date],
                        y=[volume],
                        marker_color=colors[i],
                        showlegend=False
                    ),
                    row=subplot_idx, col=1
                )
            subplot_idx += 1
            
        # 设置图表布局
        if title:
            fig.update_layout(title=title)
        elif 'symbol' in df.columns:
            fig.update_layout(title=f"{df['symbol'].iloc[0]} K线图")
            
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    def plot_price_volume(self, start_date=None, end_date=None, ma_periods=None, title=None, figsize=(12, 10)):
        """
        绘制价格-成交量图
        
        Args:
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            ma_periods (list, optional): 移动平均线周期列表. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            figsize (tuple, optional): 图表大小. Defaults to (12, 10).
            
        Returns:
            matplotlib.Figure: 价格-成交量图
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可供可视化")
            return None
            
        # 确保数据包含收盘价
        if 'close' not in self.data.columns:
            logger.error("数据缺少close列")
            return None
            
        # 确保数据包含成交量
        if 'volume' not in self.data.columns:
            logger.warning("数据缺少volume列，将只显示价格图")
            
        # 过滤日期范围
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if df.empty:
            logger.error("过滤后没有数据可供可视化")
            return None
            
        # 如果未指定移动平均线周期，使用默认值
        if ma_periods is None:
            ma_periods = [5, 20, 60]
            
        # 计算移动平均线
        for period in ma_periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
            
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制价格和移动平均线
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='收盘价', linewidth=2)
        
        for period in ma_periods:
            ax1.plot(df.index, df[f'ma{period}'], label=f'{period}日均线')
            
        ax1.set_title(title or f"{df['symbol'].iloc[0] if 'symbol' in df.columns else ''} 价格与成交量")
        ax1.set_ylabel('价格')
        ax1.grid(True)
        ax1.legend()
        
        # 绘制成交量
        ax2 = axes[1]
        if 'volume' in df.columns:
            # 为上涨和下跌创建不同颜色的成交量柱
            colors = ['red' if close >= open else 'green' for open, close in zip(df['open'], df['close'])] if 'open' in df.columns else 'blue'
            
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('成交量')
            ax2.grid(True)
            
            # 计算成交量移动平均线
            if len(df) > 5:
                vol_ma = df['volume'].rolling(window=5).mean()
                ax2.plot(df.index, vol_ma, 'k--', label='5日均量')
                ax2.legend()
                
        # 设置x轴日期格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    def plot_comparative(self, symbols=None, data_dict=None, start_date=None, end_date=None, title=None, 
                         normalize=True, figsize=(12, 6)):
        """
        绘制多支股票对比图
        
        Args:
            symbols (list, optional): 要对比的股票代码列表. Defaults to None.
            data_dict (dict, optional): 股票数据字典，格式为{symbol: df}. Defaults to None.
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            normalize (bool, optional): 是否标准化价格. Defaults to True.
            figsize (tuple, optional): 图表大小. Defaults to (12, 6).
            
        Returns:
            matplotlib.Figure: 股票对比图
        """
        if data_dict is None:
            data_dict = {}
            
        # 如果提供了symbols但未提供data_dict，使用self.data中的数据
        if symbols is not None and not data_dict and self.data is not None:
            if 'symbol' in self.data.columns:
                # 数据中包含多支股票
                for symbol in symbols:
                    symbol_data = self.data[self.data['symbol'] == symbol]
                    if not symbol_data.empty:
                        data_dict[symbol] = symbol_data
            elif len(symbols) == 1:
                # 数据只包含一支股票
                data_dict[symbols[0]] = self.data
                
        if not data_dict:
            logger.error("没有数据可供可视化")
            return None
            
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 处理每支股票的数据
        for symbol, df in data_dict.items():
            # 确保数据包含收盘价
            if 'close' not in df.columns:
                logger.warning(f"股票 {symbol} 数据缺少close列，将跳过")
                continue
                
            # 过滤日期范围
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            if df.empty:
                logger.warning(f"股票 {symbol} 过滤后没有数据，将跳过")
                continue
                
            # 处理标准化
            if normalize:
                # 计算相对收益率
                first_close = df['close'].iloc[0]
                prices = df['close'] / first_close - 1
                ylabel = '相对收益率'
            else:
                prices = df['close']
                ylabel = '收盘价'
                
            # 绘制股票价格
            ax.plot(df.index, prices, label=symbol)
            
        # 设置图表属性
        ax.set_title(title or '股票对比图')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    def plot_correlation_heatmap(self, symbols=None, data_dict=None, start_date=None, end_date=None, title=None, 
                                figsize=(10, 8), method='pearson'):
        """
        绘制股票相关性热力图
        
        Args:
            symbols (list, optional): 要分析的股票代码列表. Defaults to None.
            data_dict (dict, optional): 股票数据字典，格式为{symbol: df}. Defaults to None.
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            figsize (tuple, optional): 图表大小. Defaults to (10, 8).
            method (str, optional): 相关性计算方法，'pearson'或'spearman'. Defaults to 'pearson'.
            
        Returns:
            matplotlib.Figure: 相关性热力图
        """
        if data_dict is None:
            data_dict = {}
            
        # 如果提供了symbols但未提供data_dict，使用self.data中的数据
        if symbols is not None and not data_dict and self.data is not None:
            if 'symbol' in self.data.columns:
                # 数据中包含多支股票
                for symbol in symbols:
                    symbol_data = self.data[self.data['symbol'] == symbol]
                    if not symbol_data.empty:
                        data_dict[symbol] = symbol_data
                        
        if not data_dict:
            logger.error("没有数据可供可视化")
            return None
            
        # 准备收益率数据
        returns_data = {}
        for symbol, df in data_dict.items():
            # 确保数据包含收盘价
            if 'close' not in df.columns:
                logger.warning(f"股票 {symbol} 数据缺少close列，将跳过")
                continue
                
            # 过滤日期范围
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            if df.empty:
                logger.warning(f"股票 {symbol} 过滤后没有数据，将跳过")
                continue
                
            # 计算收益率
            returns_data[symbol] = df['close'].pct_change().dropna()
            
        if not returns_data:
            logger.error("没有有效的收益率数据可供分析")
            return None
            
        # 创建收益率数据框
        returns_df = pd.DataFrame(returns_data)
        
        # 计算相关性矩阵
        corr_matrix = returns_df.corr(method=method)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        
        # 设置图表属性
        ax.set_title(title or '股票收益率相关性热力图')
        
        plt.tight_layout()
        return fig
        
    def plot_heatmap(self, data=None, index_col=None, value_col=None, title=None, figsize=(12, 10)):
        """
        绘制热力图（例如，按月日分布的收益率热力图）
        
        Args:
            data (pandas.DataFrame, optional): 用于绘制热力图的数据. Defaults to None.
            index_col (str, optional): 用作行索引的列名. Defaults to None.
            value_col (str, optional): 用作值的列名. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            figsize (tuple, optional): 图表大小. Defaults to (12, 10).
            
        Returns:
            matplotlib.Figure: 热力图
        """
        # 使用self.data或传入的数据
        df = data if data is not None else self.data
        
        if df is None or df.empty:
            logger.error("没有数据可供可视化")
            return None
            
        # 如果未指定索引列和值列，尝试使用默认设置
        if index_col is None or value_col is None:
            # 默认情况下，尝试使用日期索引和收益率
            if isinstance(df.index, pd.DatetimeIndex) and 'close' in df.columns:
                # 计算每日收益率
                df['daily_return'] = df['close'].pct_change()
                
                # 提取月份和日期
                df['month'] = df.index.month
                df['day'] = df.index.day
                
                # 设置索引列和值列
                index_col = 'month'
                value_col = 'daily_return'
            else:
                logger.error("未能确定合适的索引列和值列")
                return None
                
        # 创建透视表
        if index_col == 'month' and 'day' in df.columns:
            # 如果是月日热力图，创建特定的透视表
            pivot_data = df.pivot_table(index=index_col, columns='day', values=value_col, aggfunc='mean')
            
            # 设置月份名称
            month_names = ['一月', '二月', '三月', '四月', '五月', '六月', 
                           '七月', '八月', '九月', '十月', '十一月', '十二月']
            pivot_data.index = [month_names[i-1] for i in pivot_data.index]
        else:
            # 一般情况下，使用提供的索引列和值列
            if 'columns' in df.columns:
                pivot_data = df.pivot_table(index=index_col, columns='columns', values=value_col)
            else:
                # 如果没有指定列，直接使用数据
                pivot_data = df
                
        # 创建热力图
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', ax=ax)
        
        # 设置图表属性
        ax.set_title(title or f'{value_col}热力图')
        
        plt.tight_layout()
        return fig
        
    def create_dashboard(self, start_date=None, end_date=None, title=None, figsize=(16, 12)):
        """
        创建股票数据综合仪表板
        
        Args:
            start_date (str, optional): 开始日期. Defaults to None.
            end_date (str, optional): 结束日期. Defaults to None.
            title (str, optional): 图表标题. Defaults to None.
            figsize (tuple, optional): 图表大小. Defaults to (16, 12).
            
        Returns:
            matplotlib.Figure: 仪表板图表
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可供可视化")
            return None
            
        # 确保数据包含必要的列
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            logger.error("数据缺少OHLC列")
            return None
            
        # 过滤日期范围
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if df.empty:
            logger.error("过滤后没有数据可供可视化")
            return None
            
        # 计算技术指标
        # 移动平均线
        for period in [5, 20, 60]:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
            
        # 相对强弱指标(RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + (df['boll_std'] * 2)
        df['boll_lower'] = df['boll_mid'] - (df['boll_std'] * 2)
        
        # 计算每日收益率
        df['daily_return'] = df['close'].pct_change()
        
        # 计算累计收益率
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # 创建仪表板布局
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1, 1])
        
        # 创建子图
        ax_price = plt.subplot(gs[0, :])
        ax_volume = plt.subplot(gs[1, :], sharex=ax_price)
        ax_returns = plt.subplot(gs[2, :2])
        ax_histogram = plt.subplot(gs[2, 2:])
        
        # 绘制价格和移动平均线
        ax_price.plot(df.index, df['close'], label='收盘价', linewidth=2)
        ax_price.plot(df.index, df['ma5'], label='5日均线')
        ax_price.plot(df.index, df['ma20'], label='20日均线')
        ax_price.plot(df.index, df['ma60'], label='60日均线')
        
        # 绘制布林带
        ax_price.plot(df.index, df['boll_upper'], 'k--', label='布林上轨')
        ax_price.plot(df.index, df['boll_lower'], 'k--', label='布林下轨')
        ax_price.fill_between(df.index, df['boll_upper'], df['boll_lower'], alpha=0.1)
        
        # 设置价格图属性
        stock_name = df['symbol'].iloc[0] if 'symbol' in df.columns else ''
        ax_price.set_title(title or f"{stock_name} 仪表板")
        ax_price.set_ylabel('价格')
        ax_price.grid(True)
        ax_price.legend()
        
        # 绘制成交量
        if 'volume' in df.columns:
            # 为上涨和下跌创建不同颜色的成交量柱
            colors = ['red' if close >= open else 'green' for open, close in zip(df['open'], df['close'])]
            
            ax_volume.bar(df.index, df['volume'], color=colors, alpha=0.7)
            ax_volume.set_ylabel('成交量')
            ax_volume.grid(True)
            
            # 计算成交量移动平均线
            if len(df) > 5:
                vol_ma = df['volume'].rolling(window=5).mean()
                ax_volume.plot(df.index, vol_ma, 'k--', label='5日均量')
                ax_volume.legend()
                
        # 绘制累计收益率
        ax_returns.plot(df.index, df['cumulative_return'] * 100)
        ax_returns.set_title('累计收益率')
        ax_returns.set_ylabel('收益率 (%)')
        ax_returns.grid(True)
        
        # 绘制每日收益率分布直方图
        returns = df['daily_return'].dropna()
        ax_histogram.hist(returns * 100, bins=50, alpha=0.7)
        ax_histogram.axvline(0, color='r', linestyle='--')
        ax_histogram.set_title('每日收益率分布')
        ax_histogram.set_xlabel('每日收益率 (%)')
        ax_histogram.grid(True)
        
        # 设置x轴日期格式
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    def analyze(self):
        """
        执行可视化分析
        
        这个方法作为Analyzer接口的实现，但实际上应该使用具体的可视化方法
        
        Returns:
            dict: 可视化分析结果（空字典）
        """
        logger.warning("Analyzer.analyze()方法在StockVisualizer中没有具体实现，请使用具体的可视化方法")
        return {}
