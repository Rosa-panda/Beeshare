"""
统计分析模块

提供股票数据的统计分析功能，包括收益率统计、风险指标计算、相关性分析等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from . import Analyzer
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer(Analyzer):
    """统计分析器，提供常用统计分析功能"""
    
    def __init__(self, data=None):
        """
        初始化统计分析器
        
        Args:
            data (pandas.DataFrame, optional): 分析数据. Defaults to None.
        """
        super().__init__(data)
        
    def analyze(self, methods=None, params=None):
        """
        执行统计分析
        
        Args:
            methods (list, optional): 要执行的分析方法列表. Defaults to None.
            params (dict, optional): 分析参数字典. Defaults to None.
            
        Returns:
            dict: 包含分析结果的字典
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可供分析")
            return {}
            
        # 如果未指定方法，默认使用所有分析方法
        if methods is None:
            methods = ['returns', 'risk', 'distribution', 'correlation']
            
        # 如果未指定参数，使用默认参数
        if params is None:
            params = {}
            
        # 分析结果字典
        results = {}
        
        # 执行指定的分析方法
        for method in methods:
            if method == 'returns':
                results['returns'] = self.analyze_returns(params.get('returns_params', {}))
            elif method == 'risk':
                results['risk'] = self.analyze_risk(params.get('risk_params', {}))
            elif method == 'distribution':
                results['distribution'] = self.analyze_distribution(params.get('distribution_params', {}))
            elif method == 'correlation':
                results['correlation'] = self.analyze_correlation(params.get('correlation_params', {}))
            else:
                logger.warning(f"未知的分析方法: {method}")
                
        return results
        
    def analyze_returns(self, params=None):
        """
        分析收益率
        
        Args:
            params (dict, optional): 参数字典. Defaults to None.
            
        Returns:
            dict: 收益率分析结果
        """
        if params is None:
            params = {}
            
        # 确保数据包含收盘价
        if 'close' not in self.data.columns:
            logger.error("数据中缺少'close'列")
            return {}
            
        # 计算每日收益率
        self.data['daily_return'] = self.data['close'].pct_change()
        
        # 计算累计收益率
        first_close = self.data['close'].iloc[0]
        self.data['cumulative_return'] = self.data['close'] / first_close - 1
        
        # 计算对数收益率
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # 分析结果
        result = {
            'total_return': self.data['cumulative_return'].iloc[-1],
            'annualized_return': self.calculate_annualized_return(self.data),
            'daily_return_mean': self.data['daily_return'].mean(),
            'daily_return_std': self.data['daily_return'].std(),
            'daily_return_min': self.data['daily_return'].min(),
            'daily_return_max': self.data['daily_return'].max(),
            'positive_days': (self.data['daily_return'] > 0).sum(),
            'negative_days': (self.data['daily_return'] < 0).sum(),
            'positive_days_ratio': (self.data['daily_return'] > 0).mean()
        }
        
        return result
        
    def analyze_risk(self, params=None):
        """
        分析风险指标
        
        Args:
            params (dict, optional): 参数字典. Defaults to None.
            
        Returns:
            dict: 风险分析结果
        """
        if params is None:
            params = {}
            
        # 确保已计算每日收益率
        if 'daily_return' not in self.data.columns:
            self.analyze_returns()
            
        # 获取参数
        risk_free_rate = params.get('risk_free_rate', 0.02 / 252)  # 默认年化2%转为日化
        max_drawdown_window = params.get('max_drawdown_window', None)
        
        # 计算波动率（年化）
        volatility = self.data['daily_return'].std() * np.sqrt(252)
        
        # 计算最大回撤
        max_drawdown, drawdown_start, drawdown_end = self.calculate_max_drawdown(
            self.data['close'], window=max_drawdown_window
        )
        
        # 计算夏普比率
        excess_returns = self.data['daily_return'] - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # 计算索提诺比率
        downside_std = self.data['daily_return'][self.data['daily_return'] < 0].std()
        sortino_ratio = np.sqrt(252) * (self.data['daily_return'].mean() - risk_free_rate) / downside_std if downside_std > 0 else np.nan
        
        # 计算VaR (Value at Risk)
        var_95 = np.percentile(self.data['daily_return'].dropna(), 5)
        var_99 = np.percentile(self.data['daily_return'].dropna(), 1)
        
        # 分析结果
        result = {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_start': drawdown_start,
            'max_drawdown_end': drawdown_end,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'risk_return_ratio': abs(volatility / self.data['daily_return'].mean() if self.data['daily_return'].mean() != 0 else np.nan)
        }
        
        return result
        
    def analyze_distribution(self, params=None):
        """
        分析收益率分布
        
        Args:
            params (dict, optional): 参数字典. Defaults to None.
            
        Returns:
            dict: 分布分析结果
        """
        if params is None:
            params = {}
            
        # 确保已计算每日收益率
        if 'daily_return' not in self.data.columns:
            self.analyze_returns()
            
        # 去除缺失值
        returns = self.data['daily_return'].dropna()
        
        # 计算描述性统计
        desc_stats = returns.describe()
        
        # 计算偏度和峰度
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # 进行正态性检验（Jarque-Bera测试）
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # 计算分位数
        quantiles = {f'q{q}': np.percentile(returns, q) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
        
        # 分析结果
        result = {
            'mean': desc_stats['mean'],
            'std': desc_stats['std'],
            'min': desc_stats['min'],
            'max': desc_stats['max'],
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05,  # p值大于0.05表示不能拒绝正态分布假设
            'quantiles': quantiles
        }
        
        return result
        
    def analyze_correlation(self, params=None):
        """
        分析相关性（如果数据中有多支股票或指数）
        
        Args:
            params (dict, optional): 参数字典. Defaults to None.
            
        Returns:
            pandas.DataFrame or None: 相关性分析结果，如果无法计算则返回None
        """
        if params is None:
            params = {}
            
        # 确保已计算每日收益率
        if 'daily_return' not in self.data.columns:
            self.analyze_returns()
            
        # 如果数据中只有一支股票，则无法计算相关性
        if 'symbol' not in self.data.columns or self.data['symbol'].nunique() <= 1:
            logger.warning("数据中只有一支股票，无法计算相关性")
            return None
            
        # 重塑数据以便计算相关性
        returns_pivot = self.data.pivot(columns='symbol', values='daily_return')
        
        # 计算相关性
        correlation = returns_pivot.corr()
        
        return correlation
        
    def visualize(self, result=None, plot_type='returns', **kwargs):
        """
        可视化统计分析结果
        
        Args:
            result (dict, optional): 分析结果字典. Defaults to None.
            plot_type (str, optional): 图表类型. Defaults to 'returns'.
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if result is None:
            result = self.analyze()
            
        # 根据图表类型选择不同的可视化方法
        if plot_type == 'returns':
            return self.plot_returns_analysis(result.get('returns', {}), **kwargs)
        elif plot_type == 'risk':
            return self.plot_risk_analysis(result.get('risk', {}), **kwargs)
        elif plot_type == 'distribution':
            return self.plot_distribution_analysis(result.get('distribution', {}), **kwargs)
        elif plot_type == 'correlation':
            return self.plot_correlation_analysis(result.get('correlation', None), **kwargs)
        else:
            logger.warning(f"未知的图表类型: {plot_type}")
            return None
            
    def plot_returns_analysis(self, returns_result, **kwargs):
        """
        绘制收益率分析图表
        
        Args:
            returns_result (dict): 收益率分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not returns_result:
            logger.error("没有收益率分析结果可供可视化")
            return None
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 绘制累计收益曲线
        ax = axes[0, 0]
        ax.plot(self.data.index, self.data['cumulative_return'])
        ax.set_title('累计收益曲线')
        ax.set_ylabel('累计收益率')
        ax.grid(True)
        
        # 绘制每日收益率分布直方图
        ax = axes[0, 1]
        ax.hist(self.data['daily_return'].dropna(), bins=50, alpha=0.7)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title('每日收益率分布')
        ax.set_xlabel('每日收益率')
        ax.set_ylabel('频率')
        ax.grid(True)
        
        # 绘制收益率滚动均值
        ax = axes[1, 0]
        window = kwargs.get('rolling_window', 20)
        self.data['rolling_return'] = self.data['daily_return'].rolling(window=window).mean()
        ax.plot(self.data.index, self.data['rolling_return'])
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title(f'{window}日滚动平均收益率')
        ax.set_ylabel('滚动平均收益率')
        ax.grid(True)
        
        # 绘制收益率统计指标表格
        ax = axes[1, 1]
        stats_text = [
            f"总收益率: {returns_result.get('total_return', 0)*100:.2f}%",
            f"年化收益率: {returns_result.get('annualized_return', 0)*100:.2f}%",
            f"平均每日收益率: {returns_result.get('daily_return_mean', 0)*100:.4f}%",
            f"收益率标准差: {returns_result.get('daily_return_std', 0)*100:.4f}%",
            f"最大单日收益: {returns_result.get('daily_return_max', 0)*100:.2f}%",
            f"最大单日回撤: {returns_result.get('daily_return_min', 0)*100:.2f}%",
            f"上涨天数: {returns_result.get('positive_days', 0)}",
            f"下跌天数: {returns_result.get('negative_days', 0)}",
            f"上涨占比: {returns_result.get('positive_days_ratio', 0)*100:.2f}%"
        ]
        ax.axis('off')
        y_pos = 0.9
        for text in stats_text:
            ax.text(0.1, y_pos, text, fontsize=12)
            y_pos -= 0.1
        ax.set_title('收益率统计指标')
        
        plt.tight_layout()
        return fig
            
    def plot_risk_analysis(self, risk_result, **kwargs):
        """
        绘制风险分析图表
        
        Args:
            risk_result (dict): 风险分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not risk_result:
            logger.error("没有风险分析结果可供可视化")
            return None
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 绘制回撤曲线
        ax = axes[0, 0]
        drawdown_series = self.calculate_drawdown_series(self.data['close'])
        ax.plot(self.data.index, drawdown_series)
        ax.set_title('回撤曲线')
        ax.set_ylabel('回撤')
        ax.grid(True)
        
        # 绘制滚动波动率
        ax = axes[0, 1]
        window = kwargs.get('rolling_window', 20)
        rolling_vol = self.data['daily_return'].rolling(window=window).std() * np.sqrt(252)
        ax.plot(self.data.index, rolling_vol)
        ax.set_title(f'{window}日滚动波动率')
        ax.set_ylabel('年化波动率')
        ax.grid(True)
        
        # 绘制风险收益散点图
        ax = axes[1, 0]
        if self.data['symbol'].nunique() > 1:
            # 如果有多支股票，绘制股票组合的风险收益散点图
            symbols = self.data['symbol'].unique()
            returns = []
            risks = []
            
            for symbol in symbols:
                stock_data = self.data[self.data['symbol'] == symbol]
                stock_returns = stock_data['daily_return']
                returns.append(stock_returns.mean() * 252)  # 年化收益率
                risks.append(stock_returns.std() * np.sqrt(252))  # 年化风险
                
            ax.scatter(risks, returns)
            
            # 添加标签
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (risks[i], returns[i]))
                
            ax.set_title('风险-收益散点图')
            ax.set_xlabel('风险（年化波动率）')
            ax.set_ylabel('收益（年化收益率）')
        else:
            # 如果只有一支股票，绘制空白图表并添加文本
            ax.text(0.5, 0.5, '需要多支股票数据才能生成风险-收益散点图', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        ax.grid(True)
        
        # 绘制风险指标表格
        ax = axes[1, 1]
        risk_text = [
            f"波动率: {risk_result.get('volatility', 0)*100:.2f}%",
            f"最大回撤: {risk_result.get('max_drawdown', 0)*100:.2f}%",
            f"夏普比率: {risk_result.get('sharpe_ratio', 0):.2f}",
            f"索提诺比率: {risk_result.get('sortino_ratio', 0):.2f}",
            f"VaR(95%): {risk_result.get('var_95', 0)*100:.2f}%",
            f"VaR(99%): {risk_result.get('var_99', 0)*100:.2f}%",
            f"风险收益比: {risk_result.get('risk_return_ratio', 0):.2f}"
        ]
        ax.axis('off')
        y_pos = 0.9
        for text in risk_text:
            ax.text(0.1, y_pos, text, fontsize=12)
            y_pos -= 0.1
        ax.set_title('风险指标')
        
        plt.tight_layout()
        return fig
            
    def plot_distribution_analysis(self, distribution_result, **kwargs):
        """
        绘制分布分析图表
        
        Args:
            distribution_result (dict): 分布分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not distribution_result:
            logger.error("没有分布分析结果可供可视化")
            return None
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 获取收益率数据
        returns = self.data['daily_return'].dropna()
        
        # 绘制收益率分布直方图和正态分布拟合
        ax = axes[0, 0]
        sns.histplot(returns, kde=True, ax=ax)
        
        # 添加正态分布拟合曲线
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std())
        ax.plot(x, y, 'r--')
        
        ax.set_title('收益率分布直方图与正态分布拟合')
        ax.set_xlabel('收益率')
        ax.set_ylabel('频率')
        ax.grid(True)
        
        # 绘制Q-Q图
        ax = axes[0, 1]
        stats.probplot(returns, plot=ax)
        ax.set_title('收益率Q-Q图')
        ax.grid(True)
        
        # 绘制收益率时间序列自相关图
        ax = axes[1, 0]
        pd.plotting.autocorrelation_plot(returns, ax=ax)
        ax.set_title('收益率自相关图')
        ax.grid(True)
        
        # 绘制分布统计指标表格
        ax = axes[1, 1]
        distribution_text = [
            f"平均收益率: {distribution_result.get('mean', 0)*100:.4f}%",
            f"标准差: {distribution_result.get('std', 0)*100:.4f}%",
            f"偏度: {distribution_result.get('skewness', 0):.4f}",
            f"峰度: {distribution_result.get('kurtosis', 0):.4f}",
            f"Jarque-Bera统计量: {distribution_result.get('jarque_bera_stat', 0):.4f}",
            f"Jarque-Bera p值: {distribution_result.get('jarque_bera_pvalue', 0):.4f}",
            f"是否服从正态分布: {'是' if distribution_result.get('is_normal', False) else '否'}",
            f"1%分位数: {distribution_result.get('quantiles', {}).get('q1', 0)*100:.2f}%",
            f"5%分位数: {distribution_result.get('quantiles', {}).get('q5', 0)*100:.2f}%",
            f"95%分位数: {distribution_result.get('quantiles', {}).get('q95', 0)*100:.2f}%",
            f"99%分位数: {distribution_result.get('quantiles', {}).get('q99', 0)*100:.2f}%"
        ]
        ax.axis('off')
        y_pos = 0.9
        for text in distribution_text:
            ax.text(0.1, y_pos, text, fontsize=12)
            y_pos -= 0.08
        ax.set_title('分布统计指标')
        
        plt.tight_layout()
        return fig
        
    def plot_correlation_analysis(self, correlation_result, **kwargs):
        """
        绘制相关性分析图表
        
        Args:
            correlation_result (pandas.DataFrame): 相关性分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if correlation_result is None or correlation_result.empty:
            logger.error("没有相关性分析结果可供可视化")
            return None
            
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(correlation_result, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('股票收益率相关性热力图')
        
        plt.tight_layout()
        return fig
    
    def calculate_annualized_return(self, data):
        """
        计算年化收益率
        
        Args:
            data (pandas.DataFrame): 股票数据
            
        Returns:
            float: 年化收益率
        """
        # 确保数据至少有两个点
        if len(data) < 2:
            return 0
            
        # 获取总收益率
        total_return = data['cumulative_return'].iloc[-1]
        
        # 计算交易天数
        trading_days = len(data)
        
        # 计算年化收益率
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return annualized_return
        
    def calculate_max_drawdown(self, prices, window=None):
        """
        计算最大回撤及其开始和结束点
        
        Args:
            prices (pandas.Series): 价格序列
            window (int, optional): 窗口大小. Defaults to None.
            
        Returns:
            tuple: (最大回撤, 开始点, 结束点)
        """
        # 价格序列的累计最大值
        roll_max = prices.expanding().max()
        
        # 计算回撤
        drawdown = prices / roll_max - 1
        
        # 找到最大回撤的结束点
        drawdown_end = drawdown.idxmin()
        
        # 找到最大回撤的开始点
        if window:
            # 如果指定了窗口，只在窗口内寻找
            drawdown_start = prices[max(0, prices.index.get_loc(drawdown_end) - window):prices.index.get_loc(drawdown_end)].idxmax()
        else:
            # 否则在整个序列中寻找
            drawdown_start = prices[:drawdown_end].idxmax()
        
        # 计算最大回撤值
        max_drawdown = drawdown.min()
        
        return max_drawdown, drawdown_start, drawdown_end
        
    def calculate_drawdown_series(self, prices):
        """
        计算回撤序列
        
        Args:
            prices (pandas.Series): 价格序列
            
        Returns:
            pandas.Series: 回撤序列
        """
        # 价格序列的累计最大值
        roll_max = prices.expanding().max()
        
        # 计算回撤序列
        drawdown = prices / roll_max - 1
        
        return drawdown
