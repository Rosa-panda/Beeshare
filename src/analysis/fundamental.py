"""
基本面分析模块

提供股票基本面分析功能，包括财务指标分析、估值分析、成长性分析等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import Analyzer
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer(Analyzer):
    """基本面分析器，提供股票基本面分析功能"""
    
    def __init__(self, data=None, stock_info=None):
        """
        初始化基本面分析器
        
        Args:
            data (pandas.DataFrame, optional): 价格等市场数据. Defaults to None.
            stock_info (pandas.DataFrame or dict, optional): 股票基本信息. Defaults to None.
        """
        super().__init__(data)
        self.stock_info = stock_info
        self.financial_data = None
        self.industry_data = None
        
    def set_stock_info(self, stock_info):
        """
        设置股票基本信息
        
        Args:
            stock_info (pandas.DataFrame or dict): 股票基本信息
        """
        self.stock_info = stock_info
        
    def set_financial_data(self, financial_data):
        """
        设置财务数据
        
        Args:
            financial_data (pandas.DataFrame): 财务数据
        """
        self.financial_data = financial_data
        
    def set_industry_data(self, industry_data):
        """
        设置行业数据
        
        Args:
            industry_data (pandas.DataFrame): 行业数据
        """
        self.industry_data = industry_data
        
    def analyze(self, methods=None, params=None):
        """
        执行基本面分析
        
        Args:
            methods (list, optional): 要执行的分析方法列表. Defaults to None.
            params (dict, optional): 分析参数字典. Defaults to None.
            
        Returns:
            dict: 包含分析结果的字典
        """
        if self.data is None or self.data.empty:
            logger.error("没有市场数据可供分析")
            return {}
            
        # 如果未指定方法，默认使用所有分析方法
        if methods is None:
            methods = ['valuation', 'growth', 'profitability', 'solvency', 'dividend']
            
        # 如果未指定参数，使用默认参数
        if params is None:
            params = {}
            
        # 分析结果字典
        results = {}
        
        # 执行指定的分析方法
        for method in methods:
            if method == 'valuation':
                results['valuation'] = self.analyze_valuation(params.get('valuation_params', {}))
            elif method == 'growth':
                results['growth'] = self.analyze_growth(params.get('growth_params', {}))
            elif method == 'profitability':
                results['profitability'] = self.analyze_profitability(params.get('profitability_params', {}))
            elif method == 'solvency':
                results['solvency'] = self.analyze_solvency(params.get('solvency_params', {}))
            elif method == 'dividend':
                results['dividend'] = self.analyze_dividend(params.get('dividend_params', {}))
            else:
                logger.warning(f"未知的分析方法: {method}")
                
        return results
        
    def analyze_valuation(self, params=None):
        """
        分析估值指标
        
        Args:
            params (dict, optional): 估值分析参数. Defaults to None.
            
        Returns:
            dict: 估值分析结果
        """
        if params is None:
            params = {}
            
        if self.financial_data is None:
            logger.error("缺少财务数据，无法进行估值分析")
            return {}
            
        # 基本估值指标
        result = {}
        
        # 如果财务数据是字典，直接获取相关指标
        if isinstance(self.financial_data, dict):
            result['pe_ttm'] = self.financial_data.get('pe_ttm', None)  # 市盈率(TTM)
            result['pb'] = self.financial_data.get('pb', None)  # 市净率
            result['ps_ttm'] = self.financial_data.get('ps_ttm', None)  # 市销率(TTM)
            result['pcf_ttm'] = self.financial_data.get('pcf_ttm', None)  # 市现率(TTM)
            result['ev_ebitda'] = self.financial_data.get('ev_ebitda', None)  # 企业价值倍数
            
        # 如果财务数据是DataFrame，从中提取最新的估值指标
        elif isinstance(self.financial_data, pd.DataFrame):
            if not self.financial_data.empty:
                # 获取估值相关列
                valuation_columns = ['pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm', 'ev_ebitda', 
                                    'pe', 'pe_forecast', 'pb', 'ps', 'pcf']
                
                # 过滤出存在的列
                existing_columns = [col for col in valuation_columns if col in self.financial_data.columns]
                
                if existing_columns:
                    # 获取最新数据
                    latest_data = self.financial_data.iloc[-1][existing_columns]
                    
                    # 将存在的指标添加到结果中
                    for col in existing_columns:
                        result[col] = latest_data[col]
                
        # 行业比较（如果有行业数据）
        if self.industry_data is not None and isinstance(self.industry_data, pd.DataFrame):
            # 假设行业数据中包含行业平均估值指标
            industry_valuation = self.industry_data.mean() if not self.industry_data.empty else {}
            
            result['industry_pe_ttm'] = industry_valuation.get('pe_ttm', None)
            result['industry_pb'] = industry_valuation.get('pb', None)
            result['industry_ps_ttm'] = industry_valuation.get('ps_ttm', None)
            
            # 计算相对于行业的溢价/折价
            for metric in ['pe_ttm', 'pb', 'ps_ttm']:
                if result.get(metric) is not None and result.get(f'industry_{metric}') is not None:
                    result[f'{metric}_premium'] = result[metric] / result[f'industry_{metric}'] - 1
        
        return result
        
    def analyze_growth(self, params=None):
        """
        分析成长性指标
        
        Args:
            params (dict, optional): 成长性分析参数. Defaults to None.
            
        Returns:
            dict: 成长性分析结果
        """
        if params is None:
            params = {}
            
        if self.financial_data is None:
            logger.error("缺少财务数据，无法进行成长性分析")
            return {}
            
        # 成长性指标
        result = {}
        
        # 如果财务数据是字典，直接获取相关指标
        if isinstance(self.financial_data, dict):
            result['revenue_growth'] = self.financial_data.get('revenue_growth', None)  # 营收增长率
            result['profit_growth'] = self.financial_data.get('profit_growth', None)  # 净利润增长率
            result['eps_growth'] = self.financial_data.get('eps_growth', None)  # 每股收益增长率
            
        # 如果财务数据是DataFrame，需要计算增长率
        elif isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            growth_metrics = {
                'revenue': '营业收入',
                'net_profit': '净利润',
                'eps': '每股收益'
            }
            
            for eng_key, cn_key in growth_metrics.items():
                # 检查是否存在中文或英文列名
                col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                
                if col_name in self.financial_data.columns:
                    # 获取近几年的数据并计算同比增长率
                    values = self.financial_data[col_name].values
                    if len(values) >= 2:
                        # 计算同比增长率
                        growth_rates = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
                        result[f'{eng_key}_yoy'] = growth_rates[-1]  # 最新同比增长率
                        result[f'{eng_key}_cagr'] = (values[-1] / values[0]) ** (1 / (len(values) - 1)) - 1  # 复合年均增长率
            
        # 如果有行业数据，计算相对于行业的成长性对比
        if self.industry_data is not None and isinstance(self.industry_data, pd.DataFrame) and not self.industry_data.empty:
            for metric in ['revenue_growth', 'profit_growth', 'eps_growth']:
                if metric in self.industry_data.columns and result.get(metric) is not None:
                    result[f'industry_{metric}'] = self.industry_data[metric].mean()
                    result[f'{metric}_vs_industry'] = result[metric] - result[f'industry_{metric}']
        
        return result
        
    def analyze_profitability(self, params=None):
        """
        分析盈利能力指标
        
        Args:
            params (dict, optional): 盈利能力分析参数. Defaults to None.
            
        Returns:
            dict: 盈利能力分析结果
        """
        if params is None:
            params = {}
            
        if self.financial_data is None:
            logger.error("缺少财务数据，无法进行盈利能力分析")
            return {}
            
        # 盈利能力指标
        result = {}
        
        # 关键盈利指标名称映射（英文名到中文名）
        metric_names = {
            'gross_margin': '毛利率',
            'operating_margin': '营业利润率',
            'net_margin': '净利率',
            'roe': '净资产收益率',
            'roa': '总资产收益率',
            'roic': '投入资本回报率'
        }
        
        # 如果财务数据是字典，直接获取相关指标
        if isinstance(self.financial_data, dict):
            for eng_key in metric_names.keys():
                result[eng_key] = self.financial_data.get(eng_key, None)
            
        # 如果财务数据是DataFrame，从中提取最新的盈利能力指标
        elif isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            # 尝试使用英文列名或中文列名获取数据
            for eng_key, cn_key in metric_names.items():
                col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                
                if col_name in self.financial_data.columns:
                    result[eng_key] = self.financial_data[col_name].iloc[-1]  # 获取最新值
            
        # 行业比较（如果有行业数据）
        if self.industry_data is not None and isinstance(self.industry_data, pd.DataFrame) and not self.industry_data.empty:
            for eng_key in metric_names.keys():
                if eng_key in self.industry_data.columns and result.get(eng_key) is not None:
                    result[f'industry_{eng_key}'] = self.industry_data[eng_key].mean()
                    result[f'{eng_key}_vs_industry'] = result[eng_key] - result[f'industry_{eng_key}']
        
        return result
        
    def analyze_solvency(self, params=None):
        """
        分析偿债能力和财务状况指标
        
        Args:
            params (dict, optional): 偿债能力分析参数. Defaults to None.
            
        Returns:
            dict: 偿债能力分析结果
        """
        if params is None:
            params = {}
            
        if self.financial_data is None:
            logger.error("缺少财务数据，无法进行偿债能力分析")
            return {}
            
        # 偿债能力指标
        result = {}
        
        # 关键偿债能力指标名称映射（英文名到中文名）
        metric_names = {
            'current_ratio': '流动比率',
            'quick_ratio': '速动比率',
            'debt_to_asset': '资产负债率',
            'debt_to_equity': '权益乘数',
            'interest_coverage': '利息保障倍数'
        }
        
        # 如果财务数据是字典，直接获取相关指标
        if isinstance(self.financial_data, dict):
            for eng_key in metric_names.keys():
                result[eng_key] = self.financial_data.get(eng_key, None)
            
        # 如果财务数据是DataFrame，从中提取最新的偿债能力指标
        elif isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            # 尝试使用英文列名或中文列名获取数据
            for eng_key, cn_key in metric_names.items():
                col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                
                if col_name in self.financial_data.columns:
                    result[eng_key] = self.financial_data[col_name].iloc[-1]  # 获取最新值
            
        # 行业比较（如果有行业数据）
        if self.industry_data is not None and isinstance(self.industry_data, pd.DataFrame) and not self.industry_data.empty:
            for eng_key in metric_names.keys():
                if eng_key in self.industry_data.columns and result.get(eng_key) is not None:
                    result[f'industry_{eng_key}'] = self.industry_data[eng_key].mean()
                    result[f'{eng_key}_vs_industry'] = result[eng_key] - result[f'industry_{eng_key}']
        
        return result
        
    def analyze_dividend(self, params=None):
        """
        分析股息和分红指标
        
        Args:
            params (dict, optional): 股息分析参数. Defaults to None.
            
        Returns:
            dict: 股息分析结果
        """
        if params is None:
            params = {}
            
        if self.financial_data is None:
            logger.error("缺少财务数据，无法进行股息分析")
            return {}
            
        # 股息指标
        result = {}
        
        # 关键股息指标名称映射（英文名到中文名）
        metric_names = {
            'dividend_yield': '股息率',
            'dividend_payout_ratio': '股息支付率',
            'dividend_per_share': '每股股息',
            'dividend_growth': '股息增长率'
        }
        
        # 如果财务数据是字典，直接获取相关指标
        if isinstance(self.financial_data, dict):
            for eng_key in metric_names.keys():
                result[eng_key] = self.financial_data.get(eng_key, None)
            
        # 如果财务数据是DataFrame，从中提取最新的股息指标
        elif isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            # 尝试使用英文列名或中文列名获取数据
            for eng_key, cn_key in metric_names.items():
                col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                
                if col_name in self.financial_data.columns:
                    result[eng_key] = self.financial_data[col_name].iloc[-1]  # 获取最新值
                    
            # 如果有股息历史数据，计算股息增长性
            if 'dividend_per_share' in self.financial_data.columns or '每股股息' in self.financial_data.columns:
                col_name = 'dividend_per_share' if 'dividend_per_share' in self.financial_data.columns else '每股股息'
                values = self.financial_data[col_name].values
                
                if len(values) >= 2 and values[0] > 0:
                    # 计算股息年均复合增长率
                    result['dividend_cagr'] = (values[-1] / values[0]) ** (1 / (len(values) - 1)) - 1
            
        # 行业比较（如果有行业数据）
        if self.industry_data is not None and isinstance(self.industry_data, pd.DataFrame) and not self.industry_data.empty:
            for eng_key in metric_names.keys():
                if eng_key in self.industry_data.columns and result.get(eng_key) is not None:
                    result[f'industry_{eng_key}'] = self.industry_data[eng_key].mean()
                    result[f'{eng_key}_vs_industry'] = result[eng_key] - result[f'industry_{eng_key}']
        
        return result
        
    def visualize(self, result=None, plot_type='overview', **kwargs):
        """
        可视化基本面分析结果
        
        Args:
            result (dict, optional): 分析结果字典. Defaults to None.
            plot_type (str, optional): 图表类型. Defaults to 'overview'.
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if result is None:
            result = self.analyze()
            
        # 根据图表类型选择不同的可视化方法
        if plot_type == 'overview':
            return self.plot_fundamental_overview(result, **kwargs)
        elif plot_type == 'valuation':
            return self.plot_valuation_analysis(result.get('valuation', {}), **kwargs)
        elif plot_type == 'growth':
            return self.plot_growth_analysis(result.get('growth', {}), **kwargs)
        elif plot_type == 'profitability':
            return self.plot_profitability_analysis(result.get('profitability', {}), **kwargs)
        elif plot_type == 'solvency':
            return self.plot_solvency_analysis(result.get('solvency', {}), **kwargs)
        elif plot_type == 'dividend':
            return self.plot_dividend_analysis(result.get('dividend', {}), **kwargs)
        elif plot_type == 'industry_comparison':
            return self.plot_industry_comparison(result, **kwargs)
        else:
            logger.warning(f"未知的图表类型: {plot_type}")
            return None
            
    def plot_fundamental_overview(self, result, **kwargs):
        """
        绘制基本面分析概览图表
        
        Args:
            result (dict): 基本面分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 绘制估值指标
        ax = axes[0, 0]
        valuation_data = result.get('valuation', {})
        if valuation_data:
            metrics = ['pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm']
            labels = ['市盈率(TTM)', '市净率', '市销率(TTM)', '市现率(TTM)']
            values = [valuation_data.get(metric, 0) for metric in metrics]
            industry_values = [valuation_data.get(f'industry_{metric}', 0) for metric in metrics]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, values, width, label='个股')
            ax.bar(x + width/2, industry_values, width, label='行业平均')
            
            ax.set_ylabel('倍数')
            ax.set_title('估值指标对比')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
        else:
            ax.text(0.5, 0.5, '缺少估值数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 绘制成长性指标
        ax = axes[0, 1]
        growth_data = result.get('growth', {})
        if growth_data:
            metrics = ['revenue_yoy', 'net_profit_yoy', 'eps_yoy']
            labels = ['营收增长率', '净利润增长率', 'EPS增长率']
            values = [growth_data.get(metric, 0) * 100 for metric in metrics]
            
            ax.bar(labels, values, color=['#1f77b4' if v >= 0 else '#d62728' for v in values])
            
            ax.set_ylabel('增长率(%)')
            ax.set_title('成长性指标')
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + (5 if v >= 0 else -5), f'{v:.1f}%', ha='center')
                
            # 设置y轴范围，确保零线可见
            y_min = min(min(values) - 10, -5)
            y_max = max(max(values) + 10, 5)
            ax.set_ylim(y_min, y_max)
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '缺少成长性数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 绘制盈利能力指标
        ax = axes[1, 0]
        profitability_data = result.get('profitability', {})
        if profitability_data:
            metrics = ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa']
            labels = ['毛利率', '营业利润率', '净利率', 'ROE', 'ROA']
            values = [profitability_data.get(metric, 0) * 100 for metric in metrics]
            
            ax.bar(labels, values)
            
            ax.set_ylabel('比率(%)')
            ax.set_title('盈利能力指标')
            
            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少盈利能力数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 绘制财务健康度指标
        ax = axes[1, 1]
        solvency_data = result.get('solvency', {})
        if solvency_data:
            metrics = ['current_ratio', 'quick_ratio', 'debt_to_asset']
            labels = ['流动比率', '速动比率', '资产负债率']
            values = [solvency_data.get(metric, 0) for metric in metrics]
            
            # 对于资产负债率，需要转换为百分比
            if len(values) >= 3:
                values[2] *= 100
                
            colors = ['#1f77b4', '#1f77b4', '#d62728' if values[2] > 60 else '#2ca02c']
            
            ax.bar(labels, values, color=colors)
            
            # 对于不同类型的指标使用不同的y轴标签
            ax.set_ylabel('比率 / 百分比')
            ax.set_title('财务健康度指标')
            
            # 添加数值标签
            for i, v in enumerate(values):
                if i < 2:  # 比率
                    ax.text(i, v + 0.1, f'{v:.2f}', ha='center')
                else:  # 百分比
                    ax.text(i, v + 3, f'{v:.1f}%', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少财务健康度数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 基本面分析概览"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig

    def plot_valuation_analysis(self, valuation_result, **kwargs):
        """
        绘制估值分析图表
        
        Args:
            valuation_result (dict): 估值分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not valuation_result:
            logger.error("没有估值分析结果可供可视化")
            return None
            
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 估值指标与行业对比柱状图
        ax = axes[0, 0]
        metrics = ['pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm']
        labels = ['市盈率(TTM)', '市净率', '市销率(TTM)', '市现率(TTM)']
        values = []
        industry_values = []
        
        for metric in metrics:
            val = valuation_result.get(metric)
            ind_val = valuation_result.get(f'industry_{metric}')
            
            # 如果个股值存在但行业值不存在，设置行业值为None
            if val is not None and ind_val is None:
                values.append(val)
                industry_values.append(None)
            # 如果行业值存在但个股值不存在，设置个股值为None
            elif val is None and ind_val is not None:
                values.append(None)
                industry_values.append(ind_val)
            # 如果都存在或都不存在
            else:
                values.append(val)
                industry_values.append(ind_val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(values) if v is not None or industry_values[i] is not None]
        if valid_indices:
            valid_labels = [labels[i] for i in valid_indices]
            valid_values = [values[i] if values[i] is not None else 0 for i in valid_indices]
            valid_industry = [industry_values[i] if industry_values[i] is not None else 0 for i in valid_indices]
            
            x = np.arange(len(valid_labels))
            width = 0.35
            
            ax.bar(x - width/2, valid_values, width, label='个股')
            ax.bar(x + width/2, valid_industry, width, label='行业平均')
            
            ax.set_ylabel('倍数')
            ax.set_title('估值指标与行业对比')
            ax.set_xticks(x)
            ax.set_xticklabels(valid_labels)
            ax.legend()
            
            # 添加数值标签
            for i, v in enumerate(valid_values):
                if v is not None:
                    ax.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center')
            
            for i, v in enumerate(valid_industry):
                if v is not None:
                    ax.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少估值与行业对比数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 2. 估值溢价/折价图
        ax = axes[0, 1]
        premium_metrics = [m+'_premium' for m in metrics]
        premium_labels = [l.replace('(TTM)', '')+'溢价率' for l in labels]
        premium_values = []
        
        for metric in premium_metrics:
            val = valuation_result.get(metric)
            premium_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(premium_values) if v is not None]
        if valid_indices:
            valid_premium_labels = [premium_labels[i] for i in valid_indices]
            valid_premium_values = [premium_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            colors = ['#d62728' if v > 0 else '#2ca02c' for v in valid_premium_values]
            ax.bar(valid_premium_labels, valid_premium_values, color=colors)
            
            ax.set_ylabel('溢价/折价(%)')
            ax.set_title('相对行业估值溢价/折价')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(valid_premium_values):
                if v >= 0:
                    ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                else:
                    ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                    
            # 设置y轴范围，确保零线可见
            y_min = min(min(valid_premium_values) - 10, -5)
            y_max = max(max(valid_premium_values) + 10, 5)
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, '缺少估值溢价/折价数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 3. 历史估值走势图（如果有）
        ax = axes[1, 0]
        if isinstance(self.financial_data, pd.DataFrame) and 'date' in self.financial_data.columns:
            for metric in ['pe_ttm', 'pb']:
                if metric in self.financial_data.columns:
                    ax.plot(self.financial_data['date'], self.financial_data[metric], 
                           label=metric.replace('pe_ttm', '市盈率').replace('pb', '市净率'))
            
            ax.set_title('历史估值走势')
            ax.set_xlabel('日期')
            ax.set_ylabel('倍数')
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, '缺少历史估值数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 4. 估值文字描述
        ax = axes[1, 1]
        ax.axis('off')
        
        # 估值指标表格
        valuation_text = []
        
        # 添加主要估值指标
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            value = valuation_result.get(metric)
            if value is not None:
                valuation_text.append(f"{label}: {value:.2f}")
        
        # 添加更多估值指标（如果有）
        more_metrics = ['ev_ebitda', 'pe_forecast']
        more_labels = ['企业价值倍数', '预期市盈率']
        
        for metric, label in zip(more_metrics, more_labels):
            value = valuation_result.get(metric)
            if value is not None:
                valuation_text.append(f"{label}: {value:.2f}")
        
        # 添加估值百分位（如果有）
        percentile_metrics = ['pe_percentile', 'pb_percentile']
        percentile_labels = ['市盈率百分位', '市净率百分位']
        
        for metric, label in zip(percentile_metrics, percentile_labels):
            value = valuation_result.get(metric)
            if value is not None:
                valuation_text.append(f"{label}: {value*100:.1f}%")
        
        y_pos = 0.9
        for text in valuation_text:
            ax.text(0.1, y_pos, text, fontsize=12)
            y_pos -= 0.1
        
        if not valuation_text:
            ax.text(0.5, 0.5, '缺少估值指标数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        else:
            ax.set_title('估值指标详情')
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 估值分析"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
        
    def plot_growth_analysis(self, growth_result, **kwargs):
        """
        绘制成长性分析图表
        
        Args:
            growth_result (dict): 成长性分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not growth_result:
            logger.error("没有成长性分析结果可供可视化")
            return None
            
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 同比增长率柱状图
        ax = axes[0, 0]
        yoy_metrics = ['revenue_yoy', 'net_profit_yoy', 'eps_yoy']
        yoy_labels = ['营收同比增长', '利润同比增长', 'EPS同比增长']
        yoy_values = []
        
        for metric in yoy_metrics:
            val = growth_result.get(metric)
            yoy_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(yoy_values) if v is not None]
        if valid_indices:
            valid_yoy_labels = [yoy_labels[i] for i in valid_indices]
            valid_yoy_values = [yoy_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in valid_yoy_values]
            ax.bar(valid_yoy_labels, valid_yoy_values, color=colors)
            
            ax.set_ylabel('增长率(%)')
            ax.set_title('同比增长率')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(valid_yoy_values):
                if v >= 0:
                    ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                else:
                    ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                    
            # 设置y轴范围，确保零线可见
            y_min = min(min(valid_yoy_values) - 10, -5)
            y_max = max(max(valid_yoy_values) + 10, 5)
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, '缺少同比增长率数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 2. 复合年均增长率(CAGR)柱状图
        ax = axes[0, 1]
        cagr_metrics = ['revenue_cagr', 'net_profit_cagr', 'eps_cagr']
        cagr_labels = ['营收CAGR', '利润CAGR', 'EPS CAGR']
        cagr_values = []
        
        for metric in cagr_metrics:
            val = growth_result.get(metric)
            cagr_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(cagr_values) if v is not None]
        if valid_indices:
            valid_cagr_labels = [cagr_labels[i] for i in valid_indices]
            valid_cagr_values = [cagr_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in valid_cagr_values]
            ax.bar(valid_cagr_labels, valid_cagr_values, color=colors)
            
            ax.set_ylabel('CAGR(%)')
            ax.set_title('复合年均增长率')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(valid_cagr_values):
                if v >= 0:
                    ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                else:
                    ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                    
            # 设置y轴范围，确保零线可见
            y_min = min(min(valid_cagr_values) - 10, -5)
            y_max = max(max(valid_cagr_values) + 10, 5)
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, '缺少复合年均增长率数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 3. 历史增长趋势图（如果有）
        ax = axes[1, 0]
        if isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            if 'date' in self.financial_data.columns:
                date_col = 'date'
            elif self.financial_data.index.name == 'date' or isinstance(self.financial_data.index, pd.DatetimeIndex):
                date_col = self.financial_data.index
            else:
                date_col = None
                
            if date_col is not None:
                growth_trend_metrics = {
                    'revenue': '营业收入',
                    'net_profit': '净利润',
                    'eps': '每股收益'
                }
                
                for eng_key, cn_key in growth_trend_metrics.items():
                    # 检查是否存在中文或英文列名
                    col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                    
                    if col_name in self.financial_data.columns:
                        ax.plot(date_col, self.financial_data[col_name], 
                               label=cn_key, marker='o')
                
                ax.set_title('历史增长趋势')
                ax.set_xlabel('日期')
                ax.set_ylabel('数值')
                ax.grid(True)
                ax.legend()
            else:
                ax.text(0.5, 0.5, '缺少日期数据，无法绘制历史趋势', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '缺少历史增长数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 4. 成长性指标与行业对比
        ax = axes[1, 1]
        vs_industry_metrics = ['revenue_growth_vs_industry', 'profit_growth_vs_industry', 'eps_growth_vs_industry']
        vs_industry_labels = ['营收增长对比', '利润增长对比', 'EPS增长对比']
        vs_industry_values = []
        
        for metric in vs_industry_metrics:
            val = growth_result.get(metric)
            vs_industry_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(vs_industry_values) if v is not None]
        if valid_indices:
            valid_vs_industry_labels = [vs_industry_labels[i] for i in valid_indices]
            valid_vs_industry_values = [vs_industry_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in valid_vs_industry_values]
            ax.bar(valid_vs_industry_labels, valid_vs_industry_values, color=colors)
            
            ax.set_ylabel('高于/低于行业平均(%)')
            ax.set_title('成长性与行业平均对比')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(valid_vs_industry_values):
                if v >= 0:
                    ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                else:
                    ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                    
            # 设置y轴范围，确保零线可见
            y_min = min(min(valid_vs_industry_values) - 10, -5)
            y_max = max(max(valid_vs_industry_values) + 10, 5)
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, '缺少与行业对比数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 成长性分析"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
        
    def plot_profitability_analysis(self, profitability_result, **kwargs):
        """
        绘制盈利能力分析图表
        
        Args:
            profitability_result (dict): 盈利能力分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not profitability_result:
            logger.error("没有盈利能力分析结果可供可视化")
            return None
            
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 利润率指标柱状图
        ax = axes[0, 0]
        margin_metrics = ['gross_margin', 'operating_margin', 'net_margin']
        margin_labels = ['毛利率', '营业利润率', '净利率']
        margin_values = []
        
        for metric in margin_metrics:
            val = profitability_result.get(metric)
            margin_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(margin_values) if v is not None]
        if valid_indices:
            valid_margin_labels = [margin_labels[i] for i in valid_indices]
            valid_margin_values = [margin_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            ax.bar(valid_margin_labels, valid_margin_values)
            
            ax.set_ylabel('百分比(%)')
            ax.set_title('利润率指标')
            
            # 添加数值标签
            for i, v in enumerate(valid_margin_values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少利润率数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 2. 回报率指标柱状图
        ax = axes[0, 1]
        return_metrics = ['roe', 'roa', 'roic']
        return_labels = ['ROE', 'ROA', 'ROIC']
        return_values = []
        
        for metric in return_metrics:
            val = profitability_result.get(metric)
            return_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(return_values) if v is not None]
        if valid_indices:
            valid_return_labels = [return_labels[i] for i in valid_indices]
            valid_return_values = [return_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            ax.bar(valid_return_labels, valid_return_values)
            
            ax.set_ylabel('百分比(%)')
            ax.set_title('回报率指标')
            
            # 添加数值标签
            for i, v in enumerate(valid_return_values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少回报率数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 3. 历史盈利趋势图（如果有）
        ax = axes[1, 0]
        if isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            if 'date' in self.financial_data.columns:
                date_col = 'date'
            elif self.financial_data.index.name == 'date' or isinstance(self.financial_data.index, pd.DatetimeIndex):
                date_col = self.financial_data.index
            else:
                date_col = None
                
            if date_col is not None:
                profit_trend_metrics = {
                    'gross_margin': '毛利率',
                    'operating_margin': '营业利润率',
                    'net_margin': '净利率',
                    'roe': 'ROE'
                }
                
                for eng_key, cn_key in profit_trend_metrics.items():
                    # 检查是否存在中文或英文列名
                    col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                    
                    if col_name in self.financial_data.columns:
                        # 转换为百分比形式
                        ax.plot(date_col, self.financial_data[col_name] * 100, 
                               label=cn_key, marker='o')
                
                ax.set_title('历史盈利能力趋势')
                ax.set_xlabel('日期')
                ax.set_ylabel('百分比(%)')
                ax.grid(True)
                ax.legend()
            else:
                ax.text(0.5, 0.5, '缺少日期数据，无法绘制历史趋势', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '缺少历史盈利能力数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 4. 盈利能力与行业对比
        ax = axes[1, 1]
        vs_industry_all_metrics = margin_metrics + return_metrics
        vs_industry_all_labels = margin_labels + return_labels
        vs_industry_metrics = []
        vs_industry_labels = []
        vs_industry_values = []
        
        # 查找所有有行业对比数据的指标
        for i, metric in enumerate(vs_industry_all_metrics):
            vs_key = f'{metric}_vs_industry'
            if vs_key in profitability_result and profitability_result[vs_key] is not None:
                vs_industry_metrics.append(vs_key)
                vs_industry_labels.append(vs_industry_all_labels[i])
                vs_industry_values.append(profitability_result[vs_key])
        
        if vs_industry_values:
            # 转为百分比
            vs_industry_values = [v * 100 for v in vs_industry_values]
            
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in vs_industry_values]
            ax.bar(vs_industry_labels, vs_industry_values, color=colors)
            
            ax.set_ylabel('高于/低于行业平均(百分点)')
            ax.set_title('盈利能力与行业平均对比')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, (v, m) in enumerate(zip(vs_industry_values, vs_industry_metrics)):
                if m == 'debt_to_asset_vs_industry':
                    if v >= 0:
                        ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                    else:
                        ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                else:
                    if v >= 0:
                        ax.text(i, v + 0.1, f'+{v:.2f}', ha='center')
                    else:
                        ax.text(i, v - 0.1, f'{v:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少与行业对比数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 盈利能力分析"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
    
    def plot_solvency_analysis(self, solvency_result, **kwargs):
        """
        绘制偿债能力和财务状况分析图表
        
        Args:
            solvency_result (dict): 偿债能力分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not solvency_result:
            logger.error("没有偿债能力分析结果可供可视化")
            return None
            
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 流动性指标柱状图
        ax = axes[0, 0]
        liquidity_metrics = ['current_ratio', 'quick_ratio']
        liquidity_labels = ['流动比率', '速动比率']
        liquidity_values = []
        
        for metric in liquidity_metrics:
            val = solvency_result.get(metric)
            liquidity_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(liquidity_values) if v is not None]
        if valid_indices:
            valid_liquidity_labels = [liquidity_labels[i] for i in valid_indices]
            valid_liquidity_values = [liquidity_values[i] for i in valid_indices]
            
            # 设置颜色：绿色表示健康，黄色表示警惕，红色表示风险
            colors = []
            for i, metric in enumerate(liquidity_metrics):
                if i < len(valid_indices):
                    if metric == 'current_ratio':
                        if valid_liquidity_values[i] >= 2:
                            colors.append('#2ca02c')  # 绿色
                        elif valid_liquidity_values[i] >= 1:
                            colors.append('#ff7f0e')  # 黄色
                        else:
                            colors.append('#d62728')  # 红色
                    elif metric == 'quick_ratio':
                        if valid_liquidity_values[i] >= 1:
                            colors.append('#2ca02c')  # 绿色
                        elif valid_liquidity_values[i] >= 0.5:
                            colors.append('#ff7f0e')  # 黄色
                        else:
                            colors.append('#d62728')  # 红色
            
            ax.bar(valid_liquidity_labels, valid_liquidity_values, color=colors)
            
            ax.set_ylabel('比率')
            ax.set_title('流动性指标')
            
            # 添加数值标签
            for i, v in enumerate(valid_liquidity_values):
                ax.text(i, v + 0.1, f'{v:.2f}', ha='center')
                
            # 添加参考线
            if 'current_ratio' in liquidity_metrics and liquidity_metrics.index('current_ratio') in valid_indices:
                ax.axhline(y=2, color='g', linestyle='--', alpha=0.3)
                ax.axhline(y=1, color='r', linestyle='--', alpha=0.3)
                
            if 'quick_ratio' in liquidity_metrics and liquidity_metrics.index('quick_ratio') in valid_indices:
                ax.axhline(y=1, color='g', linestyle='--', alpha=0.3)
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '缺少流动性指标数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 2. 杠杆指标柱状图
        ax = axes[0, 1]
        leverage_metrics = ['debt_to_asset', 'debt_to_equity', 'interest_coverage']
        leverage_labels = ['资产负债率', '权益乘数', '利息保障倍数']
        leverage_values = []
        
        for metric in leverage_metrics:
            val = solvency_result.get(metric)
            leverage_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(leverage_values) if v is not None]
        if valid_indices:
            valid_leverage_labels = [leverage_labels[i] for i in valid_indices]
            valid_leverage_values = [leverage_values[i] for i in valid_indices]
            
            # 对于资产负债率，转换为百分比形式
            for i, metric in enumerate(leverage_metrics):
                if i < len(valid_indices) and metric == 'debt_to_asset':
                    valid_leverage_values[i] *= 100
            
            # 设置颜色：资产负债率和权益乘数越低越好（绿色），利息保障倍数越高越好（绿色）
            colors = []
            for i, metric in enumerate(leverage_metrics):
                if i < len(valid_indices):
                    if metric == 'debt_to_asset':
                        if valid_leverage_values[i] <= 40:
                            colors.append('#2ca02c')  # 绿色
                        elif valid_leverage_values[i] <= 60:
                            colors.append('#ff7f0e')  # 黄色
                        else:
                            colors.append('#d62728')  # 红色
                    elif metric == 'debt_to_equity':
                        if valid_leverage_values[i] <= 1:
                            colors.append('#2ca02c')  # 绿色
                        elif valid_leverage_values[i] <= 2:
                            colors.append('#ff7f0e')  # 黄色
                        else:
                            colors.append('#d62728')  # 红色
                    elif metric == 'interest_coverage':
                        if valid_leverage_values[i] >= 5:
                            colors.append('#2ca02c')  # 绿色
                        elif valid_leverage_values[i] >= 2:
                            colors.append('#ff7f0e')  # 黄色
                        else:
                            colors.append('#d62728')  # 红色
            
            ax.bar(valid_leverage_labels, valid_leverage_values, color=colors)
            
            # 对于包含不同单位的指标，使用通用y轴标签
            ax.set_ylabel('比率 / 百分比 / 倍数')
            ax.set_title('杠杆指标')
            
            # 添加数值标签
            for i, (v, m) in enumerate(zip(valid_leverage_values, [leverage_metrics[j] for j in valid_indices])):
                if m == 'debt_to_asset':
                    ax.text(i, v + 3, f'{v:.1f}%', ha='center')
                else:
                    ax.text(i, v + 0.5, f'{v:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少杠杆指标数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 3. 历史财务健康度趋势图（如果有）
        ax = axes[1, 0]
        if isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            if 'date' in self.financial_data.columns:
                date_col = 'date'
            elif self.financial_data.index.name == 'date' or isinstance(self.financial_data.index, pd.DatetimeIndex):
                date_col = self.financial_data.index
            else:
                date_col = None
                
            if date_col is not None:
                solvency_trend_metrics = {
                    'current_ratio': '流动比率',
                    'quick_ratio': '速动比率',
                    'debt_to_asset': '资产负债率'
                }
                
                for eng_key, cn_key in solvency_trend_metrics.items():
                    # 检查是否存在中文或英文列名
                    col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                    
                    if col_name in self.financial_data.columns:
                        # 对于资产负债率，转换为百分比形式
                        data = self.financial_data[col_name]
                        if eng_key == 'debt_to_asset':
                            data = data * 100
                            
                        ax.plot(date_col, data, label=cn_key, marker='o')
                
                ax.set_title('历史财务健康度趋势')
                ax.set_xlabel('日期')
                ax.set_ylabel('比率 / 百分比')
                ax.grid(True)
                ax.legend()
            else:
                ax.text(0.5, 0.5, '缺少日期数据，无法绘制历史趋势', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '缺少历史财务健康度数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 4. 财务健康度与行业对比
        ax = axes[1, 1]
        vs_industry_all_metrics = liquidity_metrics + leverage_metrics
        vs_industry_all_labels = liquidity_labels + leverage_labels
        vs_industry_metrics = []
        vs_industry_labels = []
        vs_industry_values = []
        
        # 查找所有有行业对比数据的指标
        for i, metric in enumerate(vs_industry_all_metrics):
            vs_key = f'{metric}_vs_industry'
            if vs_key in solvency_result and solvency_result[vs_key] is not None:
                vs_industry_metrics.append(vs_key)
                vs_industry_labels.append(vs_industry_all_labels[i])
                vs_industry_values.append(solvency_result[vs_key])
        
        if vs_industry_values:
            # 对于资产负债率，转换为百分比形式
            for i, metric in enumerate(vs_industry_metrics):
                if metric == 'debt_to_asset_vs_industry':
                    vs_industry_values[i] *= 100
            
            # 设置颜色：流动比率和速动比率高于行业平均为好（绿色），
            # 资产负债率和权益乘数低于行业平均为好（绿色）
            colors = []
            for i, metric in enumerate(vs_industry_metrics):
                if metric in ['current_ratio_vs_industry', 'quick_ratio_vs_industry', 'interest_coverage_vs_industry']:
                    colors.append('#2ca02c' if vs_industry_values[i] > 0 else '#d62728')
                else:
                    colors.append('#2ca02c' if vs_industry_values[i] < 0 else '#d62728')
            
            ax.bar(vs_industry_labels, vs_industry_values, color=colors)
            
            ax.set_ylabel('高于/低于行业平均')
            ax.set_title('财务健康度与行业平均对比')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, (v, m) in enumerate(zip(vs_industry_values, vs_industry_metrics)):
                if m == 'debt_to_asset_vs_industry':
                    if v >= 0:
                        ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                    else:
                        ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                else:
                    if v >= 0:
                        ax.text(i, v + 0.1, f'+{v:.2f}', ha='center')
                    else:
                        ax.text(i, v - 0.1, f'{v:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少与行业对比数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 财务健康度分析"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
        
    def plot_dividend_analysis(self, dividend_result, **kwargs):
        """
        绘制股息和分红分析图表
        
        Args:
            dividend_result (dict): 股息分析结果
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        if not dividend_result:
            logger.error("没有股息分析结果可供可视化")
            return None
            
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 股息率和支付率柱状图
        ax = axes[0, 0]
        metrics = ['dividend_yield', 'dividend_payout_ratio']
        labels = ['股息率', '股息支付率']
        values = []
        
        for metric in metrics:
            val = dividend_result.get(metric)
            values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if valid_indices:
            valid_labels = [labels[i] for i in valid_indices]
            valid_values = [values[i] * 100 for i in valid_indices]  # 转为百分比
            
            ax.bar(valid_labels, valid_values)
            
            ax.set_ylabel('百分比(%)')
            ax.set_title('股息率和支付率')
            
            # 添加数值标签
            for i, v in enumerate(valid_values):
                ax.text(i, v + 1, f'{v:.2f}%', ha='center')
        else:
            ax.text(0.5, 0.5, '缺少股息率和支付率数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 2. 历史分红趋势图（如果有）
        ax = axes[0, 1]
        if isinstance(self.financial_data, pd.DataFrame) and not self.financial_data.empty:
            if 'date' in self.financial_data.columns:
                date_col = 'date'
            elif self.financial_data.index.name == 'date' or isinstance(self.financial_data.index, pd.DatetimeIndex):
                date_col = self.financial_data.index
            else:
                date_col = None
                
            if date_col is not None:
                dividend_trend_metrics = {
                    'dividend_per_share': '每股股息',
                    'dividend_yield': '股息率'
                }
                
                for eng_key, cn_key in dividend_trend_metrics.items():
                    # 检查是否存在中文或英文列名
                    col_name = eng_key if eng_key in self.financial_data.columns else cn_key
                    
                    if col_name in self.financial_data.columns:
                        # 对于股息率，转换为百分比形式
                        data = self.financial_data[col_name]
                        if eng_key == 'dividend_yield':
                            data = data * 100
                            # 使用右边的Y轴
                            ax2 = ax.twinx()
                            ax2.plot(date_col, data, label=cn_key, color='r', marker='s')
                            ax2.set_ylabel('股息率(%)', color='r')
                            
                            # 添加图例
                            lines2, labels2 = ax2.get_legend_handles_labels()
                        else:
                            ax.plot(date_col, data, label=cn_key, color='b', marker='o')
                
                ax.set_title('历史分红趋势')
                ax.set_xlabel('日期')
                ax.set_ylabel('每股股息')
                ax.grid(True)
                
                # 合并两个Y轴的图例
                if 'lines2' in locals() and 'labels2' in locals():
                    lines1, labels1 = ax.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                else:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, '缺少日期数据，无法绘制历史趋势', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '缺少历史分红数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 3. 股息增长率柱状图
        ax = axes[1, 0]
        growth_metrics = ['dividend_growth', 'dividend_cagr']
        growth_labels = ['股息同比增长率', '股息复合增长率(CAGR)']
        growth_values = []
        
        for metric in growth_metrics:
            val = dividend_result.get(metric)
            growth_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(growth_values) if v is not None]
        if valid_indices:
            valid_growth_labels = [growth_labels[i] for i in valid_indices]
            valid_growth_values = [growth_values[i] * 100 for i in valid_indices]  # 转为百分比
            
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in valid_growth_values]
            ax.bar(valid_growth_labels, valid_growth_values, color=colors)
            
            ax.set_ylabel('增长率(%)')
            ax.set_title('股息增长率')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(valid_growth_values):
                if v >= 0:
                    ax.text(i, v + 3, f'+{v:.1f}%', ha='center')
                else:
                    ax.text(i, v - 3, f'{v:.1f}%', ha='center')
                    
            # 设置y轴范围，确保零线可见
            y_min = min(min(valid_growth_values) - 10, -5)
            y_max = max(max(valid_growth_values) + 10, 5)
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, '缺少股息增长率数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 4. 股息与行业对比
        ax = axes[1, 1]
        vs_industry_metrics = ['dividend_yield_vs_industry', 'dividend_payout_ratio_vs_industry']
        vs_industry_labels = ['股息率对比', '股息支付率对比']
        vs_industry_values = []
        
        for metric in vs_industry_metrics:
            val = dividend_result.get(metric)
            vs_industry_values.append(val)
        
        # 过滤掉没有数据的指标
        valid_indices = [i for i, v in enumerate(vs_industry_values) if v is not None]
        if valid_indices:
            valid_vs_industry_labels = [vs_industry_labels[i] for i in valid_indices]
            valid_vs_industry_values = [vs_industry_values[i] * 100 for i in valid_indices]  # 转为百分点
            
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in valid_vs_industry_values]
            ax.bar(valid_vs_industry_labels, valid_vs_industry_values, color=colors)
            
            ax.set_ylabel('高于/低于行业平均(百分点)')
            ax.set_title('股息与行业平均对比')
            
            # 添加零线
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(valid_vs_industry_values):
                if v >= 0:
                    ax.text(i, v + 0.5, f'+{v:.2f}%', ha='center')
                else:
                    ax.text(i, v - 0.5, f'{v:.2f}%', ha='center')
                    
            # 设置y轴范围，确保零线可见
            y_min = min(min(valid_vs_industry_values) - 2, -1)
            y_max = max(max(valid_vs_industry_values) + 2, 1)
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, '缺少与行业对比数据', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 股息分析"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig

    def plot_industry_comparison(self, result, **kwargs):
        """
        绘制行业对比分析图表
        
        Args:
            result (dict): 分析结果字典
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        # 获取股票信息
        symbol = kwargs.get('symbol', '')
        stock_name = ''
        
        if self.stock_info:
            if isinstance(self.stock_info, dict):
                symbol = self.stock_info.get('symbol', symbol)
                stock_name = self.stock_info.get('name', '')
            elif isinstance(self.stock_info, pd.DataFrame) and not self.stock_info.empty:
                if 'symbol' in self.stock_info.columns:
                    symbol = self.stock_info['symbol'].iloc[0]
                if 'name' in self.stock_info.columns:
                    stock_name = self.stock_info['name'].iloc[0]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 从各个分析结果中提取行业对比数据
        industry_comparison_data = {
            '估值对比': {},
            '成长性对比': {},
            '盈利能力对比': {},
            '财务状况对比': {}
        }
        
        # 估值对比数据
        if 'valuation' in result and result['valuation']:
            for metric in ['pe_ttm', 'pb', 'ps_ttm']:
                stock_val = result['valuation'].get(metric)
                industry_val = result['valuation'].get(f'industry_{metric}')
                if stock_val is not None and industry_val is not None:
                    metric_name = metric.replace('pe_ttm', '市盈率').replace('pb', '市净率').replace('ps_ttm', '市销率')
                    industry_comparison_data['估值对比'][metric_name] = {
                        '个股': stock_val,
                        '行业': industry_val
                    }
        
        # 成长性对比数据
        if 'growth' in result and result['growth']:
            for metric in ['revenue_growth', 'profit_growth', 'eps_growth']:
                stock_val = result['growth'].get(metric)
                industry_val = result['growth'].get(f'industry_{metric}')
                if stock_val is not None and industry_val is not None:
                    metric_name = metric.replace('revenue_growth', '营收增长').replace('profit_growth', '利润增长').replace('eps_growth', 'EPS增长')
                    industry_comparison_data['成长性对比'][metric_name] = {
                        '个股': stock_val * 100,  # 转为百分比
                        '行业': industry_val * 100  # 转为百分比
                    }
        
        # 盈利能力对比数据
        if 'profitability' in result and result['profitability']:
            for metric in ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa']:
                stock_val = result['profitability'].get(metric)
                industry_val = result['profitability'].get(f'industry_{metric}')
                if stock_val is not None and industry_val is not None:
                    metric_name = metric.replace('gross_margin', '毛利率').replace('operating_margin', '营业利润率')\
                                      .replace('net_margin', '净利率').replace('roe', 'ROE').replace('roa', 'ROA')
                    industry_comparison_data['盈利能力对比'][metric_name] = {
                        '个股': stock_val * 100,  # 转为百分比
                        '行业': industry_val * 100  # 转为百分比
                    }
        
        # 财务状况对比数据
        if 'solvency' in result and result['solvency']:
            for metric in ['current_ratio', 'quick_ratio', 'debt_to_asset']:
                stock_val = result['solvency'].get(metric)
                industry_val = result['solvency'].get(f'industry_{metric}')
                if stock_val is not None and industry_val is not None:
                    # 对于资产负债率，转为百分比
                    if metric == 'debt_to_asset':
                        stock_val *= 100
                        industry_val *= 100
                    
                    metric_name = metric.replace('current_ratio', '流动比率').replace('quick_ratio', '速动比率')\
                                      .replace('debt_to_asset', '资产负债率')
                    industry_comparison_data['财务状况对比'][metric_name] = {
                        '个股': stock_val,
                        '行业': industry_val
                    }
        
        # 绘制各类对比图
        for i, (category, data) in enumerate(industry_comparison_data.items()):
            ax = axes[i // 2, i % 2]
            
            if data:
                # 准备绘图数据
                metrics = list(data.keys())
                stock_values = [data[m]['个股'] for m in metrics]
                industry_values = [data[m]['行业'] for m in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                # 绘制柱状图
                ax.bar(x - width/2, stock_values, width, label='个股')
                ax.bar(x + width/2, industry_values, width, label='行业平均')
                
                # 设置标题和标签
                ax.set_title(category)
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                
                # 添加图例
                ax.legend()
                
                # 添加数值标签
                for i, v in enumerate(stock_values):
                    ax.text(i - width/2, v * 1.02, f'{v:.1f}', ha='center', fontsize=9)
                
                for i, v in enumerate(industry_values):
                    ax.text(i + width/2, v * 1.02, f'{v:.1f}', ha='center', fontsize=9)
                
                # 如果是百分比类指标，添加百分比符号到y轴标签
                if category in ['成长性对比', '盈利能力对比'] or '资产负债率' in metrics:
                    ax.set_ylabel('百分比(%)')
                elif category == '估值对比':
                    ax.set_ylabel('倍数')
                else:
                    ax.set_ylabel('数值')
            else:
                ax.text(0.5, 0.5, f'缺少{category}数据', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
        
        # 添加图表标题
        title = f"{symbol} {stock_name} 行业对比分析"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
