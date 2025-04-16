"""
聚类分析模块

提供股票数据的聚类分析功能，包括KMeans聚类、分组统计和可视化展示
"""

from . import Analyzer
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Set

from src.utils.column_mapping import (
    StandardColumns,
    standardize_columns,
    detect_and_log_column_issues,
    suggest_column_mappings,
    get_standard_column_list
)

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has 0 variance")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")

# 设置logging
logger = logging.getLogger(__name__)

# 设置matplotlib支持中文显示
try:
    import matplotlib
    # 尝试使用系统支持的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Heiti TC', 'LiHei Pro', 
                                      'Arial Unicode MS', 'DejaVu Sans', 'DejaVu Sans Mono', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    logger.info("已设置matplotlib中文字体支持")
except Exception as e:
    logger.warning(f"设置matplotlib中文字体支持失败: {e}")

class ClusteringAnalyzer(Analyzer):
    """聚类分析器，提供股票聚类分析功能"""
    
    def __init__(self, data=None):
        """
        初始化聚类分析器
        
        Args:
            data (pandas.DataFrame, optional): 分析数据. Defaults to None.
        """
        super().__init__(data)
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.features = None
        self.scaled_data = None
        self.cluster_results = None
        
    def validate_input(self):
        """验证输入数据是否包含所需的列"""
        required_columns = [
            StandardColumns.DATE,
            StandardColumns.OPEN,
            StandardColumns.HIGH,
            StandardColumns.LOW,
            StandardColumns.CLOSE,
            StandardColumns.VOLUME,
            StandardColumns.SYMBOL
        ]
        return super().validate_input(required_columns)
    
    def analyze(self, methods=None, params=None):
        """
        执行聚类分析
        
        Args:
            methods (list, optional): 分析方法列表，例如 ['kmeans', 'hierarchical']
            params (dict, optional): 分析参数
            
        Returns:
            dict: 分析结果
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可进行聚类分析")
            return None
        
        if methods is None:
            methods = ['kmeans']  # 默认使用KMeans聚类
        
        results = {}
        
        try:
            # 检查数据是否需要预处理
            if not params or 'preprocess' not in params or params['preprocess']:
                logger.info("对数据进行预处理...")
                if self.preprocess_data():
                    logger.info("数据预处理完成")
                else:
                    logger.error("数据预处理失败")
                    return None
            
            # 执行指定的分析方法
            for method in methods:
                if method.lower() == 'kmeans':
                    logger.info("执行KMeans聚类分析...")
                    if params is None:
                        params = {}
                    # 构建KMeans特定的参数字典
                    kmeans_params = params.get('kmeans', {})
                    result = self.kmeans_clustering(kmeans_params)
                    if result:
                        results['kmeans'] = result
                        logger.info("KMeans聚类分析完成")
                    else:
                        logger.error("KMeans聚类分析失败")
                elif method.lower() == 'hierarchical':
                    logger.info("执行层次聚类分析...")
                    # 目前不支持层次聚类，后续可以实现
                    logger.warning("层次聚类分析暂未实现")
                else:
                    logger.warning(f"不支持的聚类方法: {method}")
            
            return results
        
        except Exception as e:
            logger.error(f"执行聚类分析时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def preprocess_data(self):
        """
        预处理数据，包括:
        1. 标准化列名
        2. 转换数据类型
        3. 处理缺失值
        4. 检查数据有效性
        
        Returns:
            bool: 预处理是否成功
        """
        if self.data is None or self.data.empty:
            logger.error("无数据可进行预处理")
            return False
            
        try:
            # 记录原始数据状态
            logger.info(f"原始数据大小: {self.data.shape}")
            logger.info(f"原始数据列: {list(self.data.columns)}")
            
            # 1. 使用Analyzer基类的方法标准化列名
            if not self._standardized:
                logger.info("标准化数据列名...")
                self.standardize_data()
            
            # 2. 验证必要列是否存在
            if not self.validate_input():
                logger.error("缺少关键列进行聚类分析，无法继续")
                return False
                
            # 3. 转换数据类型
            # 确保日期列为日期类型
            date_col = StandardColumns.DATE.value
            if date_col in self.data.columns:
                self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
                
            # 确保数值列为数值类型
            numeric_columns = [
                StandardColumns.OPEN.value,
                StandardColumns.HIGH.value,
                StandardColumns.LOW.value,
                StandardColumns.CLOSE.value,
                StandardColumns.VOLUME.value,
                StandardColumns.CHANGE_PCT.value
            ]
            
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # 4. 处理缺失值
            # 记录缺失值情况
            null_counts = self.data.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"数据包含缺失值: \n{null_counts[null_counts > 0]}")
                
                # 使用前向填充和均值填充处理缺失值
                symbol_col = StandardColumns.SYMBOL.value
                for col in self.data.columns:
                    if self.data[col].isnull().any():
                        # 先尝试前向填充
                        self.data[col] = self.data.groupby(symbol_col)[col].fillna(method='ffill')
                        
                        # 如果仍有缺失值，使用均值填充
                        if self.data[col].isnull().any():
                            mean_values = self.data.groupby(symbol_col)[col].transform('mean')
                            self.data[col] = self.data[col].fillna(mean_values)
                            
                        # 如果仍有缺失值，使用全局均值填充
                        if self.data[col].isnull().any():
                            if pd.api.types.is_numeric_dtype(self.data[col]):
                                global_mean = self.data[col].mean()
                                self.data[col] = self.data[col].fillna(global_mean)
                            else:
                                # 非数值列使用最常见值填充
                                most_common = self.data[col].mode()[0]
                                self.data[col] = self.data[col].fillna(most_common)
            
            # 检查是否仍有缺失值
            remaining_nulls = self.data.isnull().sum().sum()
            if remaining_nulls > 0:
                logger.warning(f"预处理后仍有 {remaining_nulls} 个缺失值")
                # 最后的手段：删除仍有缺失值的行
                self.data = self.data.dropna()
                logger.info(f"删除缺失值后的数据大小: {self.data.shape}")
            
            # 5. 添加默认的变化百分比列，如果不存在
            change_pct_col = StandardColumns.CHANGE_PCT.value
            if change_pct_col not in self.data.columns:
                close_col = StandardColumns.CLOSE.value
                if close_col in self.data.columns:
                    logger.info(f"计算 {change_pct_col} 列")
                    # 按股票和日期排序
                    self.data = self.data.sort_values([symbol_col, date_col])
                    # 计算每个股票的日收益率
                    self.data[change_pct_col] = self.data.groupby(symbol_col)[close_col].pct_change() * 100
            
            # 删除极值
            for col in numeric_columns:
                if col in self.data.columns:
                    q_low = self.data[col].quantile(0.001)
                    q_high = self.data[col].quantile(0.999)
                    self.data = self.data[(self.data[col] >= q_low) & (self.data[col] <= q_high)]
            
            logger.info(f"预处理完成，最终数据大小: {self.data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def kmeans_clustering(self, params=None):
        """
        使用KMeans算法进行聚类分析
        
        Args:
            params (dict, optional): 聚类参数
                n_clusters (int): 聚类数量，默认为3
                features (list): 用于聚类的特征，默认为['open', 'close', 'high', 'low', 'volume', 'change_pct']
                random_state (int): 随机种子，默认为42
                max_iter (int): 最大迭代次数，默认为300
                n_init (int): 使用不同初始中心点运行算法的次数，默认为10
                
        Returns:
            dict: 聚类结果
        """
        if self.data is None or self.data.empty:
            logger.error("没有数据可进行聚类分析")
            return None
            
        # 默认参数
        default_params = {
            'n_clusters': 3,
            'features': [
                StandardColumns.OPEN.value, 
                StandardColumns.CLOSE.value, 
                StandardColumns.HIGH.value, 
                StandardColumns.LOW.value, 
                StandardColumns.VOLUME.value, 
                StandardColumns.CHANGE_PCT.value
            ],
            'random_state': 42,
            'max_iter': 300,
            'n_init': 10
        }
        
        # 更新参数
        if params is not None:
            # 如果传入的是StandardColumns枚举，提取其值
            if 'features' in params:
                std_features = []
                for feature in params['features']:
                    if isinstance(feature, StandardColumns):
                        std_features.append(feature.value)
                    else:
                        std_features.append(feature)
                params['features'] = std_features
            default_params.update(params)
        
        n_clusters = default_params['n_clusters']
        features = default_params['features']
        random_state = default_params['random_state']
        max_iter = default_params['max_iter']
        n_init = default_params['n_init']
        
        # 参数验证
        if n_clusters < 2:
            logger.warning(f"聚类数量 {n_clusters} 小于2，将使用默认值2")
            n_clusters = 2
        
        if not features:
            logger.error("未指定有效的聚类特征")
            return None
            
        # 检查特征是否存在于数据中
        valid_features = []
        for feature in features:
            if feature in self.data.columns:
                valid_features.append(feature)
            else:
                logger.warning(f"特征 '{feature}' 在数据中不存在")
                
        if not valid_features:
            logger.error("指定的特征在数据中不存在")
            return None
        
        logger.info(f"使用以下特征进行聚类: {valid_features}")
        
        # 准备数据
        try:
            # 提取特征数据
            X = self.data[valid_features].copy()
            
            # 检查数据是否有效
            if X.empty:
                logger.error("提取的特征数据为空")
                return None
            
            # 处理缺失值
            if X.isnull().any().any():
                logger.warning("数据中存在缺失值，将使用均值填充")
                for col in X.columns:
                    if X[col].isnull().any():
                        mean_val = X[col].mean()
                        X[col] = X[col].fillna(mean_val)
            
            # 检查常数列
            constant_columns = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    logger.warning(f"列 '{col}' 为常数列，可能会影响聚类结果")
                    constant_columns.append(col)
            
            # 剔除常数列
            if constant_columns:
                X = X.drop(columns=constant_columns)
                logger.info(f"移除了以下常数列: {constant_columns}")
                valid_features = [col for col in valid_features if col not in constant_columns]
            
            # 特征标准化
            if len(X.columns) == 0:
                logger.error("数据处理后没有有效特征可用于聚类")
                return None
                
            self.features = valid_features
            X_scaled = self.scaler.fit_transform(X)
            self.scaled_data = X_scaled
            
            # 执行KMeans聚类
            logger.info(f"开始执行KMeans聚类 (n_clusters={n_clusters}, max_iter={max_iter}, n_init={n_init})")
            # For scikit-learn >= 1.2.0, use n_init='auto' or a specific number
            self.kmeans_model = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state, 
                max_iter=max_iter, 
                n_init=10
            )
            
            # 拟合模型
            self.kmeans_model.fit(X_scaled)
            
            # 获取聚类标签
            labels = self.kmeans_model.labels_
            
            # 计算轮廓分数 (如果有多个聚类)
            silhouette_score = None
            if n_clusters > 1:
                try:
                    from sklearn.metrics import silhouette_score as skl_silhouette_score
                    silhouette_score = skl_silhouette_score(X_scaled, labels)
                    logger.info(f"聚类轮廓分数: {silhouette_score:.4f}")
                except Exception as e:
                    logger.warning(f"计算轮廓分数失败: {e}")
            
            # 获取聚类中心
            centroids = self.kmeans_model.cluster_centers_
            
            # 将聚类标签添加到数据框中
            self.data['cluster'] = labels
            
            # 计算每个聚类的统计信息
            cluster_stats = self.calculate_cluster_stats()
            
            # 保存结果
            result = {
                'model': self.kmeans_model,
                'features': self.features,
                'labels': labels,
                'centroids': centroids,
                'scaled_data': X_scaled,
                'scaler': self.scaler,
                'cluster_stats': cluster_stats
            }
            
            if silhouette_score is not None:
                result['silhouette_score'] = silhouette_score
                
            self.cluster_results = result
            logger.info(f"KMeans聚类分析完成，生成了 {n_clusters} 个聚类")
            return result
            
        except Exception as e:
            logger.error(f"KMeans聚类分析失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def determine_optimal_clusters(self, max_clusters=10, params=None):
        """
        使用肘部法则确定最佳聚类数量
        
        Args:
            max_clusters (int): 最大聚类数
            params (dict): 聚类参数
            
        Returns:
            tuple: (最佳聚类数, inertia值列表)
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            import pandas as pd
        except ImportError as e:
            logger.error(f"导入所需库失败: {e}")
            return None, None
            
        if params is None:
            params = {}
            
        features = params.get('features', ['open', 'close', 'high', 'low', 'volume', 'change_percent'])
        
        # 检查特征列是否存在
        missing_features = [f for f in features if f not in self.data.columns]
        if missing_features:
            logger.warning(f"以下特征在数据中不存在: {missing_features}")
            features = [f for f in features if f in self.data.columns]
            
        if not features:
            logger.error("没有可用特征进行聚类")
            return None, None
            
        # 提取特征数据
        feature_data = self.data[features].copy()
        
        # 数据预处理
        # 删除全为NaN的列
        null_columns = feature_data.columns[feature_data.isnull().all()].tolist()
        if null_columns:
            logger.warning(f"删除全为空值的列: {null_columns}")
            feature_data = feature_data.drop(columns=null_columns)
            features = [f for f in features if f not in null_columns]
        
        # 对于每一列，使用该列的均值填充NaN
        for col in feature_data.columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
            if feature_data[col].isnull().any():
                col_mean = feature_data[col].mean()
                feature_data[col] = feature_data[col].fillna(col_mean)
        
        # 标准化特征
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # 处理可能的NaN和Inf
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                logger.warning("标准化后的数据包含NaN或Inf值，进行替换...")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
                
        except Exception as e:
            logger.error(f"标准化特征失败: {e}")
            return None, None
            
        # 计算不同聚类数的inertia值
        inertia_values = []
        logger.info(f"计算1到{max_clusters}个聚类的inertia值...")
        
        for k in range(1, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_features)
                inertia_values.append(kmeans.inertia_)
                logger.info(f"聚类数 {k}: inertia = {kmeans.inertia_:.2f}")
            except Exception as e:
                logger.error(f"计算聚类数 {k} 的inertia值失败: {e}")
                inertia_values.append(None)
        
        # 如果有任何inertia值计算失败，返回None
        if None in inertia_values:
            logger.warning("部分聚类数的inertia值计算失败，无法确定最佳聚类数")
            return None, inertia_values
        
        # 使用肘部法则确定最佳聚类数
        from scipy.signal import argrelextrema
        import numpy as np
        
        # 一阶导数（差分）
        deltas = np.diff(inertia_values)
        
        # 计算加速度（二阶导数）
        accelerations = np.diff(deltas)
        
        # 寻找加速度最大的点
        try:
            acc_maxima = argrelextrema(accelerations, np.greater)[0] + 2  # +2是因为二阶导数的索引偏移
            
            # 如果找不到明显的肘部点，使用简单的方法：找到斜率变化最大的点
            if len(acc_maxima) == 0:
                logger.info("未找到明显的肘部点，使用简单方法确定最佳聚类数")
                # 寻找相对下降幅度最小的点
                relative_drops = [-delta / inertia_values[i] for i, delta in enumerate(deltas)]
                optimal_k = np.argmin(relative_drops) + 2  # +2是因为差分的索引偏移和起始聚类数为1
            else:
                # 使用加速度最大的点
                optimal_k = acc_maxima[0]
                
            # 如果最佳聚类数太小或太大，可能不合理，设置一个范围
            if optimal_k < 2:
                optimal_k = 2
            elif optimal_k > max_clusters:
                optimal_k = max_clusters
                
        except Exception as e:
            logger.error(f"确定最佳聚类数失败: {e}")
            # 退化为简单启发式：选择靠前的拐点，比如聚类数为3
            optimal_k = 3
            
        logger.info(f"确定的最佳聚类数: {optimal_k}")
        return optimal_k, inertia_values
        
    def visualize(self, result=None, plot_type='clusters', **kwargs):
        """
        可视化聚类结果
        
        Args:
            result (dict, optional): 聚类结果. Defaults to None.
            plot_type (str, optional): 图表类型. Defaults to 'clusters'.
            **kwargs: 其他绘图参数
            
        Returns:
            matplotlib.Figure: 可视化结果
        """
        import numpy as np
        
        if result is None and self.cluster_results is not None:
            result = self.cluster_results
        elif result is None:
            logger.error("没有聚类结果可供可视化")
            return None
            
        # 检查结果有效性
        if not isinstance(result, dict):
            logger.error(f"聚类结果格式无效，应为字典而非 {type(result)}")
            return None
            
        # 检查必要键是否存在
        required_keys = ['n_clusters', 'features']
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            logger.error(f"聚类结果缺少必要的键: {missing_keys}")
            return None
            
        # 检查labels字段是否存在并有效
        if 'labels' in result:
            labels = result['labels']
            # 检查是否为标量或者None
            if labels is None:
                logger.error("聚类标签为None")
                return None
            elif np.isscalar(labels):
                logger.error(f"聚类标签为标量值 {labels}，而非数组")
                return None
            
            # 检查标签是否包含NaN或Inf
            try:
                if np.isnan(labels).any() or np.isinf(labels).any():
                    logger.warning("聚类标签包含NaN或Inf值，尝试修复")
                    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                    # 确保标签是整数
                    labels = labels.astype(int)
                    result['labels'] = labels
            except (TypeError, ValueError) as e:
                logger.error(f"检查聚类标签时出错: {e}")
                return None
        else:
            logger.error("聚类结果缺少'labels'字段")
            return None
        
        # 分发到不同的可视化方法
        if plot_type == 'clusters':
            return self.plot_clusters(result, **kwargs)
        elif plot_type == 'elbow':
            return self.plot_elbow_method(**kwargs)
        elif plot_type == 'feature_distribution':
            return self.plot_feature_distribution(result, **kwargs)
        elif plot_type == 'centroids':
            return self.plot_cluster_centroids(result, **kwargs)
        else:
            logger.warning(f"未知的可视化类型: {plot_type}")
            return None
            
    def plot_clusters(self, result, **kwargs):
        """
        绘制聚类散点图

        Args:
            result (dict): 聚类结果
            **kwargs: 其他参数
                use_pca (bool): 是否使用PCA降维, 默认为True
                plot_3d (bool): 是否绘制3D图，默认为False
                selected_features (list): 用于绘图的特征名，默认使用结果中的所有特征
                figsize (tuple): 图表大小
                title (str): 图表标题
                stock_names (dict): 股票代码到股票名称的映射

        Returns:
            matplotlib.figure.Figure: 绘制的图表对象
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import logging

        logger = logging.getLogger(__name__)

        try:
            # 检查结果是否有效
            if result is None or 'labels' not in result or 'features' not in result:
                logger.error("聚类结果无效，无法生成图表")
                return None

            # 获取参数
            use_pca = kwargs.get('use_pca', True)
            plot_3d = kwargs.get('plot_3d', False)
            selected_features = kwargs.get('selected_features', result['features'])
            figsize = kwargs.get('figsize', (12, 10))
            title = kwargs.get('title', f'聚类分析 ({len(set(result["labels"]))}个簇)')
            stock_names = kwargs.get('stock_names', {})
            
            # 确保所选特征都存在于数据中
            features_to_use = [f for f in selected_features if f in self.data.columns]
            if len(features_to_use) < 2:
                logger.error(f"可用特征数量不足，至少需要2个特征: {features_to_use}")
                return None

            # 提取特征数据
            feature_data = self.data[features_to_use].copy()
            
            # 验证数据有效性，处理NaN值
            if feature_data.isnull().any().any():
                logger.warning("特征数据包含NaN值，尝试填充...")
                # 用每列的均值填充NaN
                for col in feature_data.columns:
                    if feature_data[col].isnull().any():
                        feature_data[col] = feature_data[col].fillna(feature_data[col].mean())
            
            # 再次检查是否还有NaN值
            if feature_data.isnull().any().any():
                logger.warning("仍然有NaN值，使用0填充")
                feature_data = feature_data.fillna(0)
            
            # 检查数据是否为空
            if len(feature_data) == 0:
                logger.error("没有有效的数据点用于可视化")
                return None

            # 数据标准化
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)
            except Exception as e:
                logger.error(f"数据标准化失败: {e}")
                # 尝试清理数据后再标准化
                for col in feature_data.columns:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                feature_data = feature_data.fillna(0)
                
                try:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(feature_data)
                except Exception as e:
                    logger.error(f"数据清理后标准化仍失败: {e}")
                    return None

            # 检查标准化后的数据是否有NaN或Inf
            if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
                logger.warning("标准化后的数据包含NaN或Inf值，将替换为0")
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

            # 如果使用PCA降维
            if use_pca:
                try:
                    # 确定使用的维度数
                    n_components = 3 if plot_3d else 2
                    n_components = min(n_components, scaled_data.shape[1], scaled_data.shape[0])
                    
                    # 执行PCA
                    pca = PCA(n_components=n_components)
                    reduced_data = pca.fit_transform(scaled_data)
                    
                    # 如果返回的数组是1维的，尝试重塑
                    if reduced_data.ndim == 1:
                        reduced_data = reduced_data.reshape(-1, 1)
                        if n_components > 1:
                            # 添加零列来满足维度要求
                            zeros = np.zeros((reduced_data.shape[0], n_components - 1))
                            reduced_data = np.hstack((reduced_data, zeros))
                    
                    # 说明PCA贡献率
                    explained_variance = pca.explained_variance_ratio_
                    logger.info(f"PCA解释的方差比例: {explained_variance}")
                    
                    var_explained = np.sum(explained_variance)
                    logger.info(f"前{n_components}个主成分解释的总方差比例: {var_explained:.2%}")
                    
                except Exception as e:
                    logger.error(f"PCA降维失败: {e}")
                    if len(features_to_use) >= 2 and not plot_3d:
                        # 如果PCA失败，直接使用前两个特征
                        logger.info("尝试直接使用前两个特征进行可视化")
                        reduced_data = scaled_data[:, :2]
                    else:
                        return None
            else:
                # 不使用PCA，直接使用前2-3个特征
                n_features = min(3 if plot_3d else 2, scaled_data.shape[1])
                reduced_data = scaled_data[:, :n_features]

            # 确保reduced_data至少是2D的
            if reduced_data.ndim == 1:
                reduced_data = reduced_data.reshape(-1, 1)
                # 如果是1D，添加一个零列
                zeros = np.zeros((reduced_data.shape[0], 1))
                reduced_data = np.hstack((reduced_data, zeros))

            # 创建图表
            fig = plt.figure(figsize=figsize)
            
            # 获取聚类标签和唯一标签
            labels = result['labels']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # 获取颜色映射
            colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
            
            # 创建图表对象
            if plot_3d and reduced_data.shape[1] >= 3:
                ax = fig.add_subplot(111, projection='3d')
                
                # 绘制每个聚类
                for i, label in enumerate(unique_labels):
                    cluster_data = reduced_data[labels == label]
                    
                    if len(cluster_data) == 0:
                        continue
                        
                    ax.scatter(
                        cluster_data[:, 0],
                        cluster_data[:, 1],
                        cluster_data[:, 2],
                        c=[colors[i]],
                        label=f'Cluster {label}',
                        alpha=0.7,
                        s=50
                    )
                    
                # 设置轴标签
                if use_pca:
                    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    if reduced_data.shape[1] >= 3:
                        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
                else:
                    ax.set_xlabel(features_to_use[0])
                    ax.set_ylabel(features_to_use[1])
                    if reduced_data.shape[1] >= 3:
                        ax.set_zlabel(features_to_use[2])
            else:
                # 2D图
                ax = fig.add_subplot(111)
                
                # 绘制每个聚类
                for i, label in enumerate(unique_labels):
                    cluster_data = reduced_data[labels == label]
                    
                    if len(cluster_data) == 0:
                        continue
                        
                    # 确保数据至少是2D的
                    if cluster_data.shape[1] < 2:
                        # 添加零列
                        zeros = np.zeros((cluster_data.shape[0], 1))
                        cluster_data = np.hstack((cluster_data, zeros))
                        
                    ax.scatter(
                        cluster_data[:, 0],
                        cluster_data[:, 1],
                        c=[colors[i]],
                        label=f'Cluster {label}',
                        alpha=0.7,
                        s=50
                    )
                
                # 如果数据中有股票代码，添加标签
                if 'stock_id' in self.data.columns:
                    stocks = self.data['stock_id'].unique()
                    
                    for stock in stocks:
                        # 找到该股票的一个数据点
                        stock_idx = self.data['stock_id'] == stock
                        if sum(stock_idx) > 0:
                            stock_data_idx = np.where(stock_idx)[0][0]
                            
                            # 获取该点的坐标
                            x, y = reduced_data[stock_data_idx, :2]
                            
                            # 获取股票名称（如果有）
                            stock_name = stock_names.get(stock, stock)
                            
                            # 添加标签
                            ax.text(x, y, stock_name, fontsize=8)
                
                # 设置轴标签
                if use_pca:
                    ax.set_xlabel(f'主成分1 ({explained_variance[0]:.2%})')
                    ax.set_ylabel(f'主成分2 ({explained_variance[1]:.2%})')
                else:
                    ax.set_xlabel(features_to_use[0])
                    ax.set_ylabel(features_to_use[1])
            
            # 添加图例和标题
            ax.legend(title='聚类')
            ax.set_title(title)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制聚类散点图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def plot_elbow_method(self, **kwargs):
        """
        绘制肘部法则图（确定最佳聚类数）
        
        Args:
            **kwargs: 绘图参数
            
        Returns:
            matplotlib.Figure: 肘部法则图
        """
        # 获取参数
        figsize = kwargs.get('figsize', (10, 6))
        max_clusters = kwargs.get('max_clusters', 10)
        kmeans_params = kwargs.get('kmeans_params', {})
        
        # 计算不同聚类数的惯性值
        _, inertia_values = self.determine_optimal_clusters(max_clusters, kmeans_params)
        
        if not inertia_values:
            logger.error("获取惯性值失败")
            return None
            
        # 绘制图表
        plt.figure(figsize=figsize)
        plt.plot(range(1, len(inertia_values) + 1), inertia_values, 'bo-')
        plt.xlabel('聚类数量')
        plt.ylabel('惯性值 (簇内平方和)')
        plt.title('K均值聚类肘部法则图')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt.gcf()
        
    def plot_feature_distribution(self, result, **kwargs):
        """
        绘制特征分布图
        
        Args:
            result (dict): 聚类结果
            **kwargs: 绘图参数
            
        Returns:
            matplotlib.Figure: 特征分布图
        """
        if 'features' not in result or 'labels' not in result:
            logger.error("聚类结果缺少必要信息")
            return None
            
        # 获取参数
        figsize = kwargs.get('figsize', (15, 10))
        selected_features = kwargs.get('selected_features', result['features'])
        n_clusters = result['n_clusters']
        
        # 过滤选择的特征
        valid_features = [f for f in selected_features if f in result['features']]
        
        if not valid_features:
            logger.error("没有有效的特征可供绘制")
            return None
            
        # 创建画布
        fig = plt.figure(figsize=figsize)
        
        try:
            # 计算子图行列数
            n_features = len(valid_features)
            n_cols = min(3, n_features)
            n_rows = math.ceil(n_features / n_cols)
            
            # 获取数据和标签
            data = result.get('data')
            if data is None or data.empty:
                logger.error("聚类结果中没有数据")
                return None
            
            labels = result['labels']
            if len(labels) != data.shape[0]:
                logger.error(f"标签数量 ({len(labels)}) 与数据行数 ({data.shape[0]}) 不匹配")
                return None
            
            # 为每个特征创建子图
            for i, feature in enumerate(valid_features):
                # 检查特征是否存在于数据中
                if feature not in data.columns:
                    logger.warning(f"特征 '{feature}' 不在数据中，跳过")
                    continue
                
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                
                # 检查数据是否有效
                feature_data = data[feature]
                if feature_data.isnull().all():
                    logger.warning(f"特征 '{feature}' 的数据全为 NaN，跳过")
                    ax.text(0.5, 0.5, f"特征 '{feature}' 无有效数据", 
                            horizontalalignment='center', verticalalignment='center')
                    ax.set_title(feature)
                    continue
                
                # 为每个聚类绘制分布图
                for cluster_id in range(n_clusters):
                    cluster_data = feature_data[labels == cluster_id]
                    
                    if cluster_data.empty:
                        logger.warning(f"聚类 {cluster_id} 没有数据点")
                        continue
                    
                    # 检查数据是否有足够的方差进行密度估计
                    if len(cluster_data) > 1 and cluster_data.var() > 0:
                        try:
                            # 绘制密度图
                            sns.kdeplot(
                                cluster_data,
                                ax=ax,
                                label=f'簇 {cluster_id}',
                                warn_singular=False  # 禁用奇异矩阵警告
                            )
                        except Exception as e:
                            logger.warning(f"绘制聚类 {cluster_id} 的密度图时出错: {e}")
                            # 尝试使用直方图作为备选
                            try:
                                ax.hist(cluster_data, alpha=0.3, bins=10, label=f'簇 {cluster_id}')
                            except Exception as hist_e:
                                logger.warning(f"绘制聚类 {cluster_id} 的直方图时也出错: {hist_e}")
                    else:
                        # 绘制简单条形图作为替代
                        try:
                            ax.bar([f'簇 {cluster_id}'], [cluster_data.mean()], label=f'簇 {cluster_id}', alpha=0.5)
                            logger.warning(f"特征 '{feature}' 在聚类 {cluster_id} 中的方差为零或数据点不足，绘制条形图")
                        except Exception as e:
                            logger.warning(f"绘制聚类 {cluster_id} 的条形图时出错: {e}")
                    
                    # 绘制聚类平均值的垂直线
                    try:
                        mean_val = cluster_data.mean()
                        if not pd.isna(mean_val) and not np.isinf(mean_val):
                            ax.axvline(mean_val, color=f'C{cluster_id}', linestyle='--', alpha=0.7)
                    except Exception as e:
                        logger.warning(f"绘制聚类 {cluster_id} 的平均值线时出错: {e}")
                
                # 设置标题和图例
                ax.set_title(feature)
                ax.legend()
                
            plt.tight_layout()
            plt.suptitle('各聚类特征分布', fontsize=16)
            plt.subplots_adjust(top=0.9)
            
        except Exception as e:
            logger.error(f"绘制特征分布图时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 创建一个简单的错误提示图
            plt.clf()
            plt.text(0.5, 0.5, f'特征分布图绘制失败: {e}', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('错误')
        
        return fig
        
    def plot_cluster_centroids(self, result, **kwargs):
        """
        绘制簇中心点雷达图
        
        Args:
            result (dict): 聚类结果
            **kwargs: 绘图参数
            
        Returns:
            matplotlib.Figure: 雷达图
        """
        if 'features' not in result or 'cluster_centers' not in result:
            logger.error("聚类结果缺少必要信息")
            return None
            
        # 获取参数
        figsize = kwargs.get('figsize', (10, 8))
        selected_features = kwargs.get('selected_features', result['features'])
        
        # 过滤选择的特征
        valid_features = [f for f in selected_features if f in result['features']]
        
        if not valid_features:
            logger.error("没有有效的特征可供绘制")
            return None
            
        # 创建雷达图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        try:
            # 获取聚类中心点特征数据
            centers = result['cluster_centers']
            n_clusters = result['n_clusters']
            
            # 记录原始数据形状和类型以便调试
            logger.info(f"原始聚类中心数据类型: {type(centers)}")
            if hasattr(centers, 'shape'):
                logger.info(f"原始聚类中心数据形状: {centers.shape}")
            elif isinstance(centers, list):
                logger.info(f"原始聚类中心数据长度: {len(centers)}")
            
            # 确保centers是numpy数组格式
            if isinstance(centers, pd.DataFrame):
                # 如果是DataFrame，转换为numpy数组
                centers = centers.values
                logger.info("将聚类中心从DataFrame转换为numpy数组")
            elif not isinstance(centers, np.ndarray):
                try:
                    # 尝试转换为numpy数组
                    centers = np.array(centers)
                    logger.info(f"将聚类中心类型 {type(centers).__name__} 转换为numpy数组")
                except Exception as e:
                    logger.error(f"无法将聚类中心转换为numpy数组: {e}")
                    plt.text(0.5, 0.5, f"无法处理聚类中心数据: {e}", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    plt.title('雷达图绘制失败')
                    return fig
            
            # 确保数据是二维数组
            if centers.ndim == 1:
                # 如果是一维数组，假设它是一个单一的中心点，添加维度
                centers = centers.reshape(1, -1)
                logger.info(f"聚类中心是一维数组，已重塑为 {centers.shape}")
            elif centers.ndim > 2:
                # 如果维度大于2，尝试将其转换为2维
                original_shape = centers.shape
                centers = centers.reshape(centers.shape[0], -1)
                logger.info(f"聚类中心维度超过2 ({original_shape})，已重塑为 {centers.shape}")
            
            logger.info(f"处理后的聚类中心数据形状: {centers.shape}")
            
            # 创建特征索引映射
            feature_indices = []
            feature_names = []
            
            for f in valid_features:
                try:
                    if f in result['features']:
                        idx = result['features'].index(f)
                        if idx < centers.shape[1]:
                            feature_indices.append(idx)
                            feature_names.append(f)
                        else:
                            logger.warning(f"特征索引 {idx} 超出中心点数据范围 {centers.shape[1]}")
                except Exception as e:
                    logger.warning(f"获取特征 '{f}' 索引时出错: {e}")
            
            logger.info(f"有效特征数量: {len(feature_indices)}, 特征名称: {feature_names}")
            
            if not feature_indices:
                logger.error("没有找到有效特征的索引")
                plt.text(0.5, 0.5, "没有找到有效特征的索引", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                plt.title('雷达图绘制失败')
                return fig
            
            # 提取中心点对应特征的数据
            centers_filtered = None
            try:
                centers_filtered = np.zeros((centers.shape[0], len(feature_indices)))
                for i, idx in enumerate(feature_indices):
                    if idx < centers.shape[1]:
                        centers_filtered[:, i] = centers[:, idx]
                    else:
                        centers_filtered[:, i] = 0.0  # 默认值
                logger.info(f"成功创建聚类中心特征筛选数据，形状: {centers_filtered.shape}")
            except Exception as e:
                logger.error(f"提取聚类中心特征数据时出错: {e}")
                # 尝试一种更安全的方法
                centers_filtered = []
                for center in centers:
                    filtered_center = []
                    for idx in feature_indices:
                        if idx < len(center):
                            filtered_center.append(float(center[idx]))
                        else:
                            filtered_center.append(0.0)  # 默认值
                    centers_filtered.append(filtered_center)
                
                # 转换为numpy数组
                try:
                    centers_filtered = np.array(centers_filtered, dtype=float)
                    logger.info(f"使用备选方法成功创建聚类中心特征筛选数据，形状: {centers_filtered.shape}")
                except Exception as e:
                    logger.error(f"无法将过滤后的中心点数据转换为numpy数组: {e}")
                    plt.text(0.5, 0.5, f"处理中心点数据失败: {e}", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    plt.title('雷达图绘制失败')
                    return fig
            
            if centers_filtered is None or centers_filtered.size == 0:
                logger.error("提取的聚类中心数据为空")
                plt.text(0.5, 0.5, "提取的聚类中心数据为空", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                plt.title('雷达图绘制失败')
                return fig
            
            # 处理中心点数据，归一化到相同尺度
            centers_scaled = None
            try:
                # 检查是否有常量列（所有值相同的列）
                constant_cols = []
                for j in range(centers_filtered.shape[1]):
                    col = centers_filtered[:, j]
                    if np.all(col == col[0]):
                        constant_cols.append(j)
                    
                if len(constant_cols) == centers_filtered.shape[1]:
                    logger.warning("所有特征列都是常量，无法进行标准化")
                    # 使用原始值
                    centers_scaled = centers_filtered
                elif len(constant_cols) > 0:
                    # 有些列是常量
                    logger.warning(f"以下特征列是常量，将跳过标准化: {[feature_names[j] for j in constant_cols]}")
                    # 复制数据进行修改
                    centers_scaled = np.zeros_like(centers_filtered)
                    for j in range(centers_filtered.shape[1]):
                        if j in constant_cols:
                            # 常量列直接复制
                            centers_scaled[:, j] = centers_filtered[:, j]
                        else:
                            # 非常量列标准化
                            col = centers_filtered[:, j]
                            mean = np.mean(col)
                            std = np.std(col)
                            if std > 0:
                                centers_scaled[:, j] = (col - mean) / std
                            else:
                                centers_scaled[:, j] = 0
                else:
                    # 正常标准化所有列
                    scaler = StandardScaler()
                    centers_scaled = scaler.fit_transform(centers_filtered)
                    logger.info(f"所有特征列已成功标准化")
                
                # 检查centers_scaled是否有NaN或Inf
                if np.isnan(centers_scaled).any() or np.isinf(centers_scaled).any():
                    logger.warning("标准化后的数据包含NaN或Inf值，将替换为0")
                    centers_scaled = np.nan_to_num(centers_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                logger.info(f"标准化后的数据形状: {centers_scaled.shape}")
            except Exception as e:
                logger.error(f"中心点数据标准化失败: {e}")
                # 直接使用原始数据
                centers_scaled = centers_filtered
                # 检查数据是否有效
                if np.isnan(centers_scaled).any() or np.isinf(centers_scaled).any():
                    centers_scaled = np.nan_to_num(centers_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"使用未标准化的数据，形状: {centers_scaled.shape}")
            
            # 准备雷达图数据
            angles = np.linspace(0, 2*np.pi, len(feature_names), endpoint=False).tolist()
            logger.info(f"创建角度数组，长度: {len(angles)}")
            
            # 增加雷达图绘制的健壮性
            try:
                # 验证angles数组
                if len(angles) < 2:
                    logger.error(f"特征数量不足，无法创建雷达图，只有 {len(feature_names)} 个特征")
                    plt.text(0.5, 0.5, f"特征数量不足，无法创建雷达图，需要至少2个特征", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    plt.title('雷达图绘制失败')
                    return fig
                
                # 闭合雷达图
                angles += angles[:1]
                logger.info(f"闭合角度数组，最终长度: {len(angles)}")
                
                # 验证centers_scaled
                if centers_scaled is None:
                    logger.error("标准化后的聚类中心数据为None")
                    plt.text(0.5, 0.5, "标准化后的聚类中心数据为None", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    plt.title('雷达图绘制失败')
                    return fig
                
                # 确保至少一个簇可以绘制
                successful_plots = 0
                
                # 计算要绘制的簇数量
                clusters_to_plot = min(n_clusters, centers_scaled.shape[0])
                logger.info(f"将绘制 {clusters_to_plot} 个簇的雷达图")
                
                # 绘制每个聚类的中心点
                for i in range(clusters_to_plot):
                    try:
                        # 检查聚类索引是否有效
                        if i >= centers_scaled.shape[0]:
                            logger.warning(f"聚类 {i} 超出中心点数据范围 ({centers_scaled.shape[0]})，跳过")
                            continue
                        
                        # 获取当前聚类中心的数据
                        values = centers_scaled[i].tolist()
                        
                        # 验证values
                        if not values:
                            logger.warning(f"聚类 {i} 的中心点数据为空，跳过")
                            continue
                        
                        # 检查values是否包含NaN或Inf
                        if any(np.isnan(v) or np.isinf(v) for v in values):
                            logger.warning(f"聚类 {i} 的中心点数据包含NaN或Inf，将替换为0")
                            values = [0 if np.isnan(v) or np.isinf(v) else v for v in values]
                        
                        # 确保values和angles的长度匹配
                        if len(values) != len(angles) - 1:
                            logger.warning(f"聚类 {i}: 值的长度 ({len(values)}) 与角度数组长度 ({len(angles) - 1}) 不匹配")
                            
                            # 调整values的长度以匹配angles
                            if len(values) < len(angles) - 1:
                                # 如果values太短，扩展它
                                logger.info(f"扩展聚类 {i} 的值数组，从 {len(values)} 到 {len(angles) - 1}")
                                values = values + [0.0] * (len(angles) - 1 - len(values))
                            else:
                                # 如果values太长，截断它
                                logger.info(f"截断聚类 {i} 的值数组，从 {len(values)} 到 {len(angles) - 1}")
                                values = values[:len(angles) - 1]
                        
                        # 闭合雷达图
                        values += values[:1]
                        
                        # 记录最终的值数组长度是否正确
                        logger.info(f"聚类 {i} 最终值数组长度: {len(values)}, 角度数组长度: {len(angles)}")
                        if len(values) != len(angles):
                            logger.warning(f"聚类 {i} 的值数组长度 ({len(values)}) 与角度数组长度 ({len(angles)}) 仍不匹配")
                            continue
                        
                        # 绘制雷达线
                        ax.plot(angles, values, 'o-', linewidth=2, label=f'簇 {i}')
                        ax.fill(angles, values, alpha=0.1)
                        successful_plots += 1
                        logger.info(f"成功绘制聚类 {i} 的雷达图")
                        
                    except Exception as inner_e:
                        logger.error(f"绘制聚类 {i} 时出错: {inner_e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        continue
                
                # 检查是否成功绘制了任何聚类
                if successful_plots == 0:
                    logger.error("没有成功绘制任何聚类的雷达图")
                    plt.text(0.5, 0.5, "没有数据可供绘制，所有聚类绘制都失败了", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    plt.title('雷达图绘制失败')
                    return fig
                
                logger.info(f"成功绘制了 {successful_plots} 个聚类的雷达图")
                
                # 设置刻度标签
                try:
                    xticks = angles[:-1]  # 最后一个是为了闭合图形而重复的
                    if len(xticks) == len(feature_names):
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(feature_names)
                    else:
                        logger.warning(f"刻度数量({len(xticks)})与特征名数量({len(feature_names)})不匹配")
                        # 尝试修复
                        if len(feature_names) < len(xticks):
                            # 扩展特征名称列表
                            extended_names = feature_names + [f"特征_{i}" for i in range(len(feature_names), len(xticks))]
                            ax.set_xticks(xticks)
                            ax.set_xticklabels(extended_names)
                        else:
                            # 使用前len(xticks)个特征名称
                            ax.set_xticks(xticks)
                            ax.set_xticklabels(feature_names[:len(xticks)])
                except Exception as e:
                    logger.warning(f"设置刻度标签时出错: {e}")
                    # 不设置标签
                
                # 添加图例和网格
                plt.legend(loc='upper right')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.title('各聚类中心点特征雷达图（标准化值）')
                
            except Exception as e:
                logger.error(f"绘制雷达图内部处理时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 如果已经有成功的绘图，则返回当前图表
                if 'successful_plots' in locals() and successful_plots > 0:
                    logger.info(f"尽管有错误，但已成功绘制了 {successful_plots} 个聚类")
                    return fig
                
                # 否则显示错误信息
                plt.text(0.5, 0.5, f'雷达图绘制过程中出错: {e}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                plt.title('雷达图绘制部分失败')
        
        except Exception as e:
            logger.error(f"绘制雷达图时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 创建一个简单的错误提示图
            plt.text(0.5, 0.5, f'雷达图绘制失败: {e}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            plt.title('错误')
        
        return fig 

    def calculate_cluster_stats(self):
        """
        计算每个聚类的统计信息
        
        Returns:
            dict: 聚类统计信息
        """
        if 'cluster' not in self.data.columns:
            logger.error("数据中没有聚类标签，无法计算聚类统计信息")
            return None
            
        try:
            n_clusters = self.data['cluster'].nunique()
            n_samples = len(self.data)
            symbol_col = StandardColumns.SYMBOL.value
            date_col = StandardColumns.DATE.value
            
            # 计算各聚类的样本数量和占比
            cluster_counts = {}
            for i in range(n_clusters):
                count = (self.data['cluster'] == i).sum()
                percentage = count / n_samples * 100
                cluster_counts[i] = {
                    'count': int(count),
                    'percentage': float(percentage)
                }
            
            # 计算各股票的主要聚类
            symbol_clusters = {}
            if symbol_col in self.data.columns and date_col in self.data.columns:
                symbols = self.data[symbol_col].unique()
                
                for symbol in symbols:
                    symbol_data = self.data[self.data[symbol_col] == symbol]
                    symbol_labels = symbol_data['cluster']
                    
                    if len(symbol_labels) == 0:
                        continue
                    
                    # 计算该股票各聚类的占比
                    symbol_counts = {}
                    for i in range(n_clusters):
                        count = (symbol_labels == i).sum()
                        percentage = count / len(symbol_labels) * 100
                        symbol_counts[i] = {
                            'count': int(count),
                            'percentage': float(percentage)
                        }
                    
                    # 找出主要聚类（占比最高的）
                    main_cluster = max(range(n_clusters), key=lambda x: symbol_counts[x]['percentage'])
                    
                    symbol_clusters[symbol] = {
                        'main_cluster': int(main_cluster),
                        'cluster_distribution': symbol_counts
                    }
            
            # 计算每个特征的重要性
            feature_importance = {}
            if n_clusters > 1 and self.features is not None and self.kmeans_model is not None:
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # 使用随机森林评估特征重要性
                    X = self.data[self.features].copy()
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X, self.data['cluster'])
                    
                    for i, feature in enumerate(self.features):
                        feature_importance[feature] = float(rf.feature_importances_[i])
                except Exception as e:
                    logger.warning(f"计算特征重要性时出错: {e}")
                    # 简单方法：根据聚类中心点的方差评估重要性
                    for i, feature in enumerate(self.features):
                        feature_importance[feature] = float(np.var([center[i] for center in self.kmeans_model.cluster_centers_]))
            else:
                if self.features:
                    for feature in self.features:
                        feature_importance[feature] = 1.0 / len(self.features)
            
            # 计算每个聚类的统计特征
            cluster_features = {}
            for i in range(n_clusters):
                cluster_data = self.data[self.data['cluster'] == i]
                if self.features:
                    feature_stats = {}
                    for feature in self.features:
                        if feature in cluster_data.columns:
                            feature_stats[feature] = {
                                'mean': float(cluster_data[feature].mean()),
                                'median': float(cluster_data[feature].median()),
                                'std': float(cluster_data[feature].std()),
                                'min': float(cluster_data[feature].min()),
                                'max': float(cluster_data[feature].max())
                            }
                    cluster_features[i] = feature_stats
            
            # 返回统计信息
            stats = {
                'n_clusters': n_clusters,
                'cluster_counts': cluster_counts,
                'symbol_clusters': symbol_clusters,
                'feature_importance': feature_importance,
                'cluster_features': cluster_features
            }
            
            if self.kmeans_model is not None:
                stats['inertia'] = float(self.kmeans_model.inertia_)
            
            # 记录统计信息
            logger.info(f"聚类统计信息计算完成，共 {n_clusters} 个聚类")
            for i in range(n_clusters):
                logger.info(f"聚类 #{i}: {cluster_counts[i]['count']} 个样本 ({cluster_counts[i]['percentage']:.2f}%)")
            
            return stats
            
        except Exception as e:
            logger.error(f"计算聚类统计信息失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None 