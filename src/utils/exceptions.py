"""
自定义异常模块

定义系统中使用的各种自定义异常类，以实现更精确的错误处理
"""

class BeeShareException(Exception):
    """BeeShare系统的基础异常类"""
    def __init__(self, message="BeeShare系统错误", original_error=None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)
        
    def __str__(self):
        if self.original_error:
            return f"{self.message} (原始错误: {self.original_error})"
        return self.message

# 数据源相关异常
class DataSourceException(BeeShareException):
    """数据源相关异常的基类"""
    def __init__(self, message="数据源错误", source=None, original_error=None):
        self.source = source
        source_info = f" [{source}]" if source else ""
        super().__init__(f"数据源错误{source_info}: {message}", original_error)

class DataFetchException(DataSourceException):
    """获取数据时发生的异常"""
    def __init__(self, message="获取数据失败", source=None, symbol=None, original_error=None):
        self.symbol = symbol
        symbol_info = f" (股票代码: {symbol})" if symbol else ""
        super().__init__(f"获取数据失败{symbol_info}: {message}", source, original_error)

class InvalidSymbolException(DataSourceException):
    """无效的股票代码异常"""
    def __init__(self, symbol, source=None, message=None, original_error=None):
        self.symbol = symbol
        default_message = f"无效的股票代码: {symbol}"
        if message:
            default_message += f" - {message}"
        super().__init__(default_message, source, original_error)

# 存储相关异常
class StorageException(BeeShareException):
    """存储相关异常的基类"""
    def __init__(self, message="存储错误", storage_type=None, original_error=None):
        self.storage_type = storage_type
        storage_info = f" [{storage_type}]" if storage_type else ""
        super().__init__(f"存储错误{storage_info}: {message}", original_error)

class DataSaveException(StorageException):
    """保存数据时发生的异常"""
    def __init__(self, message="保存数据失败", storage_type=None, data_type=None, symbol=None, original_error=None):
        self.data_type = data_type
        self.symbol = symbol
        data_info = f" ({data_type.value if data_type else '未知类型'}"
        data_info += f", 股票代码: {symbol}" if symbol else ""
        data_info += ")"
        super().__init__(f"保存数据失败{data_info}: {message}", storage_type, original_error)

class DataLoadException(StorageException):
    """加载数据时发生的异常"""
    def __init__(self, message="加载数据失败", storage_type=None, data_type=None, symbol=None, original_error=None):
        self.data_type = data_type
        self.symbol = symbol
        data_info = f" ({data_type.value if data_type else '未知类型'}"
        data_info += f", 股票代码: {symbol}" if symbol else ""
        data_info += ")"
        super().__init__(f"加载数据失败{data_info}: {message}", storage_type, original_error)

# 分析相关异常
class AnalysisException(BeeShareException):
    """分析相关异常的基类"""
    def __init__(self, message="分析错误", analysis_type=None, original_error=None):
        self.analysis_type = analysis_type
        analysis_info = f" [{analysis_type}]" if analysis_type else ""
        super().__init__(f"分析错误{analysis_info}: {message}", original_error)

class ClusteringException(AnalysisException):
    """聚类分析异常"""
    def __init__(self, message="聚类分析错误", original_error=None):
        super().__init__(message, "聚类分析", original_error)

class InsufficientDataException(AnalysisException):
    """数据不足异常"""
    def __init__(self, message="数据不足，无法进行分析", analysis_type=None, original_error=None):
        super().__init__(message, analysis_type, original_error)

# 配置相关异常
class ConfigException(BeeShareException):
    """配置相关异常"""
    def __init__(self, message="配置错误", config_type=None, original_error=None):
        self.config_type = config_type
        config_info = f" [{config_type}]" if config_type else ""
        super().__init__(f"配置错误{config_info}: {message}", original_error) 