"""
配置文件，包含系统的各种配置参数
"""

# 数据源配置
DATA_SOURCES = {
    # 雅虎财经配置
    'yahoo': {
        'enabled': False,  # 禁用Yahoo Finance数据源
        'api_key': None,  # 不需要API密钥
        'timeout': 10,  # 请求超时时间(秒)
    },
    # 东方财富配置
    'eastmoney': {
        'enabled': False,  # 禁用东方财富数据源
        'api_key': None,  # 某些API可能需要密钥
        'timeout': 10,
    },
    # AKShare配置
    'akshare': {
        'enabled': True,
        'api_key': None,  # 大多数接口不需要API密钥
        'timeout': 30,     # 增加超时时间
        'retry_count': 3,  # 添加重试次数
        'retry_delay': 2,  # 添加重试延迟
        'testing': True,   # 开启测试模式（失败时使用模拟数据）
        'mock_data_seed': 42,  # 模拟数据随机种子
    }
}

# 数据获取配置
FETCH_CONFIG = {
    # 历史数据配置
    'historical': {
        'default_start_date': '2020-01-01',  # 默认开始日期
        'default_end_date': None,  # None表示当前日期
        'default_interval': '1d',  # 默认时间间隔，1d表示日线数据
    },
    # 实时数据配置
    'realtime': {
        'update_interval': 60,  # 更新频率(秒)
        'max_symbols_per_request': 100,  # 每次请求最大股票数量
    }
}

# 存储配置
STORAGE_CONFIG = {
    # CSV存储配置
    'csv': {
        'enabled': True,
        # 使用按数据类型划分的目录结构
        'base_path': 'data/stock_data',  # 相对于项目根目录的路径
        # 简化的文件组织结构
        'file_format': '{data_type}/{symbol}.csv',  # 按数据类型分目录存储
        'use_date_in_filename': False,  # 不在文件名中包含日期
    },
    # SQLite存储配置
    'sqlite': {
        'enabled': True,
        'db_path': 'data/database/stockdata.db',
        'tables': {
            'historical_data': 'historical_prices',
            'real_time_data': 'realtime_prices',
            'symbols': 'stock_symbols',
        }
    }
}

# 目标市场配置
MARKETS = {
    'us': {  # 美股
        'enabled': False,  # 禁用美股
        'index_symbol': '^GSPC',  # 标普500指数
        'default_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    },
    'cn': {  # A股
        'enabled': True,
        'index_symbol': '000001',  # 上证指数
        'default_symbols': ['600036', '601398', '600276', '600519', '601288'],  # 移除.SS后缀
    },
    'hk': {  # 港股
        'enabled': False,  # 禁用港股
        'index_symbol': '^HSI',  # 恒生指数
        'default_symbols': ['00700.HK', '01398.HK', '00941.HK', '03690.HK', '01810.HK'],
    }
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/stock_data.log',  # 相对于项目根目录
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# 应用配置
APP_CONFIG = {
    'name': '股票数据获取系统',
    'version': '0.1.0',
    'author': 'Beeshare Team',
} 

# 整合所有配置到CONFIG变量
CONFIG = {
    'DATA_SOURCES': DATA_SOURCES,
    'FETCH_CONFIG': FETCH_CONFIG,
    'STORAGE_CONFIG': STORAGE_CONFIG,
    'MARKETS': MARKETS,
    'LOG_CONFIG': LOG_CONFIG,
    'APP_CONFIG': APP_CONFIG
} 