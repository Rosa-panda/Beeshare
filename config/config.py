#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BeeShare系统配置文件
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

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
        'api_base_url': "",  # 默认为空，使用官方地址
        'proxy': None
    }
}

# 数据获取配置
FETCH_CONFIG = {
    "retry_interval": 5,  # 重试间隔（秒）
    "default_start_date": "2022-01-01",  # 默认开始日期
    "max_retry_count": 5,  # 最大重试次数
    "concurrent_requests": 3,  # 并发请求数
    "default_interval": "daily",  # 默认数据间隔
    "rate_limit": {
        "enabled": True,
        "max_requests": 60,  # 每分钟最大请求数
        "time_window": 60  # 时间窗口（秒）
    }
}

# 存储配置 - TimescaleDB
STORAGE_CONFIG = {
    "type": "postgresql",
    "connection": {
        "host": "localhost",
        "port": 5432,
        "database": "beeshare_db",
        "user": "beeshare",
        "password": "beeshare123"
    },
    "pool_size": 5,
    "max_overflow": 10,
    "pool_recycle": 3600,
    "echo": False
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
        'default_symbols': [
            "600519",  # 贵州茅台
            "601398",  # 工商银行
            "600036",  # 招商银行
            "000001",  # 平安银行
            "000858",  # 五粮液
            "600276",  # 恒瑞医药
            "000333",  # 美的集团
            "601318",  # 中国平安
            "600887",  # 伊利股份
            "601288"   # 农业银行
        ],
        'indices': {
            "000001": "上证指数",
            "399001": "深证成指",
            "000300": "沪深300",
            "000016": "上证50",
            "000905": "中证500",
            "399006": "创业板指"
        }
    },
    'hk': {  # 港股
        'enabled': False,  # 禁用港股
        'index_symbol': '^HSI',  # 恒生指数
        'default_symbols': ['00700.HK', '01398.HK', '00941.HK', '03690.HK', '01810.HK'],
    }
}

# 日志配置
LOG_CONFIG = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_console": True,
    "log_to_file": True,
    "log_file_dir": os.path.join(ROOT_DIR, "logs"),
    "log_file_name": "beeshare.log",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_date_format": "%Y-%m-%d %H:%M:%S",
    "max_log_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 10,
    "color_output": True
}

# 分析配置
ANALYSIS_CONFIG = {
    "technical": {
        "default_indicators": ["ma", "ema", "macd", "rsi", "boll"],
        "default_periods": {
            "ma": [5, 10, 20, 60],
            "ema": [5, 10, 20, 60],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "rsi": [14],
            "boll": {"period": 20, "std_dev": 2}
        }
    },
    "clustering": {
        "default_n_clusters": 3,
        "default_features": ["open", "close", "high", "low", "volume", "change_percent"],
        "default_max_clusters": 10,
        "default_random_state": 42
    }
}

# 输出目录配置
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

# 时间配置
TIME_CONFIG = {
    "ntp_servers": ["pool.ntp.org", "time.windows.com", "time.apple.com"],
    "time_zone": "Asia/Shanghai",
    "update_frequency": 3600  # 1小时更新一次
}

# 在开发环境中，您可以创建一个本地配置文件来覆盖上述设置
# 如果存在local_config.py文件，将导入其中的设置
try:
    from .local_config import *
except ImportError:
    pass

# 整合所有配置到CONFIG变量
CONFIG = {
    'DATA_SOURCES': DATA_SOURCES,
    'FETCH_CONFIG': FETCH_CONFIG,
    'STORAGE_CONFIG': STORAGE_CONFIG,
    'MARKETS': MARKETS,
    'LOG_CONFIG': LOG_CONFIG,
    'ANALYSIS_CONFIG': ANALYSIS_CONFIG,
    'OUTPUT_DIR': OUTPUT_DIR,
    'TIME_CONFIG': TIME_CONFIG
} 