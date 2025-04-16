"""
兼容性导入文件

为了保持向后兼容性，从akshare.py导入AKShareDS类
"""

from .akshare import AKShareDS

# 导出AKShareDS类
__all__ = ['AKShareDS'] 