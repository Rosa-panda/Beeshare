#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文档日期更新工具。

该脚本用于自动更新项目中所有Python文件的文档字符串中的创建日期和最后修改日期，
确保使用准确的网络时间，以规范化项目文档。

创建日期: 2024-04-15
最后修改: 2024-04-15
作者: BeeShare开发团队
"""

import os
import re
import sys
import glob
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加src目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    # 导入网络时间工具
    from src.utils.time_utils import get_formatted_date
except ImportError:
    logger.error("无法导入time_utils模块，请确保脚本在项目根目录下运行")
    sys.exit(1)

def update_docstring_dates(file_path):
    """
    更新单个文件中的文档字符串日期。
    
    查找并更新指定Python文件中文档字符串的创建日期和最后修改日期，
    使用网络时间作为准确的日期来源。
    
    Args:
        file_path (str): 要更新的文件路径
    
    Returns:
        bool: 是否成功更新文件
        
    Raises:
        IOError: 当文件无法读取或写入时抛出
        Exception: 处理文件时可能发生的其他异常
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有文档字符串
        if '"""' not in content:
            logger.info(f"跳过 {file_path} - 没有文档字符串")
            return False
        
        # 获取当前日期
        current_date = get_formatted_date()
        
        # 更新创建日期
        if '创建日期:' in content:
            content = re.sub(r'创建日期:\s*[\d-]+', f'创建日期: {current_date}', content)
        elif '创建日期' in content:
            # 添加创建日期到文档字符串末尾
            docstring_end = content.find('"""', content.find('"""') + 3)
            if docstring_end != -1:
                content = content[:docstring_end] + f'\n创建日期: {current_date}\n' + content[docstring_end:]
        
        # 更新最后修改日期
        if '最后修改:' in content:
            content = re.sub(r'最后修改:\s*[\d-]+', f'最后修改: {current_date}', content)
        elif '最后修改' in content:
            # 如果没有最后修改日期但有创建日期，添加最后修改日期
            creation_date_pos = content.find('创建日期:')
            if creation_date_pos != -1:
                next_line_pos = content.find('\n', creation_date_pos)
                if next_line_pos != -1:
                    content = content[:next_line_pos] + f'\n最后修改: {current_date}' + content[next_line_pos:]
        
        # 保存更新后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"已更新 {file_path} 的日期信息")
        return True
    
    except Exception as e:
        logger.error(f"更新 {file_path} 失败: {e}")
        return False

def find_python_files(directory):
    """
    在指定目录及其子目录中查找所有Python文件。
    
    使用glob模块递归搜索指定目录中的所有.py文件。
    
    Args:
        directory (str): 起始目录路径
    
    Returns:
        list: Python文件路径列表
        
    Raises:
        OSError: 当指定目录不存在或无法访问时抛出
    """
    return glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)

def main():
    """
    主函数，执行项目文件日期更新。
    
    确定项目根目录，查找所有Python文件，并更新它们的文档字符串日期信息。
    
    Returns:
        int: 程序退出状态码，0表示正常退出
    """
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    logger.info(f"开始更新项目 {project_root} 中的Python文件日期信息")
    
    # 查找所有Python文件
    python_files = find_python_files(project_root)
    logger.info(f"找到 {len(python_files)} 个Python文件")
    
    # 更新每个文件
    updated_count = 0
    for file_path in python_files:
        if update_docstring_dates(file_path):
            updated_count += 1
    
    logger.info(f"完成日期更新。共处理 {len(python_files)} 个文件，成功更新 {updated_count} 个文件")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 