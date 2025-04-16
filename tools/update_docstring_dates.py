#!/usr/bin/env python
"""
文档字符串日期更新工具。

该脚本用于扫描项目中所有Python文件，找出包含创建日期和最后修改日期的
文档字符串，并将其更新为当前日期。支持批量处理和选择性更新。

创建日期: 2023-04-15
最后修改: 2024-04-15
作者: BeeShare开发团队
"""

import os
import re
import sys
import argparse
from datetime import datetime
import importlib.util

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入time_utils模块
sys.path.insert(0, project_root)
try:
    from src.utils.time_utils import get_formatted_date
except ImportError:
    # 如果导入失败，使用当前系统时间
    def get_formatted_date(date_format='%Y-%m-%d'):
        """
        获取格式化的当前日期字符串（备用实现）。
        
        当无法导入time_utils模块时使用的备用函数。
        
        Args:
            date_format (str): 日期格式字符串，默认为'%Y-%m-%d'
            
        Returns:
            str: 格式化的日期字符串
        """
        return datetime.now().strftime(date_format)


def update_docstring_dates(file_path, update_creation=False, update_modified=True, dry_run=False):
    """
    更新Python文件中的文档字符串日期信息。
    
    Args:
        file_path (str): 要处理的Python文件路径
        update_creation (bool): 是否更新创建日期，默认为False
        update_modified (bool): 是否更新最后修改日期，默认为True
        dry_run (bool): 是否仅模拟执行而不实际修改文件，默认为False
        
    Returns:
        bool: 是否进行了更新
        
    Raises:
        IOError: 当文件无法读取或写入时抛出
        Exception: 处理文件时可能发生的其他异常
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 获取当前日期
        current_date = get_formatted_date()
        
        # 文件是否被修改的标志
        modified = False
        
        # 匹配并更新创建日期
        if update_creation:
            creation_pattern = r'(创建日期:\s*)(\d{4}-\d{2}-\d{2}|\d{4}年\d{2}月\d{2}日)'
            if re.search(creation_pattern, content):
                new_content = re.sub(creation_pattern, r'\g<1>' + current_date, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    print(f"已更新创建日期: {file_path}")
        
        # 匹配并更新最后修改日期
        if update_modified:
            modified_pattern = r'(最后修改:\s*)(\d{4}-\d{2}-\d{2}|\d{4}年\d{2}月\d{2}日)'
            if re.search(modified_pattern, content):
                new_content = re.sub(modified_pattern, r'\g<1>' + current_date, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    print(f"已更新最后修改日期: {file_path}")
        
        # 如果不是演习模式且文件有修改，写回文件
        if modified and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return modified
    except Exception as e:
        print(f"处理文件时出错 {file_path}: {e}")
        raise

def scan_project(directory=None, extensions=None, update_creation=False, update_modified=True, dry_run=False):
    """
    扫描项目目录，更新所有Python文件的文档字符串日期。
    
    Args:
        directory (str, optional): 要扫描的目录，默认为项目根目录
        extensions (list, optional): 要处理的文件扩展名列表，默认为['.py']
        update_creation (bool): 是否更新创建日期，默认为False
        update_modified (bool): 是否更新最后修改日期，默认为True
        dry_run (bool): 是否仅模拟执行而不实际修改文件，默认为False
        
    Returns:
        tuple: (处理的文件数, 更新的文件数)
        
    Raises:
        OSError: 当无法访问指定目录时抛出
    """
    if directory is None:
        directory = project_root
        
    if extensions is None:
        extensions = ['.py']
    
    processed_count = 0
    updated_count = 0
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        # 跳过隐藏目录和虚拟环境
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '__pycache__']
        
        for file in files:
            # 检查文件扩展名
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                processed_count += 1
                
                try:
                    updated = update_docstring_dates(
                        file_path, 
                        update_creation=update_creation,
                        update_modified=update_modified,
                        dry_run=dry_run
                    )
                    if updated:
                        updated_count += 1
                except Exception as e:
                    print(f"处理文件时出错 {file_path}: {e}")
    
    return processed_count, updated_count

def main():
    """
    主函数，处理命令行参数并执行文档字符串日期更新。
    
    处理用户提供的命令行参数，调用相应函数扫描项目并更新Python文件中的日期信息。
    
    Returns:
        int: 程序退出状态码，0表示正常退出
    """
    parser = argparse.ArgumentParser(description='更新Python文件的文档字符串日期信息')
    parser.add_argument('--dir', type=str, help='要扫描的目录，默认为项目根目录')
    parser.add_argument('--ext', type=str, nargs='+', default=['.py'], help='要处理的文件扩展名列表')
    parser.add_argument('--creation', action='store_true', help='更新创建日期')
    parser.add_argument('--no-modified', action='store_true', help='不更新最后修改日期')
    parser.add_argument('--dry-run', action='store_true', help='仅模拟执行而不实际修改文件')
    args = parser.parse_args()
    
    print("开始扫描项目文件...")
    processed, updated = scan_project(
        directory=args.dir,
        extensions=args.ext,
        update_creation=args.creation,
        update_modified=not args.no_modified,
        dry_run=args.dry_run
    )
    
    print(f"\n扫描完成！处理了 {processed} 个文件，{updated} 个文件已更新")
    if args.dry_run:
        print("注意：这是一次演习运行，没有实际修改任何文件")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 