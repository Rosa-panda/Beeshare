#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文档日期更新工具的示例脚本。

该脚本演示了如何使用BeeShare系统中的文档日期更新工具来更新
Python文件的文档字符串中的日期信息。该工具可以对指定目录下的
所有Python文件进行批量处理，更新创建日期和最后修改日期。

创建日期: 2024-04-15
最后修改: 2024-04-15
作者: BeeShare开发团队
"""

import os
import sys
import tempfile
import shutil

# 添加项目根目录到系统路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, script_dir)

# 导入时间工具
from src.utils.time_utils import get_network_time, get_formatted_date

def create_sample_python_file(directory, filename):
    """
    创建一个示例Python文件，用于演示日期更新
    
    Args:
        directory (str): 目录路径
        filename (str): 文件名
        
    Returns:
        str: 文件的完整路径
    """
    file_path = os.path.join(directory, filename)
    
    content = '''"""
这是一个示例Python文件，用于演示日期更新工具的功能。

创建日期: 2021-01-01
最后修改: 2021-01-02
"""

def hello_world():
    """打印Hello World"""
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
'''
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return file_path

def print_file_content(file_path):
    """打印文件内容"""
    print(f"\n文件 '{os.path.basename(file_path)}' 的内容:")
    print("-" * 40)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(content)
    print("-" * 40)

def update_file_dates(file_path, update_creation=False, update_modified=True, dry_run=False):
    """
    更新文件的日期信息
    
    Args:
        file_path (str): 文件路径
        update_creation (bool): 是否更新创建日期
        update_modified (bool): 是否更新最后修改日期
        dry_run (bool): 是否仅模拟运行
        
    Returns:
        bool: 更新是否成功
    """
    # 导入更新模块（在函数内导入，因为这是示例用途）
    try:
        from tools.update_docstring_dates import update_docstring_dates
        
        # 调用更新函数
        return update_docstring_dates(
            file_path, 
            update_creation=update_creation,
            update_modified=update_modified,
            dry_run=dry_run
        )
    except ImportError:
        print("警告: 无法导入update_docstring_dates模块，将使用简化版实现")
        
        # 如果无法导入原模块，使用简化版更新实现
        import re
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 获取当前日期（使用网络时间）
        current_date = get_formatted_date()
        
        # 更新创建日期
        if update_creation:
            content = re.sub(r'(创建日期:\s*)[\d-]+', r'\g<1>' + current_date, content)
        
        # 更新最后修改日期
        if update_modified:
            content = re.sub(r'(最后修改:\s*)[\d-]+', r'\g<1>' + current_date, content)
        
        # 如果不是演习模式，写回文件
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return True

def main():
    """主函数，演示日期更新工具的使用方法"""
    print("===== BeeShare文档日期更新工具示例 =====\n")
    
    # 创建临时目录
    tmp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {tmp_dir}")
    
    try:
        # 创建示例文件
        file1 = create_sample_python_file(tmp_dir, "example1.py")
        file2 = create_sample_python_file(tmp_dir, "example2.py")
        
        # 显示原始文件内容
        print("\n=== 原始文件内容 ===")
        print_file_content(file1)
        
        # 示例1: 只更新最后修改日期
        print("\n示例1: 只更新最后修改日期")
        update_file_dates(file1, update_creation=False, update_modified=True)
        print_file_content(file1)
        
        # 示例2: 同时更新创建日期和最后修改日期
        print("\n示例2: 同时更新创建日期和最后修改日期")
        update_file_dates(file2, update_creation=True, update_modified=True)
        print_file_content(file2)
        
        # 示例3: 演习运行模式
        print("\n示例3: 演习运行模式（不实际修改文件）")
        print("注意: 下面的更新仅模拟执行，不会实际修改文件")
        
        # 再次创建一个示例文件
        file3 = create_sample_python_file(tmp_dir, "example3.py")
        print_file_content(file3)
        
        # 演习运行，不会实际修改文件
        update_file_dates(file3, update_creation=True, update_modified=True, dry_run=True)
        print("\n演习运行后，文件内容没有变化:")
        print_file_content(file3)
        
        print("\n=== 实际项目使用建议 ===")
        print("1. 在提交代码前，运行更新工具，确保文件包含正确的日期")
        print("2. 可以创建Git钩子，在提交前自动运行更新工具")
        print("3. 在团队中统一使用这个工具，保持日期格式一致")
        
    finally:
        # 清理临时目录
        print(f"\n清理临时目录: {tmp_dir}")
        shutil.rmtree(tmp_dir)
    
    print("\n===== 示例结束 =====")
    print("要在实际项目中使用完整功能，请运行:")
    print("python tools/update_docstring_dates.py --help")

if __name__ == "__main__":
    main() 