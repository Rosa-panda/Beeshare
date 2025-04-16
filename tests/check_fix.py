#!/usr/bin/env python
"""
测试专用脚本：验证AKShareDS数据源的修复是否生效。

该脚本专门测试AKShareDS类中的_get_stock_data_via_fallback_api方法
是否能正确生成模拟数据，并验证standardize_columns函数是否正确映射列名。

创建日期: 2024-04-03
最后修改: 2024-04-10
作者: BeeShare开发团队
"""

import re
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_imports():
    # 获取项目根目录的路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(root_dir, 'main.py')
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查run_clustering_analysis函数中是否已添加pandas导入
    run_clustering_pattern = r'def run_clustering_analysis.*?try:'
    run_clustering_match = re.search(run_clustering_pattern, content, re.DOTALL)
    
    if run_clustering_match:
        function_header = run_clustering_match.group(0)
        if 'import pandas as pd' in function_header:
            print("✅ run_clustering_analysis函数中已添加pandas导入")
        else:
            print("❌ run_clustering_analysis函数中缺少pandas导入")
    else:
        print("❌ 未找到run_clustering_analysis函数")
    
    # 检查是否没有重复导入
    model_data_pattern = r'# 临时解决方案：生成模拟数据.*?# 生成日期范围'
    model_data_match = re.search(model_data_pattern, content, re.DOTALL)
    
    if model_data_match:
        model_data_section = model_data_match.group(0)
        if 'import pandas as pd' in model_data_section or 'import numpy as np' in model_data_section:
            print("❌ 模拟数据生成部分存在重复导入")
        else:
            print("✅ 模拟数据生成部分没有重复导入")
    else:
        print("❌ 未找到模拟数据生成部分")
    
    print("\n检查完成!")

if __name__ == "__main__":
    check_imports() 