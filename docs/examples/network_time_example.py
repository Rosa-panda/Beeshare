#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络时间工具的示例脚本。

该脚本演示了如何使用BeeShare系统中的网络时间工具获取准确的网络时间
并生成不同格式的日期字符串。

创建日期: 2024-04-15
最后修改: 2024-04-15
作者: BeeShare开发团队
"""

import os
import sys
import time

# 添加项目根目录到系统路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, script_dir)

# 导入网络时间工具
from src.utils.time_utils import get_network_time, get_formatted_date

def main():
    """主函数，演示网络时间工具的使用方法"""
    print("===== BeeShare网络时间工具示例 =====\n")
    
    # 示例1: 获取当前网络时间
    print("示例1: 获取当前网络时间")
    print("-" * 40)
    
    # 获取网络时间，这是一个datetime对象
    current_time = get_network_time()
    print(f"当前网络时间: {current_time}")
    
    # 直接使用datetime对象的方法格式化
    print(f"格式化输出: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"年份: {current_time.year}")
    print(f"月份: {current_time.month}")
    print(f"日期: {current_time.day}")
    print(f"时间: {current_time.hour}:{current_time.minute}:{current_time.second}")
    print()
    
    # 示例2: 使用get_formatted_date函数获取格式化的日期字符串
    print("示例2: 获取格式化的日期字符串")
    print("-" * 40)
    
    # 默认格式 (%Y-%m-%d)
    today = get_formatted_date()
    print(f"默认格式 (YYYY-MM-DD): {today}")
    
    # 自定义格式
    formats = {
        "年-月-日": '%Y-%m-%d',
        "月/日/年": '%m/%d/%Y',
        "日.月.年": '%d.%m.%Y',
        "中文格式": '%Y年%m月%d日',
        "带时间": '%Y-%m-%d %H:%M:%S',
        "ISO 8601": '%Y-%m-%dT%H:%M:%S',
        "仅时间": '%H:%M:%S',
    }
    
    for name, fmt in formats.items():
        date_str = get_formatted_date(fmt)
        print(f"{name}: {date_str}")
    
    print()
    
    # 示例3: 网络时间响应测试
    print("示例3: 网络时间响应测试")
    print("-" * 40)
    
    print("获取3次网络时间，计算平均响应时间...")
    
    total_time = 0
    for i in range(3):
        start = time.time()
        _ = get_network_time()
        end = time.time()
        elapsed = end - start
        total_time += elapsed
        print(f"请求 {i+1}: {elapsed:.3f} 秒")
    
    print(f"平均响应时间: {total_time/3:.3f} 秒")
    print()
    
    print("===== 示例结束 =====")

if __name__ == "__main__":
    main() 