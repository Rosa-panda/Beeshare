#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日期计算工具的示例脚本。

该脚本演示了如何使用BeeShare系统中的日期计算工具来获取过去的日期，
并进行各种日期操作。适合初学者学习如何在项目中使用日期和时间。

创建日期: 2024-04-15
最后修改: 2024-04-15
作者: BeeShare开发团队
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, script_dir)

# 导入日期计算工具
from src.utils.date_utils import get_date_n_days_ago

def print_section(title):
    """打印带有分隔线的段落标题"""
    print(f"\n{title}")
    print("-" * 40)

def main():
    """主函数，演示日期计算工具的使用方法"""
    print("===== BeeShare日期计算工具示例 =====\n")
    
    # 示例1: 获取n天前的日期
    print_section("示例1: 获取n天前的日期")
    
    # 默认值（90天前）
    ninety_days_ago = get_date_n_days_ago()
    print(f"90天前的日期: {ninety_days_ago}")
    
    # 自定义天数
    days_list = [1, 7, 30, 180, 365]
    for days in days_list:
        date_str = get_date_n_days_ago(n=days)
        print(f"{days}天前: {date_str}")
    
    # 示例2: 使用不同的日期格式
    print_section("示例2: 使用不同的日期格式")
    
    formats = {
        "标准格式 (YYYY-MM-DD)": '%Y-%m-%d',
        "无分隔符": '%Y%m%d',
        "斜杠分隔": '%Y/%m/%d',
        "中文格式": '%Y年%m月%d日',
        "月份名称": '%Y年%B%d日',  # %B表示月份的完整名称
    }
    
    for name, fmt in formats.items():
        date_str = get_date_n_days_ago(n=30, date_format=fmt)
        print(f"{name}: {date_str}")
    
    # 示例3: 结合日期计算与日期解析
    print_section("示例3: 使用Python标准库进行更多日期操作")
    
    # 获取今天和30天前的日期
    today = datetime.now()
    thirty_days_ago_str = get_date_n_days_ago(n=30)
    
    # 将字符串转回datetime对象（日期解析）
    thirty_days_ago = datetime.strptime(thirty_days_ago_str, '%Y-%m-%d')
    
    # 计算两个日期之间的差异
    date_diff = today - thirty_days_ago
    print(f"从{thirty_days_ago_str}到今天已经过去了{date_diff.days}天")
    
    # 示例4: 日期区间生成
    print_section("示例4: 生成日期区间")
    
    start_date = datetime.strptime(get_date_n_days_ago(n=10), '%Y-%m-%d')
    end_date = datetime.now()
    
    print(f"生成从{start_date.strftime('%Y-%m-%d')}到{end_date.strftime('%Y-%m-%d')}的日期列表:")
    
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    for date_str in date_list:
        print(f"  - {date_str}")
    
    print(f"总共生成了{len(date_list)}个日期")
    
    # 示例5: 日期比较
    print_section("示例5: 日期比较操作")
    
    date1_str = get_date_n_days_ago(n=5)
    date2_str = get_date_n_days_ago(n=10)
    
    date1 = datetime.strptime(date1_str, '%Y-%m-%d')
    date2 = datetime.strptime(date2_str, '%Y-%m-%d')
    
    print(f"日期1: {date1_str}")
    print(f"日期2: {date2_str}")
    
    if date1 > date2:
        print(f"{date1_str} 晚于 {date2_str}")
    else:
        print(f"{date1_str} 早于 {date2_str}")
    
    print(f"两个日期相差 {abs((date1 - date2).days)} 天")
    
    print("\n===== 示例结束 =====")

if __name__ == "__main__":
    main() 