#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用于更新main.py中的manage_storage函数的脚本
使其支持PostgreSQL/TimescaleDB数据库的状态查询
"""

import os
import re
import sys
import shutil
from datetime import datetime

# 颜色代码
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_color(message, color):
    """打印彩色信息"""
    colors = {
        "blue": Colors.BLUE,
        "green": Colors.GREEN,
        "yellow": Colors.YELLOW,
        "red": Colors.RED,
        "bold": Colors.BOLD,
        "underline": Colors.UNDERLINE
    }
    
    print(f"{colors.get(color.lower(), '')}{message}{Colors.END}")

def backup_file(file_path):
    """备份文件"""
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = f"{file_path}.bak.{timestamp}"
        shutil.copy2(file_path, backup_path)
        print_color(f"已备份文件: {file_path} -> {backup_path}", "blue")
        return backup_path
    return None

def update_manage_storage_function():
    """更新main.py中的manage_storage函数"""
    main_py_path = os.path.join(os.getcwd(), "main.py")
    if not os.path.exists(main_py_path):
        print_color(f"找不到main.py文件: {main_py_path}", "red")
        return False
    
    # 备份main.py文件
    backup_file(main_py_path)
    
    # 读取main.py内容
    with open(main_py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找manage_storage函数
    manage_storage_pattern = r"def\s+manage_storage\s*\(args\)[\s\S]*?(?=def\s+|if\s+__name__\s*==\s*\"__main__\")"
    match = re.search(manage_storage_pattern, content)
    
    if not match:
        print_color(f"未找到manage_storage函数", "red")
        return False
    
    old_manage_storage = match.group(0)
    
    # 创建更新后的manage_storage函数
    new_manage_storage = '''def manage_storage(args):
    """管理存储功能
    
    Args:
        args: 命令行参数
    """
    storage_instances = init_storage()
    
    # 确定使用的存储方式
    from src.storage.config import StorageConfig
    storage_config = StorageConfig()
    active_storage_type = storage_config.get_storage_type()
    storage = storage_instances.get(active_storage_type)
    
    if not storage:
        logger.error(f"存储类型 {active_storage_type} 不可用")
        return False
    
    if args.status:
        try:
            logger.info("获取存储状态信息...")
            
            # 使用存储对象的get_status方法获取状态
            if hasattr(storage, 'get_status'):
                status = storage.get_status()
                
                print("\\n====== 存储状态信息 ======")
                print(f"存储类型: {status.get('type', active_storage_type)}")
                
                # 显示连接信息
                if 'connection' in status:
                    conn_info = status['connection']
                    print(f"数据库连接: {conn_info.get('host', 'localhost')}:{conn_info.get('port', '5432')}/{conn_info.get('database', '')}")
                    print(f"用户: {conn_info.get('user', '')}")
                
                # 显示版本信息
                if 'version' in status:
                    version_info = status['version']
                    for k, v in version_info.items():
                        print(f"{k.capitalize()} 版本: {v}")
                
                # 显示表信息
                if 'tables' in status:
                    print("\\n表数据:")
                    for table, count in status['tables'].items():
                        size = status.get('size', {}).get(table, '未知')
                        print(f"  - {table}: {count} 条记录, 大小: {size}")
                
                # 显示总大小
                if 'total_size' in status:
                    print(f"\\n数据库总大小: {status['total_size']}")
                
                print("==========================\\n")
                
            # SQLite老版本兼容处理
            elif active_storage_type == 'sqlite' and hasattr(storage, 'db_path'):
                # 获取数据库文件大小
                db_path = storage.db_path
                
                if os.path.exists(db_path):
                    db_size = os.path.getsize(db_path) / (1024 * 1024)  # 转换为MB
                    
                    # 获取表数量
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                    table_count = cursor.fetchone()[0]
                    
                    # 获取各类表的数量
                    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'historical_stock_%'")
                    hist_stock_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'historical_index_%'")
                    hist_index_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'realtime_%'")
                    realtime_count = cursor.fetchone()[0]
                    
                    # 输出状态信息
                    print("\\n====== 存储状态信息 ======")
                    print(f"存储类型: SQLite")
                    print(f"数据库路径: {db_path}")
                    print(f"数据库大小: {db_size:.2f} MB")
                    print(f"表数量: {table_count}")
                    print(f"  - 股票历史数据表: {hist_stock_count}")
                    print(f"  - 指数历史数据表: {hist_index_count}")
                    print(f"  - 实时数据表: {realtime_count}")
                    print("==========================\\n")
                    
                    conn.close()
                else:
                    print(f"数据库文件不存在: {db_path}")
            else:
                print(f"存储类型 {active_storage_type} 不支持状态查询")
                
        except Exception as e:
            logger.error(f"获取存储状态失败: {e}")
            print(f"获取存储状态失败: {e}")
    
    elif args.optimize:
        try:
            logger.info("开始优化存储...")
            print(f"开始优化 {active_storage_type} 数据库...")
            
            # 使用存储对象的优化方法
            if hasattr(storage, 'optimize'):
                storage.optimize()
                print("数据库优化完成")
            elif hasattr(storage, '_optimize_database'):  # 兼容旧版SQLite
                storage._optimize_database()
                print("基本数据库优化完成")
                
                # 调用索引优化方法
                if hasattr(storage, 'optimize_indexes'):
                    print("开始优化数据库索引...")
                    storage.optimize_indexes()
                    print("索引优化完成")
                
                print("数据库优化完成！")
            else:
                print(f"存储类型 {active_storage_type} 不支持优化操作")
            
        except Exception as e:
            logger.error(f"优化存储失败: {e}")
            print(f"优化存储失败: {e}")
'''

    # 更新manage_storage函数
    updated_content = content.replace(old_manage_storage, new_manage_storage)
    
    # 写入更新后的文件
    with open(main_py_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    print_color("成功更新main.py中的manage_storage函数", "green")
    return True

def main():
    """主函数"""
    print_color("开始更新存储状态查询支持...", "blue")
    
    # 确保在项目根目录执行
    if os.path.basename(os.getcwd()) != "Beeshare":
        # 尝试查找项目根目录
        if os.path.exists("../Beeshare"):
            os.chdir("../Beeshare")
        elif os.path.exists("Beeshare"):
            os.chdir("Beeshare")
        else:
            print_color("请在BeeShare项目根目录下运行此脚本", "red")
            sys.exit(1)
    
    # 更新manage_storage函数
    if update_manage_storage_function():
        print_color("\n✅ manage_storage函数更新成功！", "green")
        print_color("现在您可以使用 'python main.py storage --status' 查看PostgreSQL/TimescaleDB数据库状态", "green")
    else:
        print_color("\n❌ manage_storage函数更新失败", "red")
        sys.exit(1)

if __name__ == "__main__":
    main() 