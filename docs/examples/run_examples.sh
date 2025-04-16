#!/bin/bash
#====================================================
# BeeShare时间管理工具示例运行脚本 (Linux/Mac版)
# 
# 该脚本用于在Linux/Mac环境下轻松运行时间管理工具的示例
# 创建日期: 2024-04-15
# 最后修改: 2024-04-15
# 作者: BeeShare开发团队
#====================================================

# 使用方法: 
# 1. 确保脚本有执行权限: chmod +x run_examples.sh
# 2. 运行脚本: ./run_examples.sh

# 清屏函数 - 清除终端内容
clear_screen() {
    clear
}

# 显示菜单函数 - 打印选项菜单
show_menu() {
    echo "BeeShare时间管理工具示例"
    echo "================================="
    echo
    echo "请选择要运行的示例:"
    echo "1. 网络时间工具示例"
    echo "2. 日期计算工具示例"
    echo "3. 文档日期更新工具示例"
    echo "4. 运行所有示例"
    echo "0. 退出"
    echo
    echo -n "请输入选项 (0-4): "
}

# 运行示例函数 - 根据选项运行相应的示例
run_example() {
    case $1 in
        1)
            echo
            echo "正在运行网络时间工具示例..."
            echo
            python network_time_example.py
            ;;
        2)
            echo
            echo "正在运行日期计算工具示例..."
            echo
            python date_utils_example.py
            ;;
        3)
            echo
            echo "正在运行文档日期更新工具示例..."
            echo
            python update_dates_example.py
            ;;
        4)
            echo
            echo "正在运行所有示例..."
            echo
            
            echo "=== 网络时间工具示例 ==="
            python network_time_example.py
            echo
            
            echo "=== 日期计算工具示例 ==="
            python date_utils_example.py
            echo
            
            echo "=== 文档日期更新工具示例 ==="
            python update_dates_example.py
            ;;
        *)
            echo "无效选项!"
            ;;
    esac
}

# 主循环 - 程序的主要逻辑
while true; do
    # 清屏并显示菜单
    clear_screen
    show_menu
    read choice
    
    # 处理用户选择
    if [ "$choice" = "0" ]; then
        # 退出程序
        echo "感谢使用BeeShare时间管理工具示例！"
        exit 0
    fi
    
    # 运行所选示例
    run_example $choice
    echo
    read -p "按Enter键继续..." dummy
done 