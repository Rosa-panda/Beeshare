@echo off
REM ====================================================
REM BeeShare时间管理工具示例运行脚本 (Windows版)
REM 
REM 该脚本用于在Windows环境下轻松运行时间管理工具示例
REM 创建日期: 2024-04-15
REM 最后修改: 2024-04-15
REM 作者: BeeShare开发团队
REM ====================================================

REM 显示欢迎信息和菜单
echo BeeShare时间管理工具示例
echo =================================
echo.
echo 请选择要运行的示例:
echo 1. 网络时间工具示例
echo 2. 日期计算工具示例
echo 3. 文档日期更新工具示例
echo 4. 运行所有示例
echo 0. 退出
echo.

REM 菜单选择标签
:menu
set /p choice=请输入选项 (0-4): 

REM 根据用户选择执行相应的示例
if "%choice%"=="1" (
    echo.
    echo 正在运行网络时间工具示例...
    echo.
    python network_time_example.py
    echo.
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo 正在运行日期计算工具示例...
    echo.
    python date_utils_example.py
    echo.
    pause
    goto menu
) else if "%choice%"=="3" (
    echo.
    echo 正在运行文档日期更新工具示例...
    echo.
    python update_dates_example.py
    echo.
    pause
    goto menu
) else if "%choice%"=="4" (
    echo.
    echo 正在运行所有示例...
    echo.
    
    echo === 网络时间工具示例 ===
    python network_time_example.py
    echo.
    
    echo === 日期计算工具示例 ===
    python date_utils_example.py
    echo.
    
    echo === 文档日期更新工具示例 ===
    python update_dates_example.py
    echo.
    
    pause
    goto menu
) else if "%choice%"=="0" (
    REM 退出脚本
    echo 感谢使用BeeShare时间管理工具示例！
    goto end
) else (
    REM 无效选项处理
    echo 无效选项，请重新输入！
    goto menu
)

REM 脚本结束标签
:end 