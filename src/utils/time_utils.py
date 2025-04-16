"""
时间工具模块。

该模块提供获取当前网络时间和生成格式化日期字符串的功能，
用于确保项目文档和代码中使用的日期时间信息准确一致。

创建日期: 2023-04-15
最后修改: 2024-04-15
作者: BeeShare开发团队
"""

import datetime
import time
import urllib.request
import json
import socket
import logging

logger = logging.getLogger(__name__)

def get_network_time():
    """
    获取当前的网络时间。
    
    尝试从多个时间API获取当前准确的网络时间。如果所有API都失败，
    则回退到系统本地时间。
    
    Returns:
        datetime.datetime: 当前的网络时间
        
    Raises:
        urllib.error.URLError: 当无法连接到时间服务器时可能抛出
        socket.error: 当NTP服务器连接失败时可能抛出
        
    Examples:
        >>> current_time = get_network_time()
        >>> print(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        '2023-04-15 08:30:45'  # 示例输出
    """
    # 尝试从worldtimeapi.org获取时间
    try:
        response = urllib.request.urlopen('http://worldtimeapi.org/api/ip', timeout=3)
        data = json.loads(response.read().decode())
        time_str = data['datetime']
        # 解析ISO 8601格式的时间字符串
        network_time = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        logger.info(f"成功从worldtimeapi.org获取网络时间: {network_time}")
        return network_time
    except Exception as e:
        logger.warning(f"从worldtimeapi.org获取时间失败: {e}")
    
    # 尝试从网络时间服务器获取时间（NTP协议）
    try:
        # NTP服务器
        ntp_server = 'pool.ntp.org'
        # 参考时间（1970年1月1日）
        ref_time = datetime.datetime(1970, 1, 1)
        # 创建UDP套接字
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(3)
        # NTP请求的结构
        data = b'\x1b' + 47 * b'\0'
        # 发送请求
        client.sendto(data, (ntp_server, 123))
        # 接收响应
        data, address = client.recvfrom(1024)
        # 关闭套接字
        client.close()
        
        if data:
            # 解析NTP时间（从数据包的第40字节开始的8字节表示秒数）
            t = int.from_bytes(data[40:48], byteorder='big') - 2208988800
            # 转换为datetime对象
            network_time = ref_time + datetime.timedelta(seconds=t)
            logger.info(f"成功从NTP服务器获取网络时间: {network_time}")
            return network_time
    except Exception as e:
        logger.warning(f"从NTP服务器获取时间失败: {e}")
    
    # 如果所有方法都失败，使用系统时间
    logger.warning("所有网络时间获取方法失败，使用系统本地时间")
    return datetime.datetime.now()

def get_formatted_date(date_format='%Y-%m-%d'):
    """
    获取格式化的当前日期字符串。
    
    使用网络时间获取当前日期，并根据指定格式进行格式化。
    
    Args:
        date_format (str): 日期格式字符串，默认为'%Y-%m-%d'
        
    Returns:
        str: 格式化的日期字符串
        
    Raises:
        ValueError: 当date_format参数不是有效的日期格式字符串时抛出
        
    Examples:
        >>> today = get_formatted_date()
        >>> print(today)
        '2023-04-15'  # 示例输出
        
        >>> today_custom = get_formatted_date('%Y年%m月%d日')
        >>> print(today_custom)
        '2023年04月15日'  # 示例输出
    """
    current_time = get_network_time()
    return current_time.strftime(date_format) 