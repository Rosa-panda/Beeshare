# A股数据获取系统 API 使用指南

本文档详细说明了A股数据获取系统的API使用方法，包括数据源API和存储API的详细说明、参数解释以及使用示例。

## 目录

- [1. 数据源API](#1-数据源api)
  - [1.1 初始化](#11-初始化)
  - [1.2 获取历史数据](#12-获取历史数据)
  - [1.3 获取实时数据](#13-获取实时数据)
  - [1.4 搜索股票信息](#14-搜索股票信息)
  - [1.5 获取指数数据](#15-获取指数数据)
  - [1.6 获取股票详细信息](#16-获取股票详细信息)
- [2. 存储API](#2-存储api)
  - [2.1 CSV存储](#21-csv存储)
  - [2.2 SQLite存储](#22-sqlite存储)
- [3. 完整使用示例](#3-完整使用示例)
- [4. 常见问题](#4-常见问题)

## 1. 数据源API

数据源API主要通过`AKShareDS`类实现，提供A股数据的获取功能。

### 1.1 初始化

首先需要初始化AKShare数据源对象：

```python
from src.data_sources import AKShareDS
from config.config import DATA_SOURCES

# 初始化数据源
akshare_ds = AKShareDS(DATA_SOURCES['akshare'])

# 检查连接状态
if akshare_ds.check_connection():
    print("AKShare数据源连接成功")
else:
    print("AKShare数据源连接失败")
```

### 1.2 获取历史数据

获取指定股票的历史交易数据：

```python
# 获取600036(招商银行)从2023-01-01到2023-12-31的日线数据
historical_data = akshare_ds.get_historical_data(
    symbol='600036',
    start_date='2023-01-01',
    end_date='2023-12-31',
    interval='daily'  # 支持'daily', 'weekly', 'monthly'
)

# 打印数据
if not historical_data.empty:
    print(f"获取到 {len(historical_data)} 条历史数据")
    print(historical_data.head())
```

**返回数据说明**：
- DataFrame包含以下列：date, open, high, low, close, volume, amount, change_percent, change, turnover_rate, symbol, source等
- 日期(date)列为datetime类型
- 所有价格数据为浮点数类型

### 1.3 获取实时数据

获取多只股票的实时行情数据：

```python
# 获取多只股票的实时数据
symbols = ['600036', '601398', '600519']
realtime_data = akshare_ds.get_realtime_data(symbols)

# 打印数据
if not realtime_data.empty:
    print(f"获取到 {len(realtime_data)} 条实时数据")
    print(realtime_data)
```

**注意事项**：
- 实时数据受市场交易时间限制，非交易时间可能获取不到数据
- 当日数据作为实时数据返回，包含最新价格、涨跌幅等信息

### 1.4 搜索股票信息

通过关键词搜索股票：

```python
# 搜索包含"银行"的股票
results = akshare_ds.search_symbols("银行")

# 打印搜索结果
if results:
    print(f"找到 {len(results)} 个匹配的股票:")
    for item in results[:5]:  # 只显示前5个
        print(f"代码: {item['symbol']}, 名称: {item['name']}")
```

**返回数据说明**：
- 返回类型为列表，每个元素是包含股票信息的字典
- 每个字典包含symbol(代码)、name(名称)、market(市场)等字段

### 1.5 获取指数数据

获取指数的历史数据：

```python
# 获取上证指数(000001)数据
index_data = akshare_ds.get_index_data(
    symbol='000001',  # 上证指数
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 打印数据
if not index_data.empty:
    print(f"获取到 {len(index_data)} 条指数数据")
    print(index_data.head())
```

**支持的主要指数代码**：
- `000001`: 上证指数
- `399001`: 深证成指
- `000300`: 沪深300
- `000016`: 上证50
- `000905`: 中证500

**指数数据的直接获取**:
你也可以直接使用`get_historical_data`方法获取指数数据，系统会自动识别指数代码，并使用相应的接口获取数据：

```python
# 直接使用get_historical_data获取上证指数数据
index_data = akshare_ds.get_historical_data(
    symbol='000001',  # 上证指数
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

**指数数据存储**:
与普通股票数据一样，可以使用相同的存储API保存指数数据：

```python
# 将指数数据保存到CSV
success = csv_storage.save_data(
    data=index_data,
    data_type='index',  # 使用'index'类型区分指数数据
    symbol='000001',
    mode='overwrite'
)
```

### 1.6 获取股票详细信息

获取单只股票的详细信息：

```python
# 获取600519(贵州茅台)的详细信息
info = akshare_ds.get_symbol_info('600519')

# 打印信息
if info:
    print("股票详细信息:")
    for key, value in info.items():
        print(f"{key}: {value}")
```

**返回数据说明**：
- 返回类型为字典，包含股票的基本信息
- 包含字段：symbol(代码)、name(名称)、industry(行业)、listing_date(上市日期)等

## 2. 存储API

系统提供了CSV存储和SQLite存储两种方式，目前主要支持CSV存储。

### 2.1 CSV存储

CSV存储通过`CSVStorage`类实现，提供数据的保存、加载和删除功能。

**初始化**：

```python
from src.storage import CSVStorage
from config.config import STORAGE_CONFIG

# 初始化CSV存储
csv_storage = CSVStorage(STORAGE_CONFIG['csv'])
```

**保存数据**：

```python
# 保存历史数据
success = csv_storage.save_data(
    data=historical_data,          # DataFrame数据
    data_type='historical',        # 数据类型
    symbol='600036',               # 股票代码
    mode='overwrite'               # 保存模式：'append'或'overwrite'
)

if success:
    print("数据保存成功")
```

**加载数据**：

```python
# 加载历史数据
loaded_data = csv_storage.load_data(
    data_type='historical',        # 数据类型
    symbol='600036',               # 股票代码
    start_date='2023-01-01',       # 起始日期(可选)
    end_date='2023-12-31'          # 结束日期(可选)
)

if not loaded_data.empty:
    print(f"加载了 {len(loaded_data)} 条数据")
```

**删除数据**：

```python
# 删除数据
success = csv_storage.delete_data(
    data_type='historical',        # 数据类型
    symbol='600036'                # 股票代码(可选，不指定则删除所有该类型数据)
)

if success:
    print("数据删除成功")
```

### 2.2 SQLite存储

SQLite存储功能计划在未来版本实现。

## 3. 完整使用示例

下面是一个获取、分析和存储数据的完整示例：

```python
import pandas as pd
from src.data_sources import AKShareDS
from src.storage import CSVStorage
from config.config import DATA_SOURCES, STORAGE_CONFIG
from datetime import datetime, timedelta

# 初始化数据源和存储
akshare_ds = AKShareDS(DATA_SOURCES['akshare'])
csv_storage = CSVStorage(STORAGE_CONFIG['csv'])

# 设置参数
symbol = '600519'  # 贵州茅台
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # 获取近30天数据

# 1. 获取历史数据
historical_data = akshare_ds.get_historical_data(
    symbol=symbol,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

if not historical_data.empty:
    print(f"成功获取 {symbol} 的历史数据，共 {len(historical_data)} 行")
    
    # 2. 简单数据分析
    avg_price = historical_data['close'].mean()
    max_price = historical_data['high'].max()
    min_price = historical_data['low'].min()
    
    print(f"平均收盘价: {avg_price:.2f}")
    print(f"最高价: {max_price:.2f}")
    print(f"最低价: {min_price:.2f}")
    
    # 3. 保存数据到CSV
    save_result = csv_storage.save_data(
        historical_data, 
        'historical', 
        symbol, 
        mode='overwrite'
    )
    
    if save_result:
        print(f"成功保存 {symbol} 的历史数据")
    
    # 4. 从CSV加载数据并验证
    loaded_data = csv_storage.load_data('historical', symbol)
    
    if not loaded_data.empty and len(loaded_data) == len(historical_data):
        print("数据验证成功")
else:
    print(f"获取 {symbol} 的历史数据失败")

# 5. 获取指数数据并分析
index_symbol = '000001'  # 上证指数
index_data = akshare_ds.get_index_data(
    symbol=index_symbol,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

if not index_data.empty:
    print(f"成功获取 {index_symbol} 指数数据，共 {len(index_data)} 行")
    
    # 简单分析
    avg_close = index_data['close'].mean()
    highest = index_data['high'].max()
    lowest = index_data['low'].min()
    
    print(f"指数平均收盘点位: {avg_close:.2f}")
    print(f"期间最高点: {highest:.2f}")
    print(f"期间最低点: {lowest:.2f}")
    
    # 保存指数数据
    csv_storage.save_data(index_data, 'index', index_symbol, mode='overwrite')
```

## 4. 常见问题

### 问题1: 连接AKShare时出现timeout错误

**解决方案**：
- 检查网络连接是否正常
- 增加配置中的timeout值：在config.py中修改`DATA_SOURCES['akshare']['timeout']`
- 可能是AKShare服务器暂时不可用，稍后再试

### 问题2: 获取实时数据返回空DataFrame

**解决方案**：
- 检查是否为交易时间
- 确认股票代码格式正确，A股代码为6位数字
- 检查AKShare连接是否正常

### 问题3: CSV文件路径问题

**解决方案**：
- 确保配置文件中的`base_path`设置正确
- 检查系统是否有对应目录的写入权限
- 可以使用绝对路径代替相对路径

### 问题4: 指数数据与个股数据列不匹配

**解决方案**：
- 系统已尝试统一指数数据和个股数据的列名，但可能存在部分差异
- 使用前先检查返回的DataFrame的列名
- 对缺失的列使用默认值或空值填充
- 在新的API中，`get_index_data`方法已专门优化处理指数数据格式，确保与股票数据列名保持一致

### 问题5: 获取指数数据时出现错误

**解决方案**：
- 确保使用正确的指数代码（如：000001为上证指数，399001为深证成指）
- 使用`get_index_data`方法代替`get_historical_data`方法获取指数数据
- 对于上证系列指数，系统会自动添加"sh"前缀，深证系列指数会自动添加"sz"前缀
- 如果使用命令行，请使用`index`子命令代替`historical`命令来获取指数数据 