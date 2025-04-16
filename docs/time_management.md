# 项目时间管理指南

## 概述

在项目开发过程中，正确处理和显示时间是非常重要的。本指南介绍了在Beeshare项目中如何处理时间相关的操作，以及如何使用项目提供的时间工具。

## 时间工具库

项目提供了两个主要的时间处理工具库：

### 1. time_utils.py

位置：`src/utils/time_utils.py`

主要功能：
- 获取网络时间（优先通过网络API获取，确保时间准确性）
- 生成格式化的日期字符串

使用示例：
```python
from src.utils.time_utils import get_network_time, get_formatted_date

# 获取当前网络时间
current_time = get_network_time()
print(f"当前时间是：{current_time}")

# 获取格式化的日期字符串（默认格式：YYYY-MM-DD）
today = get_formatted_date()
print(f"今天是：{today}")

# 自定义格式
custom_date = get_formatted_date('%Y年%m月%d日')
print(f"今天是：{custom_date}")
```

### 2. date_utils.py

位置：`src/utils/date_utils.py`

主要功能：
- 日期格式转换
- 日期计算（如获取N天前的日期）

使用示例：
```python
from src.utils.date_utils import get_date_n_days_ago

# 获取90天前的日期
ninety_days_ago = get_date_n_days_ago()
print(f"90天前是：{ninety_days_ago}")

# 获取10天前的日期
ten_days_ago = get_date_n_days_ago(10)
print(f"10天前是：{ten_days_ago}")

# 自定义输出格式
formatted_date = get_date_n_days_ago(5, '%Y年%m月%d日')
print(f"5天前是：{formatted_date}")
```

## 自动更新文档日期工具

为了确保项目中所有Python文件的文档字符串中的创建日期和最后修改日期保持准确，我们提供了一个自动更新工具。

### 使用方法

1. 导航到项目根目录
2. 运行以下命令：

```bash
python src/tools/update_doc_dates.py
```

该工具会：
- 扫描项目中所有Python文件
- 使用网络时间更新文档字符串中的"创建日期"和"最后修改"字段
- 生成操作日志，显示更新了哪些文件

### 何时使用

推荐在以下情况使用该工具：
- 创建新的Python模块或脚本后
- 对现有文件进行重大修改后
- 发布新版本前，确保所有文件的日期信息都是最新的

## 最佳实践

1. **使用网络时间**：在需要获取当前时间的场景中，优先使用`time_utils.py`中的函数，而不是Python内置的`datetime`模块，以确保时间准确性。

2. **统一日期格式**：项目默认使用`YYYY-MM-DD`格式表示日期，除非有特殊需求，应坚持使用此格式。

3. **文档日期规范**：每个Python文件的文档字符串中应包含以下信息：
   ```python
   """
   模块描述。
   
   详细说明...
   
   创建日期: YYYY-MM-DD
   最后修改: YYYY-MM-DD
   """
   ```

4. **避免硬编码时间**：避免在代码中硬编码日期和时间，而是使用相对时间（如"90天前"）或通过配置文件设置时间参数。

5. **时区处理**：当处理跨时区的时间时，确保明确指定时区，避免产生混淆。

## 常见问题解答

**Q: 为什么要使用网络时间而不是系统时间？**

A: 网络时间通常更准确，不依赖于本地系统设置，可以避免因用户系统时间不准确而导致的问题。特别是在分布式系统中，保持时间一致性尤为重要。

**Q: 如何测试网络时间获取功能？**

A: 可以通过以下方式测试：
```python
from src.utils.time_utils import get_network_time, get_formatted_date
import time

# 获取网络时间
network_time = get_network_time()
print(f"网络时间: {network_time}")

# 获取系统时间作为比较
system_time = time.time()
print(f"系统时间: {system_time}")

# 比较两者差异
print(f"时间差异: {abs(network_time - system_time)} 秒")
```

**Q: 我修改了文件但忘记更新文档日期，怎么办？**

A: 运行`update_doc_dates.py`工具，它会自动更新所有文件的日期信息。 