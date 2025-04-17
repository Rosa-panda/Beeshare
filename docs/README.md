# BeeShare列标准化系统增强版

## 项目概述

本项目为BeeShare系统提供了增强版的列标准化功能。列标准化系统负责将不同数据源的列名映射为标准格式，使数据在整个系统中保持一致性。增强版在原有功能基础上，添加了模糊匹配、内容推断和用户自定义映射支持等智能特性。

## 主要功能

1. **模糊匹配**：通过字符串相似度算法，识别相似但不完全相同的列名
2. **内容推断**：基于列数据内容和特征，推断列的实际含义和类型
3. **日期格式标准化**：自动识别和转换各种日期格式
4. **用户自定义映射**：支持从配置文件加载自定义的列名映射
5. **完整的标准化流程**：结合精确匹配、模糊匹配和内容推断，提高列标准化的成功率

## 实现方式

本项目包含以下核心组件：

- `enhanced_column_mapping.py`：增强版列标准化功能的主要实现
- `column_mappings.json`：用户自定义映射配置文件
- `test_enhanced_mapping.py`：测试用例
- `enhanced_mapping_demo.py`：使用示例

## 使用方法

### 基本用法

```python
from src.utils.enhanced_column_mapping import standardize_columns

# 使用精确映射
result = standardize_columns(df, 'A_SHARE')

# 启用模糊匹配
result = standardize_columns(df, 'A_SHARE', use_fuzzy_match=True)

# 启用内容推断
result = standardize_columns(df, 'UNKNOWN_SOURCE', infer_column_types=True)

# 使用所有功能
result = standardize_columns(df, 'A_SHARE', use_fuzzy_match=True, infer_column_types=True)
```

### 日期格式标准化

```python
from src.utils.enhanced_column_mapping import standardize_date_format

# 标准化日期列
df = standardize_date_format(df, date_column='date')
```

### 自定义数据源

1. 在`config/column_mappings.json`中添加自定义映射：

```json
{
  "CUSTOM_SOURCE": {
    "日期时间": "date",
    "开盘价格": "open",
    "收盘价格": "close",
    // 更多映射...
  }
}
```

2. 使用自定义数据源：

```python
# 将使用config/column_mappings.json中的映射
result = standardize_columns(df, 'CUSTOM_SOURCE')
```

## 运行演示

演示脚本展示了增强版列标准化功能的各项特性：

```bash
python examples/enhanced_mapping_demo.py
```

## 测试

运行测试用例以验证功能：

```bash
python tests/test_enhanced_mapping.py
```

## 与原有系统的集成

增强版列标准化功能与原有的`column_mapping.py`模块完全兼容，可以平滑替换或并行使用：

```python
# 原有标准化函数
from src.utils.column_mapping import standardize_columns as old_standardize

# 增强版标准化函数
from src.utils.enhanced_column_mapping import standardize_columns as enhanced_standardize

# 原有方式调用
df1 = old_standardize(df, 'A_SHARE')

# 增强版调用
df2 = enhanced_standardize(df, 'A_SHARE', use_fuzzy_match=True, infer_column_types=True)
```

## 文件结构

```
beeshare/
├── config/
│   └── column_mappings.json    # 用户自定义映射配置
├── docs/
│   ├── column_mapping_analysis.md    # 原始列映射系统分析
│   ├── column_mapping_enhancement.md # 增强方案文档
│   └── README.md               # 本文档
├── examples/
│   └── enhanced_mapping_demo.py # 演示脚本
├── src/
│   └── utils/
│       ├── column_mapping.py   # 原始列映射系统
│       └── enhanced_column_mapping.py # 增强版系统
└── tests/
    ├── test_standardize.py     # 原始功能测试
    └── test_enhanced_mapping.py # 增强功能测试
```

## 贡献与维护

增强版列标准化系统的设计目标是易于扩展和维护：

1. 要添加新的数据源映射，可以直接编辑`config/column_mappings.json`文件
2. 要改进模糊匹配算法，可以修改`find_closest_column_match`函数
3. 要增强内容推断能力，可以更新`infer_column_type`函数的规则

## 未来改进计划

1. 支持更多数据源类型
2. 实现机器学习辅助的列类型推断
3. 提供Web界面配置列名映射
4. 增加列数据验证和清洗功能

## 许可证

本项目与BeeShare系统使用相同的许可证。 