# 列标准化系统增强方案

## 1. 现状分析

现有的列标准化系统(`column_mapping.py`)提供了将不同数据源的列名映射为统一标准格式的功能，这对于跨数据源的数据处理和分析非常重要。目前支持的主要功能包括：

1. 通过`StandardColumns`枚举定义标准列名
2. 提供多个数据源(A股、AKShare、Tushare、美股)的列名映射
3. 实现`standardize_columns`函数进行列名标准化
4. 提供辅助函数用于检测列问题和提供反向映射

## 2. 增强目标

在保持现有功能的基础上，我们提出以下增强目标：

1. **提高标准化成功率**：改进映射机制，更智能地处理未知列名
2. **扩展支持的数据源**：增加更多数据源的映射支持
3. **增加智能推断功能**：基于列名相似度和数据内容自动推断可能的映射
4. **加强数据验证**：在标准化过程中提供更强的数据验证能力
5. **提供更灵活的配置方式**：允许用户通过外部配置文件定制映射规则

## 3. 具体实施方案

### 3.1 增强映射机制

#### 3.1.1 模糊匹配实现

添加基于相似度的模糊匹配功能，匹配未明确定义的列名：

```python
def find_closest_column_match(column_name, target_columns, threshold=0.8):
    """
    查找最相似的列名匹配
    
    Args:
        column_name: 待匹配的列名
        target_columns: 目标列名列表
        threshold: 相似度阈值
        
    Returns:
        最匹配的列名或None
    """
    from difflib import SequenceMatcher
    
    best_match = None
    best_ratio = 0
    
    for target in target_columns:
        ratio = SequenceMatcher(None, column_name.lower(), target.lower()).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = target
            
    return best_match
```

#### 3.1.2 数据内容推断

基于列数据内容推断可能的列类型：

```python
def infer_column_type(df, column_name):
    """
    根据列的数据特征推断可能的列类型
    
    Args:
        df: 数据框
        column_name: 列名
        
    Returns:
        推断的列类型
    """
    col_data = df[column_name]
    
    # 日期类型判断
    if pd.api.types.is_datetime64_any_dtype(col_data):
        return StandardColumns.DATE
    
    # 数值类型判断
    if pd.api.types.is_numeric_dtype(col_data):
        # 检查数值范围特征
        if col_data.min() >= 0 and col_data.max() <= 100:
            # 可能是百分比列
            if 'change' in column_name.lower() or 'pct' in column_name.lower():
                return StandardColumns.CHANGE_PCT
            elif 'rate' in column_name.lower():
                return StandardColumns.TURNOVER
                
        # 检查数值级别
        if col_data.mean() > 1_000_000:
            return StandardColumns.AMOUNT
        elif col_data.mean() > 10_000:
            return StandardColumns.VOLUME
    
    # 字符串类型判断
    if pd.api.types.is_string_dtype(col_data):
        if len(col_data.unique()) < 100:  # 唯一值较少
            if any(len(str(x)) == 6 for x in col_data.dropna().head(10)):
                return StandardColumns.SYMBOL  # 可能是股票代码
    
    return None  # 无法推断
```

### 3.2 扩展数据源支持

#### 3.2.1 新增数据源映射

为`SOURCE_COLUMN_MAPPINGS`添加更多数据源的映射：

```python
# 新增雅虎财经数据源映射
"YAHOO_FINANCE": {
    "Date": StandardColumns.DATE,
    "Open": StandardColumns.OPEN,
    "High": StandardColumns.HIGH,
    "Low": StandardColumns.LOW,
    "Close": StandardColumns.CLOSE,
    "Adj Close": StandardColumns.CLOSE,  # 复权收盘价
    "Volume": StandardColumns.VOLUME,
    "Symbol": StandardColumns.SYMBOL,
    "Name": StandardColumns.NAME,
    "Industry": StandardColumns.INDUSTRY,
    "PE Ratio (TTM)": StandardColumns.PE,
    "Market Cap": StandardColumns.MARKET_CAP,
},

# 新增新浪财经数据源映射
"SINA_FINANCE": {
    "日期": StandardColumns.DATE,
    "开盘价": StandardColumns.OPEN,
    "最高价": StandardColumns.HIGH,
    "最低价": StandardColumns.LOW,
    "收盘价": StandardColumns.CLOSE,
    "成交量": StandardColumns.VOLUME,
    "成交额": StandardColumns.AMOUNT,
    "股票代码": StandardColumns.SYMBOL,
    "股票名称": StandardColumns.NAME,
    "涨跌额": StandardColumns.CHANGE,
    "涨跌幅": StandardColumns.CHANGE_PCT,
},

# 新增东方财富数据源映射
"EASTMONEY": {
    "交易日期": StandardColumns.DATE,
    "开盘价": StandardColumns.OPEN,
    "最高价": StandardColumns.HIGH,
    "最低价": StandardColumns.LOW,
    "收盘价": StandardColumns.CLOSE,
    "成交量": StandardColumns.VOLUME,
    "成交额": StandardColumns.AMOUNT,
    "代码": StandardColumns.SYMBOL,
    "名称": StandardColumns.NAME,
    "涨跌额": StandardColumns.CHANGE,
    "涨跌幅": StandardColumns.CHANGE_PCT,
    "换手率": StandardColumns.TURNOVER,
    "市盈率": StandardColumns.PE,
    "市净率": StandardColumns.PB,
    "总市值": StandardColumns.MARKET_CAP,
    "流通市值": StandardColumns.CIRC_MARKET_CAP,
}
```

#### 3.2.2 用户自定义映射支持

添加从JSON配置文件加载用户自定义映射的功能：

```python
def load_user_mappings(mapping_file="config/column_mappings.json"):
    """
    从配置文件加载用户自定义的列映射
    
    Args:
        mapping_file: 映射配置文件路径
        
    Returns:
        加载的映射字典
    """
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            user_mappings = json.load(f)
            
        # 验证用户配置格式
        for source, mappings in user_mappings.items():
            for col, std_col in mappings.items():
                # 验证标准列名是否有效
                if std_col not in [c.value for c in StandardColumns]:
                    logger.warning(f"用户配置中的标准列名 '{std_col}' 不是有效的StandardColumns值")
                    
        logger.info(f"从 {mapping_file} 成功加载了用户自定义映射")
        return user_mappings
    except Exception as e:
        logger.error(f"加载用户映射失败: {e}")
        return {}
```

### 3.3 改进标准化函数

增强`standardize_columns`函数，添加智能映射和验证功能：

```python
def standardize_columns(df: pd.DataFrame, source_type: str, 
                       use_fuzzy_match: bool = False,
                       infer_column_types: bool = False,
                       logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    增强版列名标准化函数
    
    Args:
        df: 输入数据框
        source_type: 数据源类型
        use_fuzzy_match: 是否使用模糊匹配
        infer_column_types: 是否推断列类型
        logger: 可选的日志记录器
        
    Returns:
        标准化后的数据框
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if df is None or df.empty:
        logger.warning("无法标准化空数据框")
        return df
        
    # 首先尝试通过常规映射进行标准化
    mapping = {}
    if source_type in SOURCE_COLUMN_MAPPINGS:
        mapping = SOURCE_COLUMN_MAPPINGS[source_type]
    else:
        # 尝试加载用户自定义映射
        user_mappings = load_user_mappings()
        if source_type in user_mappings:
            mapping = user_mappings[source_type]
        else:
            logger.warning(f"未知的数据源类型: {source_type}，无法进行列名标准化")
            # 尝试使用模糊匹配或类型推断
    
    renamed_columns = {}
    
    # 第一步：应用精确映射
    for col in df.columns:
        if col in mapping:
            std_col = mapping[col]
            # 如果是StandardColumns枚举类型，获取它的值
            if isinstance(std_col, StandardColumns):
                std_col = std_col.value
            renamed_columns[col] = std_col
            logger.debug(f"精确列名映射: {col} -> {std_col}")
    
    # 第二步：应用模糊匹配（如果启用）
    if use_fuzzy_match:
        unmapped_columns = [col for col in df.columns if col not in renamed_columns]
        standard_column_values = [c.value for c in StandardColumns]
        
        for col in unmapped_columns:
            # 首先尝试匹配键
            key_match = find_closest_column_match(col, list(mapping.keys()))
            if key_match and key_match not in renamed_columns:
                std_col = mapping[key_match]
                if isinstance(std_col, StandardColumns):
                    std_col = std_col.value
                renamed_columns[col] = std_col
                logger.debug(f"模糊键匹配: {col} -> {std_col} (通过 {key_match})")
                continue
                
            # 然后尝试直接匹配标准列名
            std_match = find_closest_column_match(col, standard_column_values)
            if std_match:
                renamed_columns[col] = std_match
                logger.debug(f"模糊值匹配: {col} -> {std_match}")
    
    # 第三步：根据数据内容推断（如果启用）
    if infer_column_types:
        still_unmapped = [col for col in df.columns if col not in renamed_columns]
        for col in still_unmapped:
            inferred_type = infer_column_type(df, col)
            if inferred_type:
                std_col = inferred_type.value if isinstance(inferred_type, StandardColumns) else inferred_type
                renamed_columns[col] = std_col
                logger.debug(f"内容推断映射: {col} -> {std_col}")
    
    # 如果没有列被映射，发出警告
    if not renamed_columns:
        logger.warning(f"数据源 {source_type} 没有任何列可以映射到标准列名")
        return df  # 返回原始DataFrame，而不是空DataFrame
        
    # 记录映射信息
    logger.info(f"数据源 {source_type} 映射了 {len(renamed_columns)}/{len(df.columns)} 列到标准列名")
    
    # 重命名列
    result = df.rename(columns=renamed_columns)
    
    # 验证结果
    if result.empty and not df.empty:
        logger.warning("列名标准化导致数据丢失，返回原始数据")
        return df
        
    # 记录未映射的列名
    unmapped = [col for col in df.columns if col not in renamed_columns]
    if unmapped:
        logger.debug(f"以下列未映射: {unmapped}")
        
    return result
```

### 3.4 日期格式处理增强

增加日期格式标准化功能：

```python
def standardize_date_format(df, date_column='date', inplace=False):
    """
    标准化日期列的格式
    
    Args:
        df: 数据框
        date_column: 日期列名（应已标准化为'date'）
        inplace: 是否原位修改
        
    Returns:
        标准化日期格式后的数据框
    """
    result = df if inplace else df.copy()
    
    if date_column in result.columns:
        # 尝试将日期列转换为datetime格式
        try:
            # 先检测原始格式
            if pd.api.types.is_datetime64_any_dtype(result[date_column]):
                # 已经是日期类型，确保格式一致
                result[date_column] = pd.to_datetime(result[date_column])
            elif pd.api.types.is_string_dtype(result[date_column]):
                # 字符串格式，尝试自动解析
                result[date_column] = pd.to_datetime(result[date_column], errors='coerce')
            elif pd.api.types.is_integer_dtype(result[date_column]):
                # 可能是整数时间戳
                if result[date_column].max() > 10000000000:  # 毫秒级时间戳
                    result[date_column] = pd.to_datetime(result[date_column], unit='ms')
                else:  # 秒级时间戳
                    result[date_column] = pd.to_datetime(result[date_column], unit='s')
        except Exception as e:
            logging.warning(f"转换日期格式失败: {e}")
            
    return result
```

## 4. 实现计划

1. **第一阶段**：基础功能增强
   - 实现模糊匹配功能
   - 添加新的数据源映射
   - 编写相应的单元测试

2. **第二阶段**：高级功能开发
   - 实现基于内容的列类型推断
   - 开发用户自定义映射加载功能
   - 增强日期格式标准化功能

3. **第三阶段**：集成和测试
   - 将新功能集成到现有系统
   - 更新相关文档
   - 全面测试不同数据源的标准化效果

## 5. 使用示例

```python
from src.utils.column_mapping import standardize_columns, standardize_date_format

# 加载原始数据
raw_data = load_stock_data_from_some_source()

# 应用增强版标准化功能
standardized_data = standardize_columns(
    raw_data, 
    source_type='SOME_SOURCE', 
    use_fuzzy_match=True,
    infer_column_types=True
)

# 标准化日期格式
standardized_data = standardize_date_format(standardized_data)

# 后续处理...
```

## 6. 预期效果

通过这些增强措施，我们预期将：

1. 提高列标准化的成功率，减少人工干预
2. 为用户提供更灵活的配置选项
3. 支持更多数据源和格式
4. 改进数据质量和一致性
5. 为后续分析提供更可靠的数据基础

这些增强功能将使BeeShare系统在处理不同来源的股票数据时更加强大和灵活。 