# 列名标准化系统更新说明

## 背景

在系统使用过程中，发现SQLite存储模块中对于实时数据的处理存在潜在问题。主要问题是`StandardColumns`枚举中缺少实时数据相关的标准列定义（如`TIMESTAMP`、`PRICE`等），这导致了以下问题：

1. 代码中大量使用`hasattr(StandardColumns, 'TIMESTAMP')`等检查，代码结构不清晰
2. 当字段不存在时，使用字符串硬编码作为备选，不符合规范
3. 索引创建使用的是硬编码列名，可能导致索引无效

## 修改内容

我们对系统进行了以下更新：

### 1. 更新了`StandardColumns`枚举

在`src/utils/column_mapping.py`中，为标准列名枚举添加了实时数据所需的字段：

```python
# 实时数据特有字段
TIMESTAMP = "timestamp"      # 时间戳
PRICE = "price"              # 当前价格
BID_PRICE = "bid_price"      # 买入价
ASK_PRICE = "ask_price"      # 卖出价
BID_VOLUME = "bid_volume"    # 买入量
ASK_VOLUME = "ask_volume"    # 卖出量
```

### 2. 更新了存储映射字典

在`STORAGE_TO_STANDARD_MAPPING`和`STANDARD_TO_STORAGE_MAPPING`中添加了对应的映射关系：

```python
"timestamp": StandardColumns.TIMESTAMP.value,
"price": StandardColumns.PRICE.value,
"bid_price": StandardColumns.BID_PRICE.value,
"ask_price": StandardColumns.ASK_PRICE.value,
"bid_volume": StandardColumns.BID_VOLUME.value,
"ask_volume": StandardColumns.ASK_VOLUME.value,
```

### 3. 更新了数据源映射

在AKShare的映射中，增加了对应的中英文列名映射：

```python
# 中文列名
"时间": StandardColumns.TIMESTAMP,
"时间戳": StandardColumns.TIMESTAMP,
"最新价": StandardColumns.PRICE,
"买入价": StandardColumns.BID_PRICE,
"卖出价": StandardColumns.ASK_PRICE,
"买入量": StandardColumns.BID_VOLUME,
"卖出量": StandardColumns.ASK_VOLUME,

# 英文列名
"timestamp": StandardColumns.TIMESTAMP,
"price": StandardColumns.PRICE,
"bid_price": StandardColumns.BID_PRICE,
"ask_price": StandardColumns.ASK_PRICE,
"bid_volume": StandardColumns.BID_VOLUME,
"ask_volume": StandardColumns.ASK_VOLUME,
```

### 4. 优化了SQLite存储模块

在`src/storage/sqlite_storage.py`中：

1. 移除了`_get_data_type_schema`方法中的`hasattr`检查，直接使用枚举值
2. 优化了`_load_stock_data`方法中的字段选择逻辑
3. 更新了`add_time_date_index_to_realtime_tables`方法，使用正确的时间戳字段创建索引

## 优势

此次修改带来以下优势：

1. **代码更简洁**：减少了条件判断，代码逻辑更清晰
2. **标准化更完善**：实时数据的字段也纳入到标准列定义中
3. **维护性更好**：字段变更时只需修改枚举，不需要修改多处代码
4. **索引更有效**：使用正确的标准字段名创建索引，提高查询性能

## 注意事项

1. 此更新向后兼容，不影响现有数据
2. 旧版数据库中已使用"time"和"date"作为索引的表可能需要重建索引
3. 建议运行`optimize_database()`方法，重新创建优化的索引

## 后续改进

1. 考虑添加数据库版本管理机制，在库版本更新时自动进行索引优化
2. 进一步丰富标准列定义，支持更多类型的市场数据

---

更新日期：2023-07-05 