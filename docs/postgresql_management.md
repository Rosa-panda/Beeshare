# PostgreSQL/TimescaleDB 管理功能

本文档描述了 BeeShare 系统中用于管理 PostgreSQL/TimescaleDB 的功能。

## 功能概述

BeeShare 系统已成功迁移到使用 PostgreSQL 与 TimescaleDB 扩展作为其主要数据存储引擎。为了便于管理数据库，系统实现了以下功能：

1. **数据库状态查询**：显示数据库连接信息、版本信息、表数据统计以及总体大小。
2. **数据库优化**：执行 VACUUM ANALYZE 操作以优化数据库性能。

## 实现细节

### 存储状态查询

PostgreSQL 存储实现了 `get_status` 方法，用于获取数据库状态信息：

```python
def get_status(self, **kwargs):
    """获取数据库状态
    
    Returns:
        dict: 数据库状态信息
    """
    status = {
        "type": "PostgreSQL/TimescaleDB",
        "connection": {
            "host": self.config['connection']['host'],
            "port": self.config['connection']['port'],
            "database": self.config['connection']['database'],
            "user": self.config['connection']['user']
        },
        "tables": {},
        "size": {},
        "version": {}
    }
    
    # 获取版本信息
    with self.engine.connect() as conn:
        pg_version = conn.execute(text("SHOW server_version")).fetchone()[0]
        ts_version = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='timescaledb'")).fetchone()[0]
        status["version"] = {
            "postgresql": pg_version,
            "timescaledb": ts_version
        }
        
        # 获取表信息
        for data_type, table_name in self._table_mappings.items():
            count_query = text(f"SELECT COUNT(*) FROM {table_name}")
            size_query = text(f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))")
            
            count = conn.execute(count_query).fetchone()[0]
            size = conn.execute(size_query).fetchone()[0]
            
            status["tables"][table_name] = count
            status["size"][table_name] = size
        
        # 获取数据库总大小
        total_size_query = text("SELECT pg_size_pretty(pg_database_size(current_database()))")
        status["total_size"] = conn.execute(total_size_query).fetchone()[0]
    
    return status
```

主程序中的 `manage_storage` 函数已更新，以使用上述 `get_status` 方法：

```python
if args.status:
    try:
        logger.info("获取存储状态信息...")
        
        # 使用存储对象的get_status方法获取状态
        if hasattr(storage, 'get_status'):
            status = storage.get_status()
            
            print("\n====== 存储状态信息 ======")
            print(f"存储类型: {status.get('type', active_storage_type)}")
            
            # 显示连接信息、版本信息、表数据和总大小
            # ...
```

### 数据库优化

优化功能通过 `optimize` 方法实现：

```python
def optimize(self, **kwargs):
    """优化数据库
    
    Returns:
        bool: 优化是否成功
    """
    # 对于 VACUUM 命令，需要在自动提交模式下执行
    with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        # 执行VACUUM ANALYZE
        conn.execute(text("VACUUM ANALYZE"))
        logger.info("已执行VACUUM ANALYZE优化")
        
        # 检查物化视图是否存在并刷新
        # ...
        
        logger.info("数据库优化完成")
        return True
```

## 使用方法

### 查看数据库状态

使用以下命令查看 PostgreSQL/TimescaleDB 数据库的状态：

```bash
python main.py storage --status
```

输出示例：

```
====== 存储状态信息 ======
存储类型: PostgreSQL/TimescaleDB
数据库连接: localhost:5432/beeshare_db
用户: beeshare
Postgresql 版本: 14.17 (Ubuntu 14.17-1.pgdg22.04+1)
Timescaledb 版本: 2.19.3

表数据:
  - stock_historical_data: 193 条记录, 大小: 32 kB
  - stock_realtime_data: 120 条记录, 大小: 32 kB
  - stocks: 3 条记录, 大小: 56 kB
  - index_data: 186 条记录, 大小: 32 kB

数据库总大小: 17 MB
==========================
```

### 优化数据库

使用以下命令优化 PostgreSQL/TimescaleDB 数据库：

```bash
python main.py storage --optimize
```

优化过程包括：

1. 执行 `VACUUM ANALYZE` 操作，回收空间并更新统计信息
2. 如果存在物化视图，刷新物化视图

## 维护建议

### 定期优化

建议定期执行数据库优化操作，以保持良好的性能：

1. 每天一次（对于高频交易数据）
2. 每周一次（对于一般使用情况）

可以使用 cron 任务自动执行：

```bash
# 每周日凌晨3点执行优化
0 3 * * 0 cd /path/to/beeshare && python main.py storage --optimize
```

### 监控数据库大小

通过 `--status` 命令定期监控数据库大小，如果数据库增长过快，可以考虑：

1. 实施数据分区策略
2. 配置数据保留策略
3. 归档历史数据

## 故障排除

### VACUUM 失败

如果 VACUUM 操作失败，可能的原因包括：

1. 数据库连接设置问题
2. 权限不足
3. 磁盘空间不足

解决方案：

1. 检查数据库连接配置
2. 确保用户有足够的权限
3. 确保系统有足够的磁盘空间

### 版本兼容性

不同版本的 PostgreSQL 和 TimescaleDB 可能有不同的行为。如果遇到兼容性问题，请检查：

1. PostgreSQL 版本（目前使用：14.17）
2. TimescaleDB 版本（目前使用：2.19.3） 