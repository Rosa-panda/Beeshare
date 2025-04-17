"""
SQLite存储模块，提供基于SQLite的数据存储实现

流程图:
```mermaid
classDiagram
    class SQLiteStorage {
        -db_path: str
        -_connections: dict
        -_connection_timeout: int
        +__init__(db_path)
        +save_data(data, data_type, symbol)
        +load_data(data_type, symbol, start_date, end_date)
        +delete_data(data_type, symbol)
        -_get_connection()
        -_get_table_name(data_type, symbol)
        -_create_tables_if_not_exists()
        -_get_data_type_schema(data_type)
        -_execute_query(query, params)
        -_add_table_indexes(table_name, columns)
        -_optimize_database()
        -_close_all_connections()
    }
    
    SQLiteStorage --|> Storage : 实现
```
"""

import os
import sqlite3
import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, date, timedelta
import numpy as np
import threading
import time
import traceback

from .base import Storage, DataType
from src.utils.exceptions import StorageException, DataSaveException, DataLoadException
from src.utils.logging_utils import log_function_call, log_exception
from src.utils.column_mapping import (
    StandardColumns,
    standardize_columns,
    STANDARD_TO_STORAGE_MAPPING,
    apply_reverse_mapping_for_analysis
)

logger = logging.getLogger(__name__)

class SQLiteStorage(Storage):
    """
    SQLite存储实现，提供基于SQLite数据库的数据存储和访问功能
    
    优化特性:
    1. 连接池管理 - 避免频繁创建和关闭连接
    2. 查询索引 - 提高查询性能
    3. 批量操作 - 使用事务和executemany提高批量插入性能
    4. 查询缓存 - 缓存常用查询结果
    5. WAL模式 - 提高并发写入性能
    """
    
    def __init__(self, db_path: str):
        """
        初始化SQLite存储
        
        Args:
            db_path (str): SQLite数据库文件路径
        """
        super().__init__("sqlite")
        self.db_path = db_path
        
        # 连接池 - 线程ID到连接的映射
        self._connections = {}
        self._connection_lock = threading.Lock()
        self._connection_timeout = 30  # 连接超时时间（秒）
        self._connection_last_used = {}  # 连接最后使用时间
        self._query_cache = {}  # 查询缓存
        self._cache_timeout = 60  # 缓存超时时间（秒）
        self._cache_lock = threading.RLock()  # 缓存锁
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 创建表结构（如果不存在）
        self._create_tables_if_not_exists()
        
        # 启动定期清理任务
        self._start_cleanup_timer()
    
    def _start_cleanup_timer(self):
        """启动定期清理任务"""
        def cleanup():
            while True:
                time.sleep(60)  # 每60秒执行一次清理
                try:
                    self._cleanup_connections()
                    self._cleanup_cache()
                except Exception as e:
                    logger.error(f"清理任务执行失败: {e}")
        
        # 创建守护线程执行清理任务
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_connections(self):
        """清理超时连接"""
        current_time = time.time()
        with self._connection_lock:
            expired_threads = []
            for thread_id, last_used in self._connection_last_used.items():
                if current_time - last_used > self._connection_timeout:
                    expired_threads.append(thread_id)
            
            for thread_id in expired_threads:
                if thread_id in self._connections:
                    try:
                        self._connections[thread_id].close()
                    except Exception:
                        pass
                    del self._connections[thread_id]
                    del self._connection_last_used[thread_id]
                    logger.debug(f"关闭并删除超时连接 (线程ID: {thread_id})")
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        with self._cache_lock:
            expired_keys = [k for k, (_, timestamp) in self._query_cache.items() 
                           if current_time - timestamp > self._cache_timeout]
            for key in expired_keys:
                del self._query_cache[key]
            if expired_keys:
                logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def __del__(self):
        """析构函数，关闭所有连接"""
        try:
            self._close_all_connections()
        except Exception:
            pass
    
    def _close_all_connections(self):
        """关闭所有数据库连接"""
        with self._connection_lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            self._connection_last_used.clear()
            logger.debug("已关闭所有数据库连接")
    
    def _get_connection(self):
        """
        获取数据库连接，如果连接不存在或已关闭则创建新连接
        
        Returns:
            sqlite3.Connection: 数据库连接对象
        """
        thread_id = threading.get_ident()
        
        with self._connection_lock:
            # 检查当前线程是否已有连接
            if thread_id in self._connections:
                conn = self._connections[thread_id]
                try:
                    # 测试连接是否有效
                    conn.execute("SELECT 1")
                    # 更新最后使用时间
                    self._connection_last_used[thread_id] = time.time()
                    return conn
                except Exception:
                    # 连接无效，删除并创建新连接
                    del self._connections[thread_id]
            
            # 创建新连接
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # 启用外键约束
            conn.execute("PRAGMA foreign_keys = ON")
            # 配置WAL模式提高并发写入性能
            conn.execute("PRAGMA journal_mode = WAL")
            # 提高查询性能
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 30000000000")
            
            # 注册适配器和转换器
            sqlite3.register_adapter(np.int64, lambda val: int(val))
            sqlite3.register_adapter(np.float64, lambda val: float(val))
            sqlite3.register_adapter(pd.Timestamp, lambda val: val.isoformat())
            
            # 存储连接并记录最后使用时间
            self._connections[thread_id] = conn
            self._connection_last_used[thread_id] = time.time()
            
            return conn
    
    def _get_table_name(self, data_type: DataType, symbol: Optional[str] = None) -> str:
        """
        根据数据类型和股票代码生成表名
        
        Args:
            data_type (DataType): 数据类型
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            str: 表名
        """
        if data_type == DataType.SYMBOL:
            return "symbols"
        elif data_type == DataType.INDEX:
            return "indices"
        elif symbol:
            if data_type == DataType.HISTORICAL:
                return f"historical_{symbol}"
            elif data_type == DataType.REALTIME:
                return f"realtime_{symbol}"
        
        # 默认表名
        return f"{data_type.value}"
    
    def _get_data_type_schema(self, data_type: DataType) -> Dict[str, Tuple[str, bool]]:
        """
        获取数据类型对应的表结构定义
        
        Args:
            data_type (DataType): 数据类型
        
        Returns:
            Dict[str, Tuple[str, bool]]: 列名到(SQLite类型,是否为主键)的映射
        """
        # 将字符串类型的data_type转换为DataType枚举
        if isinstance(data_type, str):
            try:
                data_type = DataType(data_type)
            except (ValueError, TypeError):
                # 如果无法转换，则保持原样
                pass

        # 获取data_type的值，无论它是字符串还是DataType枚举
        data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
        
        if data_type_value == "symbol" or (isinstance(data_type, DataType) and data_type == DataType.SYMBOL):
            return {
                StandardColumns.SYMBOL.value: ("TEXT", True),
                StandardColumns.NAME.value: ("TEXT", False),
                StandardColumns.MARKET.value: ("TEXT", False),
                StandardColumns.INDUSTRY.value: ("TEXT", False),
                "updated_at": ("TIMESTAMP", False)
            }
        elif data_type_value == "historical" or (isinstance(data_type, DataType) and data_type == DataType.HISTORICAL):
            return {
                StandardColumns.DATE.value: ("DATE", True),
                StandardColumns.OPEN.value: ("REAL", False),
                StandardColumns.HIGH.value: ("REAL", False),
                StandardColumns.LOW.value: ("REAL", False),
                StandardColumns.CLOSE.value: ("REAL", False),
                StandardColumns.VOLUME.value: ("INTEGER", False),
                StandardColumns.AMOUNT.value: ("REAL", False),  # 直接使用标准列名而不是映射
                STANDARD_TO_STORAGE_MAPPING.get(StandardColumns.CHANGE_PCT.value, "change_pct"): ("REAL", False)
            }
        elif data_type_value == "realtime" or (isinstance(data_type, DataType) and data_type == DataType.REALTIME):
            return {
                StandardColumns.TIMESTAMP.value: ("TIMESTAMP", True),
                StandardColumns.PRICE.value: ("REAL", False),
                StandardColumns.VOLUME.value: ("INTEGER", False),
                StandardColumns.BID_PRICE.value: ("REAL", False),
                StandardColumns.ASK_PRICE.value: ("REAL", False),
                StandardColumns.BID_VOLUME.value: ("INTEGER", False),
                StandardColumns.ASK_VOLUME.value: ("INTEGER", False),
                STANDARD_TO_STORAGE_MAPPING.get(StandardColumns.CHANGE_PCT.value, "change_pct"): ("REAL", False)
            }
        elif data_type_value == "index" or (isinstance(data_type, DataType) and data_type == DataType.INDEX):
            return {
                StandardColumns.DATE.value: ("DATE", True),
                StandardColumns.OPEN.value: ("REAL", False),
                StandardColumns.HIGH.value: ("REAL", False),
                StandardColumns.LOW.value: ("REAL", False),
                StandardColumns.CLOSE.value: ("REAL", False),
                StandardColumns.VOLUME.value: ("INTEGER", False),
                StandardColumns.AMOUNT.value: ("REAL", False),  # 添加amount列
                STANDARD_TO_STORAGE_MAPPING.get(StandardColumns.CHANGE_PCT.value, "change_pct"): ("REAL", False)
            }
        else:
            error_msg = f"不支持的数据类型: {data_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _add_table_indexes(self, table_name: str, columns: List[str]) -> None:
        """
        为表添加索引
        
        Args:
            table_name: 表名
            columns: 需要添加索引的列名列表
        """
        logger.debug(f"为表 {table_name} 添加索引，列：{columns}")
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for col in columns:
                index_name = f"idx_{table_name}_{col}"
                query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col})"
                cursor.execute(query)
                logger.debug(f"创建索引：{index_name}")
            
            conn.commit()
            logger.info(f"成功为表 {table_name} 添加索引")
            
        except Exception as e:
            logger.error(f"为表 {table_name} 添加索引失败：{e}")
            raise
    
    def _create_tables_if_not_exists(self) -> None:
        """
        如果表不存在，则创建所有必要的表
        """
        conn = self._get_connection()
        
        # 创建符号表
        symbol_schema = self._get_data_type_schema(DataType.SYMBOL)
        schema_parts = []
        primary_keys = []
        
        for col, (col_type, is_primary) in symbol_schema.items():
            schema_parts.append(f"{col} {col_type}")
            if is_primary:
                primary_keys.append(col)
        
        if primary_keys:
            schema_parts.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS symbols (
            {', '.join(schema_parts)}
        )
        """
        conn.execute(create_table_sql)
        
        # 为symbols表添加索引
        self._add_table_indexes("symbols", ["market", "industry"])
        
        # 创建指数表
        index_schema = self._get_data_type_schema(DataType.INDEX)
        schema_parts = []
        primary_keys = []
        
        for col, (col_type, is_primary) in index_schema.items():
            schema_parts.append(f"{col} {col_type}")
            if is_primary:
                primary_keys.append(col)
        
        # 指数表需要symbol和date作为联合主键，但date已在schema中标记为主键
        # 移除schema_parts中可能已添加的PRIMARY KEY定义，以避免重复定义
        schema_parts = [part for part in schema_parts if not part.startswith("PRIMARY KEY")]
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS indices (
            symbol TEXT,
            {', '.join(schema_parts)},
            PRIMARY KEY (symbol, date)
        )
        """
        conn.execute(create_table_sql)
        
        # 为indices表添加索引
        self._add_table_indexes("indices", ["date"])
        
        conn.commit()
    
    def _execute_query(self, query: str, params: tuple = ()) -> Any:
        """
        执行SQL查询
        
        Args:
            query (str): SQL查询语句
            params (tuple, optional): 查询参数. Defaults to ().
        
        Returns:
            Any: 查询结果
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            logger.error(f"执行SQL查询失败: {e}, 查询: {query}, 参数: {params}")
            conn.rollback()
            raise
    
    def add_date_range_index_to_historical_tables(self) -> None:
        """
        为所有历史数据表的date字段创建范围索引
        用于优化日期范围查询
        """
        logger.info("开始为历史数据表添加日期范围索引")
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 获取所有历史数据表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND "
                          "(name LIKE 'historical_stock_%' OR name LIKE 'historical_index_%')")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                index_name = f"idx_{table_name}_date_range"
                
                # 创建日期范围索引
                query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}(date)"
                cursor.execute(query)
                logger.debug(f"为表 {table_name} 创建日期范围索引：{index_name}")
            
            conn.commit()
            logger.info(f"成功为 {len(tables)} 个历史数据表添加日期范围索引")
            
        except Exception as e:
            logger.error(f"为历史数据表添加日期范围索引失败：{e}")
            raise
    
    def add_time_date_index_to_realtime_tables(self) -> None:
        """
        为所有实时数据表创建time和date的复合索引
        用于优化时间范围查询
        """
        logger.info("开始为实时数据表添加时间和日期的复合索引")
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 获取所有实时数据表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND "
                          "(name LIKE 'realtime_stock_%' OR name LIKE 'realtime_index_%')")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                index_name = f"idx_{table_name}_timestamp"
                
                # 创建时间戳索引
                query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({StandardColumns.TIMESTAMP.value})"
                cursor.execute(query)
                logger.debug(f"为表 {table_name} 创建时间戳索引：{index_name}")
            
            conn.commit()
            logger.info(f"成功为 {len(tables)} 个实时数据表添加时间戳索引")
            
        except Exception as e:
            logger.error(f"为实时数据表添加时间戳索引失败：{e}")
            raise

    def optimize_indexes(self) -> None:
        """
        优化数据库索引
        1. 为历史数据表添加日期范围索引
        2. 为实时数据表添加时间和日期的复合索引
        """
        logger.info("开始优化数据库索引...")
        
        try:
            # 添加日期范围索引
            self.add_date_range_index_to_historical_tables()
            
            # 添加时间和日期的复合索引
            self.add_time_date_index_to_realtime_tables()
            
            logger.info("数据库索引优化完成")
            
        except Exception as e:
            logger.error(f"优化数据库索引失败：{e}")
            raise

    def _optimize_database(self) -> None:
        """
        优化数据库性能
        - 执行VACUUM操作释放空间
        - 优化索引
        """
        logger.info("开始优化数据库...")
        
        try:
            conn = self._get_connection()
            
            # 分析数据库以优化查询计划
            conn.execute("PRAGMA analysis_limit=1000")
            conn.execute("PRAGMA optimize")
            
            # 执行VACUUM操作
            logger.info("执行VACUUM操作...")
            conn.execute("VACUUM")
            
            # 开启WAL模式提高写入性能
            logger.info("设置数据库PRAGMA参数...")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")  # 约40MB缓存
            conn.execute("PRAGMA temp_store = MEMORY")
            
            # 优化索引
            self.optimize_indexes()
            
            logger.info("数据库优化完成")
            
        except Exception as e:
            logger.error(f"优化数据库失败：{e}")
            raise
    
    @log_function_call(level='DEBUG')
    def save_data(self, data: pd.DataFrame, data_type: DataType, symbol: Optional[str] = None) -> bool:
        """
        保存数据到SQLite数据库
        
        Args:
            data (pd.DataFrame): 要保存的数据
            data_type (DataType): 数据类型
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            bool: 是否保存成功
        
        Raises:
            DataSaveException: 保存数据失败时抛出
        """
        if data.empty:
            logger.warning("尝试保存空数据，操作被跳过")
            return False
        
        # 确保data_type是DataType枚举类型
        try:
            if isinstance(data_type, str):
                try:
                    # 尝试将字符串转换为DataType枚举
                    data_type = DataType(data_type)
                except ValueError:
                    error_msg = f"无效的数据类型字符串: {data_type}"
                    logger.error(error_msg)
                    raise DataSaveException(error_msg, storage_type="sqlite")
            elif not isinstance(data_type, DataType) and hasattr(data_type, 'value'):
                # 可能是从其他模块导入的枚举类型，尝试通过值匹配
                data_type_value = data_type.value
                try:
                    data_type = DataType(data_type_value)
                except ValueError:
                    logger.warning(f"无法直接转换 {data_type} 到 DataType 枚举，将使用值匹配")
                    # 使用值进行匹配
                    if data_type_value == "historical":
                        data_type = DataType.HISTORICAL
                    elif data_type_value == "realtime":
                        data_type = DataType.REALTIME
                    elif data_type_value == "symbol":
                        data_type = DataType.SYMBOL
                    elif data_type_value == "index":
                        data_type = DataType.INDEX
                    else:
                        error_msg = f"未知的数据类型值: {data_type_value}"
                        logger.error(error_msg)
                        raise DataSaveException(error_msg, storage_type="sqlite")
        except Exception as e:
            error_msg = f"处理数据类型时出错: {e}"
            logger.error(error_msg)
            raise DataSaveException(error_msg, storage_type="sqlite", original_error=e)
        
        conn = None
        try:
            conn = self._get_connection()
            
            # 对于股票和指数数据，直接保存到对应表
            if data_type == DataType.SYMBOL:
                self._save_symbol_data(conn, data)
                
            elif data_type == DataType.INDEX:
                self._save_index_data(conn, data)
                
            # 对于历史和实时数据，需要为每个股票创建独立的表
            elif symbol:
                success = self._save_stock_data(conn, data, data_type, symbol)
                if not success:
                    # 如果保存失败但没有异常，返回失败
                    return False
            else:
                error_msg = f"保存 {data_type.value if hasattr(data_type, 'value') else str(data_type)} 数据失败: 必须提供symbol参数"
                logger.error(error_msg)
                raise DataSaveException(error_msg, storage_type="sqlite", data_type=data_type)
            
            conn.commit()
            
            # 清除相关的查询缓存
            self._clear_related_cache(data_type, symbol)
            
            return True
            
        except sqlite3.Error as e:
            error_msg = f"SQLite错误: {e}"
            logger.error(error_msg)
            if conn:
                conn.rollback()
            # 记录详细的SQL错误信息
            logger.debug(f"SQL错误详情: {traceback.format_exc()}")
            raise DataSaveException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
        except ValueError as e:
            error_msg = f"值错误: {e}"
            logger.error(error_msg)
            if conn:
                conn.rollback()
            raise DataSaveException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
        except Exception as e:
            error_msg = f"保存数据失败: {e}"
            logger.error(error_msg)
            logger.debug(f"异常详情: {traceback.format_exc()}")
            if conn:
                conn.rollback()
            raise DataSaveException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
    
    def _save_symbol_data(self, conn, data):
        """
        保存股票代码数据
        
        Args:
            conn: 数据库连接
            data: 要保存的数据
        
        Raises:
            DataSaveException: 保存失败时抛出
        """
        try:
            # 确保dataframe包含必要的列
            required_columns = list(self._get_data_type_schema(DataType.SYMBOL).keys())
            for col in required_columns:
                if col not in data.columns and col != 'updated_at':
                    data[col] = None
            
            # 添加更新时间
            data['updated_at'] = datetime.now()
            
            # 使用executemany进行批量插入
            insert_columns = ', '.join(required_columns)
            placeholders = ', '.join(['?' for _ in required_columns])
            insert_sql = f"INSERT OR REPLACE INTO symbols ({insert_columns}) VALUES ({placeholders})"
            
            # 准备插入数据
            rows = []
            for _, row in data.iterrows():
                rows.append(tuple(row[col] for col in required_columns))
            
            # 执行批量插入
            cursor = conn.cursor()
            cursor.executemany(insert_sql, rows)
            conn.commit()
            
            logger.info(f"成功保存 {len(data)} 条股票代码数据")
        except Exception as e:
            error_msg = f"保存股票代码数据失败: {e}"
            logger.error(error_msg)
            raise DataSaveException(error_msg, storage_type="sqlite", data_type=DataType.SYMBOL, original_error=e)
    
    def _save_index_data(self, conn, data):
        """
        保存指数数据
        
        Args:
            conn: 数据库连接
            data: 要保存的数据
        
        Raises:
            DataSaveException: 保存失败时抛出
        """
        try:
            if 'symbol' not in data.columns:
                raise ValueError("指数数据必须包含symbol列")
            
            # 确保包含所需列
            required_columns = ['symbol'] + list(self._get_data_type_schema(DataType.INDEX).keys())
            for col in required_columns:
                if col not in data.columns:
                    data[col] = None
            
            # 使用executemany进行批量插入
            insert_columns = ', '.join(required_columns)
            placeholders = ', '.join(['?' for _ in required_columns])
            insert_sql = f"INSERT OR REPLACE INTO indices ({insert_columns}) VALUES ({placeholders})"
            
            # 准备插入数据
            rows = []
            for _, row in data.iterrows():
                rows.append(tuple(row[col] for col in required_columns))
            
            # 执行批量插入
            cursor = conn.cursor()
            cursor.executemany(insert_sql, rows)
            conn.commit()
            
            logger.info(f"成功保存 {len(data)} 条指数数据")
        except Exception as e:
            error_msg = f"保存指数数据失败: {e}"
            logger.error(error_msg)
            raise DataSaveException(error_msg, storage_type="sqlite", data_type=DataType.INDEX, original_error=e)
    
    def _save_stock_data(self, conn, data, data_type, symbol):
        """
        保存股票数据（历史或实时）
        
        Args:
            conn: 数据库连接
            data: 要保存的数据
            data_type: 数据类型
            symbol: 股票代码
        
        Raises:
            DataSaveException: 保存失败时抛出
        """
        try:
            table_name = self._get_table_name(data_type, symbol)
            
            # 检查表是否存在，如果不存在则创建
            schema = self._get_data_type_schema(data_type)
            schema_parts = []
            primary_keys = []
            
            for col, (col_type, is_primary) in schema.items():
                schema_parts.append(f"{col} {col_type}")
                if is_primary:
                    primary_keys.append(col)
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(schema_parts)}
                {', PRIMARY KEY (' + ', '.join(primary_keys) + ')' if primary_keys else ''}
            )
            """
            conn.execute(create_table_sql)
            
            # 为表添加索引
            if data_type == DataType.HISTORICAL or data_type.value == "historical" if hasattr(data_type, 'value') else str(data_type) == "historical":
                self._add_table_indexes(table_name, [StandardColumns.DATE.value])
            elif data_type == DataType.REALTIME or data_type.value == "realtime" if hasattr(data_type, 'value') else str(data_type) == "realtime":
                self._add_table_indexes(table_name, [StandardColumns.TIMESTAMP.value])
            
            # 记录原始列
            original_columns = list(data.columns)
            logger.debug(f"原始数据列: {original_columns}")
            
            # 标准化列名 - 提供一个更安全的过程
            try:
                data = standardize_columns(data, source_type="A_SHARE", logger=logger)
                logger.debug(f"标准化后的列: {list(data.columns)}")
            except Exception as e:
                logger.warning(f"列名标准化失败: {e}，将使用原始列名")
            
            # 创建标准列名到存储列名的映射
            column_mappings = {}
            for std_col, storage_col in STANDARD_TO_STORAGE_MAPPING.items():
                if std_col in data.columns and storage_col not in data.columns:
                    column_mappings[std_col] = storage_col
            
            # 应用映射
            renamed_columns = {}
            if column_mappings:
                for std_col, storage_col in column_mappings.items():
                    data[storage_col] = data[std_col]
                    renamed_columns[std_col] = storage_col
                logger.info(f"已应用存储映射: {renamed_columns}")
            
            # 获取表中实际存在的列
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            logger.debug(f"表 {table_name} 中的现有列: {existing_columns}")
            
            # 检查是否需要添加新列
            for col in schema.keys():
                if col not in existing_columns:
                    col_type = schema[col][0]
                    try:
                        logger.info(f"向表 {table_name} 添加新列: {col} {col_type}")
                        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
                    except Exception as e:
                        logger.warning(f"添加列 {col} 失败: {e}")
            
            # 确保data包含schema中定义的amount列
            if "amount" not in data.columns and StandardColumns.AMOUNT.value in schema:
                logger.warning(f"数据中缺少amount列，将添加空列")
                data[StandardColumns.AMOUNT.value] = None
            
            # 重新获取表中实际存在的列，确保获取最新列表
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            # 确保数据包含所需列
            required_columns = [col for col in schema.keys() if col in existing_columns]
            logger.debug(f"准备插入的列: {required_columns}")
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"数据缺少必要列: {missing_columns}，将使用null值填充")
                for col in missing_columns:
                    data[col] = None
            
            # 检查数据类型并转换
            for col in required_columns:
                col_type = schema[col][0].split()[0].upper()
                if col in data.columns:
                    # 尝试转换数据类型
                    try:
                        if col_type in ('INTEGER', 'INT'):
                            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int64')
                        elif col_type in ('REAL', 'FLOAT'):
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        elif col_type.startswith('TEXT') and 'date' in col:
                            # 处理日期字段
                            if not pd.api.types.is_datetime64_any_dtype(data[col]):
                                data[col] = pd.to_datetime(data[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"转换列 {col} 的数据类型时出错: {e}")
            
            # 使用executemany进行批量插入，只使用表中存在的列
            # 过滤数据列，只使用实际存在于表中的列
            insert_columns = [col for col in required_columns if col in data.columns]
            
            if not insert_columns:
                logger.error(f"没有可插入的列")
                return False
                
            placeholders = ', '.join(['?' for _ in insert_columns])
            insert_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(insert_columns)}) VALUES ({placeholders})"
            
            # 准备插入数据
            rows = []
            for _, row in data.iterrows():
                row_data = []
                for col in insert_columns:
                    val = row.get(col)
                    # 格式化日期时间对象
                    if isinstance(val, (datetime, date)):
                        val = val.isoformat()
                    row_data.append(val)
                rows.append(tuple(row_data))
            
            # 执行批量插入
            cursor = conn.cursor()
            cursor.executemany(insert_sql, rows)
            
            logger.info(f"成功保存 {len(data)} 条 {symbol} 的 {data_type.value if hasattr(data_type, 'value') else str(data_type)} 数据")
            return True
            
        except Exception as e:
            error_msg = f"保存股票数据失败: {e}"
            logger.error(error_msg)
            # 记录详细错误信息以便调试
            logger.debug(f"错误详情: {traceback.format_exc()}")
            raise DataSaveException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
    
    def _clear_related_cache(self, data_type: DataType, symbol: Optional[str] = None) -> None:
        """
        清除与特定数据类型和股票相关的缓存
        
        Args:
            data_type (DataType): 数据类型
            symbol (str, optional): 股票代码. Defaults to None.
        """
        # 获取data_type的值
        data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
        
        with self._cache_lock:
            keys_to_delete = []
            
            # 查找与指定数据类型和股票相关的缓存项
            for key in self._query_cache.keys():
                cache_parts = key.split(':', 2)
                if len(cache_parts) >= 2:
                    cache_type, cache_symbol = cache_parts[0], cache_parts[1]
                    if cache_type == data_type_value and (symbol is None or cache_symbol == symbol):
                        keys_to_delete.append(key)
            
            # 删除相关缓存项
            for key in keys_to_delete:
                del self._query_cache[key]
            
            if keys_to_delete:
                logger.debug(f"已清除 {len(keys_to_delete)} 个相关缓存项")
    
    @log_function_call(level='DEBUG')
    def load_data(self, data_type: DataType, symbol: Optional[str] = None, 
                 start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
        """
        从SQLite数据库加载数据
        
        Args:
            data_type (DataType): 数据类型
            symbol (str, optional): 股票代码. Defaults to None.
            start_date (date, optional): 开始日期. Defaults to None.
            end_date (date, optional): 结束日期. Defaults to None.
        
        Returns:
            pd.DataFrame: 加载的数据
            
        Raises:
            DataLoadException: 加载数据失败时抛出
        """
        # 确保data_type是DataType枚举类型
        if isinstance(data_type, str):
            try:
                # 尝试将字符串转换为DataType枚举
                data_type = DataType(data_type)
            except ValueError:
                error_msg = f"无效的数据类型: {data_type}"
                logger.error(error_msg)
                raise DataLoadException(error_msg, storage_type="sqlite")
        
        # 生成缓存键
        data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
        cache_key = f"{data_type_value}:{symbol or 'all'}:{start_date}:{end_date}"
        
        # 检查缓存
        with self._cache_lock:
            if cache_key in self._query_cache:
                data, timestamp = self._query_cache[cache_key]
                # 检查缓存是否过期
                if time.time() - timestamp <= self._cache_timeout:
                    logger.debug(f"使用缓存数据: {cache_key}")
                    return data.copy()  # 返回数据副本以避免修改缓存
        
        # 使用异常日志上下文管理器
        with log_exception(logger, reraise=False, level='ERROR') as exc_log:
            try:
                conn = self._get_connection()
                
                if data_type == DataType.SYMBOL:
                    result = self._load_symbol_data(conn)
                elif data_type == DataType.INDEX:
                    result = self._load_index_data(conn, symbol)
                else:
                    if not symbol:
                        error_msg = f"加载 {data_type.value} 数据时必须提供symbol参数"
                        logger.error(error_msg)
                        raise DataLoadException(error_msg, storage_type="sqlite", data_type=data_type)
                    
                    result = self._load_stock_data(conn, data_type, symbol, start_date, end_date)
                
                # 缓存结果
                if not result.empty:
                    with self._cache_lock:
                        self._query_cache[cache_key] = (result.copy(), time.time())
                
                return result
                
            except sqlite3.Error as e:
                error_msg = f"SQLite错误: {e}"
                logger.error(error_msg)
                raise DataLoadException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
            except Exception as e:
                error_msg = f"加载数据失败: {e}"
                logger.error(error_msg)
                logger.debug(f"异常详情: {traceback.format_exc()}")
                raise DataLoadException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
        
        # 如果有异常，返回空DataFrame
        return pd.DataFrame()
        
    def _load_symbol_data(self, conn):
        """加载股票代码数据"""
        query = "SELECT * FROM symbols"
        try:
            # 直接使用pandas从sqlite读取
            df = pd.read_sql_query(query, conn)
            logger.info(f"成功加载 {len(df)} 条股票代码数据")
            return df
        except Exception as e:
            error_msg = f"加载股票代码数据失败: {e}"
            logger.error(error_msg)
            raise DataLoadException(error_msg, storage_type="sqlite", data_type=DataType.SYMBOL, original_error=e)
    
    def _load_index_data(self, conn, symbol):
        """加载指数数据"""
        try:
            if symbol:
                query = "SELECT * FROM indices WHERE symbol = ?"
                params = [symbol]
            else:
                query = "SELECT * FROM indices"
                params = []
            
            df = pd.read_sql_query(query, conn, params=params)
            symbol_info = f" {symbol}" if symbol else ""
            logger.info(f"成功加载{symbol_info} {len(df)} 条指数数据")
            return df
        except Exception as e:
            error_msg = f"加载指数数据失败: {e}"
            logger.error(error_msg)
            raise DataLoadException(error_msg, storage_type="sqlite", data_type=DataType.INDEX, symbol=symbol, original_error=e)
    
    def _load_stock_data(self, conn, data_type, symbol, start_date, end_date):
        """加载股票数据（历史或实时）"""
        try:
            table_name = self._get_table_name(data_type, symbol)
            
            # 检查表是否存在
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                (table_name,)
            ).fetchone()
            
            if not table_check:
                logger.warning(f"表 {table_name} 不存在")
                return pd.DataFrame()
            
            # 构建查询
            query = f"SELECT * FROM {table_name}"
            params = []
            
            date_col = StandardColumns.DATE.value if data_type == DataType.HISTORICAL else StandardColumns.TIMESTAMP.value
            
            # 添加日期过滤
            if start_date or end_date:
                filters = []
                
                if start_date:
                    filters.append(f"{date_col} >= ?")
                    if isinstance(start_date, date):
                        start_date = start_date.isoformat()
                    params.append(start_date)
                
                if end_date:
                    filters.append(f"{date_col} <= ?")
                    if isinstance(end_date, date):
                        end_date = end_date.isoformat()
                    params.append(end_date)
                
                if filters:
                    query += " WHERE " + " AND ".join(filters)
            
            # 按日期排序
            query += f" ORDER BY {date_col} ASC"
            
            # 执行查询
            df = pd.read_sql_query(query, conn, params=params)
            
            # 记录原始列
            original_columns = list(df.columns)
            logger.debug(f"从数据库加载的原始列: {original_columns}")
            
            # 转换日期列
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            
            # 应用反向映射，将存储列名映射回标准列名
            df = apply_reverse_mapping_for_analysis(df)
            
            # 记录处理后的列
            logger.debug(f"应用映射后的列: {list(df.columns)}")
            
            # 记录获取到的数据行数
            if not df.empty:
                logger.info(f"成功加载 {symbol} 的 {len(df)} 条 从 {start_date} 到 {end_date} {data_type.name.lower()} 数据")
            
            return df
        
        except Exception as e:
            error_msg = f"加载股票数据失败: {e}"
            logger.error(error_msg)
            raise DataLoadException(error_msg, storage_type="sqlite", data_type=data_type, symbol=symbol, original_error=e)
    
    def delete_data(self, data_type: DataType, symbol: Optional[str] = None) -> bool:
        """
        从SQLite数据库删除数据
        
        Args:
            data_type (DataType): 数据类型
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            bool: 是否删除成功
        """
        # 确保data_type是DataType枚举类型
        if isinstance(data_type, str):
            try:
                # 尝试将字符串转换为DataType枚举
                data_type = DataType(data_type)
            except ValueError:
                error_msg = f"无效的数据类型: {data_type}"
                logger.error(error_msg)
                return False
        
        try:
            conn = self._get_connection()
            
            # 处理股票代码数据
            if data_type == DataType.SYMBOL:
                if symbol:
                    query = "DELETE FROM symbols WHERE symbol = ?"
                    self._execute_query(query, (symbol,))
                    logger.info(f"删除了股票 {symbol} 的代码数据")
                else:
                    query = "DELETE FROM symbols"
                    self._execute_query(query)
                    logger.info("删除了所有股票代码数据")
            
            # 处理指数数据
            elif data_type == DataType.INDEX:
                if symbol:
                    query = "DELETE FROM indices WHERE symbol = ?"
                    self._execute_query(query, (symbol,))
                    logger.info(f"删除了指数 {symbol} 的数据")
                else:
                    query = "DELETE FROM indices"
                    self._execute_query(query)
                    logger.info("删除了所有指数数据")
            
            # 处理历史和实时数据
            elif symbol:
                table_name = self._get_table_name(data_type, symbol)
                
                # 检查表是否存在
                check_query = f"""
                SELECT name FROM sqlite_master WHERE type='table' AND name=?
                """
                cursor = conn.execute(check_query, (table_name,))
                if cursor.fetchone():
                    # 删除表
                    query = f"DROP TABLE {table_name}"
                    self._execute_query(query)
                    logger.info(f"删除了 {symbol} 的 {data_type.value} 数据表")
                else:
                    logger.warning(f"表 {table_name} 不存在，无法删除")
            else:
                data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
                logger.error(f"删除 {data_type_value} 数据失败: 必须提供symbol参数")
                return False
            
            conn.commit()
            
            # 清除相关的查询缓存
            self._clear_related_cache(data_type, symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"删除数据失败: {e}")
            if conn:
                conn.rollback()
            return False
    
    def exists(self, data_type: DataType, symbol: Optional[str] = None) -> bool:
        """
        检查数据是否存在
        
        Args:
            data_type (DataType): 数据类型
            symbol (str, optional): 股票代码. Defaults to None.
        
        Returns:
            bool: 数据是否存在
        """
        # 确保data_type是DataType枚举类型
        if isinstance(data_type, str):
            try:
                # 尝试将字符串转换为DataType枚举
                data_type = DataType(data_type)
            except ValueError:
                error_msg = f"无效的数据类型: {data_type}"
                logger.error(error_msg)
                return False
                
        try:
            conn = self._get_connection()
            
            # 处理股票代码数据
            if data_type == DataType.SYMBOL:
                query = "SELECT 1 FROM symbols"
                if symbol:
                    query += " WHERE symbol = ?"
                    cursor = conn.execute(query, (symbol,))
                else:
                    cursor = conn.execute(query)
                return cursor.fetchone() is not None
            
            # 处理指数数据
            elif data_type == DataType.INDEX:
                query = "SELECT 1 FROM indices"
                if symbol:
                    query += " WHERE symbol = ?"
                    cursor = conn.execute(query, (symbol,))
                else:
                    cursor = conn.execute(query)
                return cursor.fetchone() is not None
            
            # 处理历史和实时数据
            elif symbol:
                table_name = self._get_table_name(data_type, symbol)
                
                # 检查表是否存在
                check_query = """
                SELECT name FROM sqlite_master WHERE type='table' AND name=?
                """
                cursor = conn.execute(check_query, (table_name,))
                return cursor.fetchone() is not None
            else:
                data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
                logger.error(f"检查 {data_type_value} 数据是否存在失败: 必须提供symbol参数")
                return False
            
        except Exception as e:
            logger.error(f"检查数据是否存在失败: {e}")
            return False 