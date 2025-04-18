# -*- coding: utf-8 -*-
"""
PostgreSQL/TimescaleDB存储实现
用于将数据存储到PostgreSQL/TimescaleDB数据库
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, MetaData, Table, select, and_, or_, text
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from ..utils.exceptions import StorageException
from .base import Storage, DataType

# 定义TIMESTAMPTZ为DateTime，并设置timezone=True
from sqlalchemy import DateTime as SQLAlchemyDateTime
TIMESTAMPTZ = lambda: SQLAlchemyDateTime(timezone=True)

logger = logging.getLogger(__name__)
Base = declarative_base()

class PostgreSQLStorage(Storage):
    """PostgreSQL/TimescaleDB存储实现"""
    
    def __init__(self, config):
        """初始化PostgreSQL存储
        
        Args:
            config: 包含连接信息的配置字典
        """
        super().__init__("postgresql", config)
        self.connection_url = f"postgresql://{config['connection']['user']}:{config['connection']['password']}@{config['connection']['host']}:{config['connection']['port']}/{config['connection']['database']}"
        
        # 创建数据库引擎
        self.engine = create_engine(
            self.connection_url,
            pool_size=config.get('pool_size', 5),
            max_overflow=config.get('max_overflow', 10),
            pool_recycle=config.get('pool_recycle', 3600),
            echo=config.get('echo', False)
        )
        
        # 创建会话工厂
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
        # 表映射
        self._table_mappings = {
            DataType.HISTORICAL: "stock_historical_data",
            DataType.REALTIME: "stock_realtime_data",
            DataType.SYMBOL: "stocks",
            DataType.INDEX: "index_data"
        }
        
        logger.info(f"PostgreSQL存储已初始化，连接到：{config['connection']['host']}:{config['connection']['port']}/{config['connection']['database']}")
    
    def _get_table_name(self, data_type):
        """获取数据类型对应的表名
        
        Args:
            data_type: 数据类型枚举
            
        Returns:
            表名字符串
        """
        if data_type not in self._table_mappings:
            raise StorageException(f"未知的数据类型: {data_type}")
        return self._table_mappings[data_type]
    
    def save_data(self, data, data_type, symbol=None, mode='append', **kwargs):
        """保存数据到PostgreSQL/TimescaleDB
        
        Args:
            data: DataFrame数据
            data_type: 数据类型枚举
            symbol: 股票代码（可选）
            mode: 保存模式，'append'或'overwrite'
            
        Returns:
            bool: 保存是否成功
        """
        if data is None or data.empty:
            logger.warning("尝试保存空数据")
            return False
        
        table_name = self._get_table_name(data_type)
        
        try:
            if symbol and 'symbol' not in data.columns:
                data['symbol'] = symbol
            
            # 处理索引和时间格式
            if data_type in [DataType.HISTORICAL, DataType.REALTIME, DataType.INDEX]:
                # 转换时间列
                if 'date' in data.columns:
                    data = data.rename(columns={'date': 'time'})
                if 'time' not in data.columns and data.index.name != 'time':
                    if isinstance(data.index, pd.DatetimeIndex):
                        data = data.reset_index().rename(columns={'index': 'time'})
                    else:
                        logger.warning(f"数据没有时间列或索引不是DatetimeIndex: {data.columns}")
                
                # 确保时间列是datetime类型
                if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
                    data['time'] = pd.to_datetime(data['time'])
            
            # 股票信息特殊处理
            if data_type == DataType.SYMBOL:
                data['created_at'] = datetime.now()
                data['updated_at'] = datetime.now()
                
                # 检查股票是否已存在
                with self.Session() as session:
                    for _, row in data.iterrows():
                        result = session.execute(
                            text(f"SELECT 1 FROM {table_name} WHERE symbol = :symbol"),
                            {"symbol": row['symbol']}
                        ).fetchone()
                        
                        if result:
                            # 更新现有记录
                            update_stmt = text(f"UPDATE {table_name} SET name = :name, exchange = :exchange, "
                                              f"sector = :sector, industry = :industry, updated_at = :updated_at "
                                              f"WHERE symbol = :symbol")
                            session.execute(update_stmt, {
                                "name": row['name'],
                                "exchange": row['exchange'],
                                "sector": row.get('sector', None),
                                "industry": row.get('industry', None),
                                "updated_at": datetime.now(),
                                "symbol": row['symbol']
                            })
                        else:
                            # 插入新记录
                            insert_columns = ['symbol', 'name', 'exchange']
                            insert_values = [row['symbol'], row['name'], row['exchange']]
                            
                            optional_columns = ['sector', 'industry', 'listing_date']
                            for col in optional_columns:
                                if col in row and pd.notna(row[col]):
                                    insert_columns.append(col)
                                    insert_values.append(row[col])
                            
                            # 添加时间戳
                            insert_columns.extend(['created_at', 'updated_at'])
                            insert_values.extend([datetime.now(), datetime.now()])
                            
                            cols_str = ", ".join(insert_columns)
                            vals_str = ", ".join([f":{col}" for col in insert_columns])
                            
                            insert_stmt = text(f"INSERT INTO {table_name} ({cols_str}) VALUES ({vals_str})")
                            params = {col: val for col, val in zip(insert_columns, insert_values)}
                            session.execute(insert_stmt, params)
                    
                    session.commit()
                return True
            
            # 其他数据类型使用pandas to_sql方法
            if_exists = 'replace' if mode == 'overwrite' else 'append'
            data.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"已保存 {len(data)} 条记录到表 {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到PostgreSQL出错: {str(e)}")
            raise StorageException(f"保存数据到PostgreSQL出错: {str(e)}")
    
    def load_data(self, data_type, symbol=None, start_date=None, end_date=None, limit=None, **kwargs):
        """从PostgreSQL/TimescaleDB加载数据
        
        Args:
            data_type: 数据类型枚举
            symbol: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            limit: 限制返回的记录数（可选）
            
        Returns:
            DataFrame: 加载的数据
        """
        table_name = self._get_table_name(data_type)
        
        try:
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            
            if data_type in [DataType.HISTORICAL, DataType.REALTIME, DataType.INDEX]:
                if start_date:
                    conditions.append(f"time >= '{start_date}'")
                if end_date:
                    conditions.append(f"time <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            if data_type in [DataType.HISTORICAL, DataType.REALTIME, DataType.INDEX]:
                query += " ORDER BY time DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            data = pd.read_sql(query, self.engine)
            logger.info(f"已从表 {table_name} 加载 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"从PostgreSQL加载数据出错: {str(e)}")
            raise StorageException(f"从PostgreSQL加载数据出错: {str(e)}")
    
    def delete_data(self, data_type, symbol=None, start_date=None, end_date=None, **kwargs):
        """从PostgreSQL/TimescaleDB删除数据
        
        Args:
            data_type: 数据类型枚举
            symbol: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            bool: 删除是否成功
        """
        table_name = self._get_table_name(data_type)
        
        try:
            query = f"DELETE FROM {table_name}"
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            
            if data_type in [DataType.HISTORICAL, DataType.REALTIME, DataType.INDEX]:
                if start_date:
                    conditions.append(f"time >= '{start_date}'")
                if end_date:
                    conditions.append(f"time <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                conn.commit()
                logger.info(f"已从表 {table_name} 删除 {result.rowcount} 条记录")
                return True
            
        except Exception as e:
            logger.error(f"从PostgreSQL删除数据出错: {str(e)}")
            raise StorageException(f"从PostgreSQL删除数据出错: {str(e)}")
    
    def exists(self, data_type, symbol=None, date=None, **kwargs):
        """检查数据是否存在
        
        Args:
            data_type: 数据类型枚举
            symbol: 股票代码（可选）
            date: 特定日期（可选）
            
        Returns:
            bool: 数据是否存在
        """
        table_name = self._get_table_name(data_type)
        
        try:
            query = f"SELECT 1 FROM {table_name}"
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            
            if data_type in [DataType.HISTORICAL, DataType.REALTIME, DataType.INDEX] and date:
                conditions.append(f"DATE(time) = '{date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " LIMIT 1"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                return result is not None
            
        except Exception as e:
            logger.error(f"检查PostgreSQL数据是否存在出错: {str(e)}")
            raise StorageException(f"检查PostgreSQL数据是否存在出错: {str(e)}")
    
    def get_latest_data(self, data_type, symbol=None, limit=1, **kwargs):
        """获取最新数据
        
        Args:
            data_type: 数据类型枚举
            symbol: 股票代码（可选）
            limit: 返回的记录数（默认为1）
            
        Returns:
            DataFrame: 最新数据
        """
        table_name = self._get_table_name(data_type)
        
        try:
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            if data_type in [DataType.HISTORICAL, DataType.REALTIME, DataType.INDEX]:
                query += " ORDER BY time DESC"
            
            query += f" LIMIT {limit}"
            
            data = pd.read_sql(query, self.engine)
            return data
            
        except Exception as e:
            logger.error(f"获取PostgreSQL最新数据出错: {str(e)}")
            raise StorageException(f"获取PostgreSQL最新数据出错: {str(e)}")
    
    def optimize(self, **kwargs):
        """优化数据库
        
        Returns:
            bool: 优化是否成功
        """
        try:
            # 对于 VACUUM 命令，需要在自动提交模式下执行
            with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                # 执行VACUUM ANALYZE
                conn.execute(text("VACUUM ANALYZE"))
                logger.info("已执行VACUUM ANALYZE优化")
                
                # 检查物化视图是否存在
                try:
                    # 检查daily_stock_metrics是否存在
                    exists_query = text("""
                    SELECT 1 FROM pg_matviews 
                    WHERE matviewname = 'daily_stock_metrics'
                    """)
                    result = conn.execute(exists_query).fetchone()
                    if result:
                        conn.execute(text("REFRESH MATERIALIZED VIEW daily_stock_metrics"))
                        logger.info("已刷新物化视图 daily_stock_metrics")
                    
                    # 检查weekly_stock_metrics是否存在
                    exists_query = text("""
                    SELECT 1 FROM pg_matviews 
                    WHERE matviewname = 'weekly_stock_metrics'
                    """)
                    result = conn.execute(exists_query).fetchone()
                    if result:
                        conn.execute(text("REFRESH MATERIALIZED VIEW weekly_stock_metrics"))
                        logger.info("已刷新物化视图 weekly_stock_metrics")
                    
                except Exception as e:
                    logger.warning(f"刷新物化视图失败: {e}")
                
                logger.info("数据库优化完成")
                return True
            
        except Exception as e:
            logger.error(f"优化PostgreSQL数据库出错: {str(e)}")
            raise StorageException(f"优化PostgreSQL数据库出错: {str(e)}")
    
    def get_status(self, **kwargs):
        """获取数据库状态
        
        Returns:
            dict: 数据库状态信息
        """
        try:
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
            
        except Exception as e:
            logger.error(f"获取PostgreSQL数据库状态出错: {str(e)}")
            raise StorageException(f"获取PostgreSQL数据库状态出错: {str(e)}") 