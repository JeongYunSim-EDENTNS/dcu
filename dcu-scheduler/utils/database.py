from peewee_async import PooledPostgresqlDatabase
from peewee import *
from config import config
from models.config import DatabaseConfig
from utils.logger import get_logger
import peewee_async
import oracledb
import asyncio
# Logger 설정
logger = get_logger(name="database", log_dir="logs")

def get_database(db_config: DatabaseConfig):
    """
    데이터베이스 연결 객체를 반환
    :param db_config: DatabaseConfig 객체           
    :return: Database 객체
    """
    class OracleDatabase(Database):
        def __init__(self, database, **kwargs):
            self.dsn = kwargs.pop('dsn')
            self.user = kwargs.pop('user')
            self.password = kwargs.pop('password')
            self._conn = None
            super().__init__(database or 'oracle', **kwargs)

        def _connect(self):
            return oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )

        async def connect_async(self):
            """비동기 연결 메서드"""
            if not self._conn:
                self._conn = self._connect()
            return self._conn

        async def close_async(self):
            """비동기 연결 종료 메서드"""
            if self._conn:
                self._conn.close()
                self._conn = None

        def close(self):
            """동기 연결 종료 메서드"""
            if self._conn:
                self._conn.close()
                self._conn = None

        async def aio_execute(self, query):
            """비동기 쿼리 실행 메서드"""
            conn = await self.connect_async()
            cursor = conn.cursor()
            try:
                # Oracle SQL 문법에 맞게 쿼리 변환
                query_str = str(query)
                if "RETURNING" in query_str.upper():
                    query_str = query_str.split("RETURNING")[0].strip()
                
                cursor.execute(query_str)
                results = cursor.fetchall()
                return results
            finally:
                cursor.close()

        def create_tables(self, models, **options):
            """Oracle용 테이블 생성 메서드 오버라이드"""
            for model in models:
                table_name = model._meta.table_name.upper()
                
                # 테이블 존재 여부 확인
                check_table_sql = "SELECT COUNT(*) FROM ALL_TABLES WHERE TABLE_NAME = :1"
                cursor = self.execute_sql(check_table_sql, (table_name,))
                exists = cursor.fetchone()[0] > 0
                cursor.close()
                
                if not exists:
                    columns = []
                    for field in model._meta.sorted_fields:
                        column_name = field.column_name.upper()
                        
                        # Oracle 데이터 타입 매핑
                        if isinstance(field, IntegerField):
                            if field.primary_key:
                                column_type = "NUMBER PRIMARY KEY"
                            else:
                                column_type = "NUMBER"
                        elif isinstance(field, BigIntegerField):
                            column_type = "NUMBER(19)"
                        elif isinstance(field, CharField):
                            max_length = getattr(field, 'max_length', 255)
                            column_type = f"VARCHAR2({max_length})"
                        elif isinstance(field, TextField):
                            column_type = "CLOB"
                        elif isinstance(field, DateField):
                            column_type = "DATE"
                        elif isinstance(field, BooleanField):
                            column_type = "NUMBER(1)"
                        elif isinstance(field, JSONField):
                            column_type = "CLOB"
                        else:
                            column_type = "CLOB"
                        
                        null = "" if field.null else " NOT NULL"
                        columns.append(f"{column_name} {column_type}{null}")
                    
                    # Oracle CREATE TABLE 문법
                    create_table_sql = "CREATE TABLE {} ({})".format(
                        table_name,
                        ", ".join(columns)
                    )
                    
                    logger.debug(f"Oracle CREATE TABLE SQL: {create_table_sql}")
                    self.execute_sql(create_table_sql)
                    
                    # 자동 커밋
                    if self._conn:
                        self._conn.commit()
                        
                    logger.info(f"테이블 '{table_name}' 생성 완료")

    if db_config.db_type == "postgresql":
        logger.info(f"데이터베이스 '{db_config.database_name}' 연결 준비 중...")
        try:
            database = PooledPostgresqlDatabase(
                db_config.database_name,
                user=db_config.username,
                password=db_config.password,
                host=db_config.host,
                port=db_config.port
            )
            logger.info(f"데이터베이스 '{db_config.database_name}' 연결 완료")
            return database
        except Exception as e:
            logger.error(f"데이터베이스 '{db_config.database_name}' 연결 실패: {e}")
            raise e
    
    elif db_config.db_type == "oracle":
        logger.info(f"데이터베이스 '{db_config.database_name}' 연결 준비 중...")
        try:
            dsn = f"{db_config.host}:{db_config.port}/{db_config.database_name}"
            database = OracleDatabase(
                db_config.database_name,
                user=db_config.username,
                password=db_config.password,
                dsn=dsn
            )
            logger.info(f"데이터베이스 '{db_config.database_name}' 연결 완료")
            return database
        except Exception as e:
            logger.error(f"데이터베이스 '{db_config.database_name}' 연결 실패: {e}")
            raise e
    
    else:
        raise ValueError(f"'{db_config.db_type}'은 지원하지 않는 데이터베이스이거나 잘못된 설정입니다.")

source_db = get_database(config.source_db)
pgvector_db = get_database(config.pgvector_db)

# 비동기 데이터베이스 설정
manager_source = peewee_async.Manager(source_db)
manager_pgvector = peewee_async.Manager(pgvector_db)