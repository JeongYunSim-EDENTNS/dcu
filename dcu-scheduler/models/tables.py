from peewee import *
from playhouse.postgres_ext import ArrayField, JSONField

from config import config
from utils.database import get_database


sourceTableName = config.source_db.table_name
sourceSchemaName = config.source_db.schema_name
pgvectorTableName = config.pgvector_db.table_name
pgvectorSchemaName = config.pgvector_db.schema_name


class BaseModel_source(Model):
    class Meta:
        database = get_database(config.source_db)
        schema = sourceSchemaName if config.source_db.db_type == 'postgresql' else None

class BaseModel_pgvector(Model):
    class Meta:
        database = get_database(config.pgvector_db)
        schema = pgvectorSchemaName


# PostgreSQL용 테이블 정의
class PostgresSourceTable(BaseModel_source):
    """원본 데이터베이스 테이블 (PostgreSQL)"""
    
    idx = IntegerField(primary_key=True)
    code = CharField(max_length=50)
    subject = CharField(max_length=200)
    userid = CharField(max_length=16)
    name = CharField(max_length=50)
    content = TextField()
    sdate = DateField()
    
    class Meta:
        schema = sourceSchemaName
        table_name = sourceTableName

# Oracle용 테이블 정의
class OracleSourceTable(BaseModel_source):
    """원본 데이터베이스 테이블 (Oracle)"""
    
    idx = IntegerField(primary_key=True)
    code = CharField(max_length=50)
    subject = CharField(max_length=200)
    userid = CharField(max_length=16)
    name = CharField(max_length=50)
    content = TextField()  # Oracle에서는 CLOB으로 매핑
    sdate = DateField()
    
    class Meta:
        table_name = sourceTableName.upper()  # Oracle은 대문자 사용

# 데이터베이스 타입에 따라 적절한 테이블 클래스 선택
SourceTable = OracleSourceTable if config.source_db.db_type == 'oracle' else PostgresSourceTable

class VectorField(Field):
    field_type = 'vector'

    def __init__(self, dimension, *args, **kwargs):
        self.dimension = dimension
        super().__init__(*args, **kwargs)

    def db_value(self, value):
        return value

    def python_value(self, value):
        return value

class DCUEmbeddingTable(BaseModel_pgvector):
    """DCUEmbedding 테이블"""
    
    id = BigIntegerField(primary_key=True)
    original_idx = IntegerField(default=0)
    content = TextField()
    embedding = VectorField(dimension=768)
    metadata = JSONField()
    
    class Meta:
        schema = pgvectorSchemaName
        table_name = pgvectorTableName
        
        
class SchedulerConfigTable(BaseModel_pgvector):
    """스케줄러 설정 테이블"""
    
    watcher_active = BooleanField(default=True)

    class Meta:
        schema = pgvectorSchemaName
        table_name = "scheduler_config"