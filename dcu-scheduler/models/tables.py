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
    """게시글 인덱스"""
    code = CharField(max_length=50)
    """게시판 코드"""
    subject = CharField(max_length=200)
    """게시글 제목"""
    name = CharField(max_length=50)
    """작성자 이름"""
    content = TextField()
    """게시글 내용"""
    memo1 = TextField()
    """게시글 내용 추가 정보"""
    bbs_file0 = TextField()
    """첨부파일 0"""
    bbs_file1 = TextField()
    """첨부파일 1"""
    bbs_file2 = TextField()
    """첨부파일 2"""
    bbs_file3 = TextField()
    """첨부파일 3"""
    bbs_file4 = TextField()
    """첨부파일 4"""
    sdate = DateField()
    """작성일"""
    
    class Meta:
        schema = sourceSchemaName
        table_name = sourceTableName

# Oracle용 테이블 정의
class OracleSourceTable(BaseModel_source):
    """원본 데이터베이스 테이블 (Oracle)"""
    
    idx = IntegerField(primary_key=True)
    """게시글 인덱스"""
    code = CharField(max_length=50)
    """게시판 코드"""
    subject = CharField(max_length=200)
    """게시글 제목"""
    name = CharField(max_length=50)
    """작성자 이름"""
    content = TextField()  # Oracle에서는 CLOB으로 매핑
    """게시글 내용"""
    memo1 = TextField() 
    """게시글 내용 추가 정보"""
    bbs_file0 = TextField()
    """첨부파일 0"""
    bbs_file1 = TextField()
    """첨부파일 1"""
    bbs_file2 = TextField()
    """첨부파일 2"""
    bbs_file3 = TextField()
    """첨부파일 3"""
    bbs_file4 = TextField()
    """첨부파일 4"""
    sdate = DateField()
    """작성일"""
    
    class Meta:
        table_name = sourceTableName.upper() 

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

class NoticeEmbeddingTable(BaseModel_pgvector):
    """게시판 데이터 임베딩 테이블"""
    
    id = BigIntegerField(primary_key=True)
    """게시판 데이터 인덱스"""
    original_idx = IntegerField(default=0)
    """원본 게시판 데이터 인덱스"""
    content = TextField()
    """게시판 데이터 내용"""
    embedding = VectorField(dimension=768)
    """임베딩 벡터"""
    metadata = JSONField()
    """메타데이터"""
    created_at = DateTimeField(default=datetime.datetime.now)
    """생성일"""
    updated_at = DateTimeField(default=datetime.datetime.now)
    """수정일"""
    
    class Meta:
        schema = pgvectorSchemaName
        table_name = pgvectorTableName
        
class AttachmentEmbeddingTable(BaseModel_pgvector):
    """첨부파일 데이터 임베딩 테이블"""
    
    id = AutoField(primary_key=True)
    """고유 ID"""
    embedding_notices_id = ForeignKeyField(NoticeEmbeddingTable, on_delete='CASCADE')
    """embedding_notices 테이블의 ID (FK)"""
    filename_org = CharField()
    """원본 파일명"""
    filename_converted = CharField()
    """변환된 파일명"""
    embedding = VectorField(dimension=768)
    """벡터 임베딩 데이터"""
    content = TextField(null=True)
    """첨부파일 내용"""
    metadata = JSONField(null=True)
    """추가 메타데이터"""
    created_at = DateTimeField(default=datetime.datetime.now)
    """생성 시간"""
    updated_at = DateTimeField(default=datetime.datetime.now)
    """업데이트 시간"""
    
    class Meta:
        schema = pgvectorSchemaName
        table_name = "embedding_attachments"

class SchedulerConfigTable(BaseModel_pgvector):
    """스케줄러 설정 테이블"""
    
    watcher_active = BooleanField(default=True)

    class Meta:
        schema = pgvectorSchemaName
        table_name = "scheduler_config"