from typing import Literal, Dict

from pydantic import BaseModel


class SchedulerConfig(BaseModel):
    """스케줄러 관련 설정 클래스"""

    refresh_interval: int
    """스케줄 변경 감지 주기 (초)"""
    retry_interval: int
    """재시도 주기 (초)"""
    retry_count: int
    """재시도 횟수"""
    retry_delay_seconds: int
    """재시도 전 대기 시간 (초)"""
    coalesce: bool
    """누락된 스케줄 병합 여부, 누락된 스케줄이 있을 경우 병합하여 한 번만 실행"""
    max_instances: int
    """최대 인스턴스 수, 동시에 실행될 수 있는 스케줄의 최대 인스턴스 수"""
    misfire_grace_time: int
    """스케줄 실행 시간 초과 허용 시간 (초), 스케줄이 누락된 후 실행될 수 있는 최대 시간"""

    def __getitem__(self, item):
        return getattr(self, item)


class DatabaseConfig(BaseModel):
    """데이터베이스 관련 설정 클래스"""

    db_type: Literal["postgresql", "oracle"]
    """데이터베이스 타입"""
    host: str
    """서버 IP"""
    port: int
    """포트 번호"""
    username: str
    """사용자명"""
    password: str
    """사용자 비밀번호"""
    database_name: str
    """데이터베이스 이름"""
    schema_name: str
    """스키마 이름"""
    table_name: str
    """테이블 이름"""

    def __getitem__(self, item):
        return getattr(self, item)


class AuthConfig(BaseModel):
    """인증 관련 설정 클래스"""
    server_url: str
    """인증 서버 주소"""
    realm_name: str
    """Realm 이름"""
    client_id: str
    """Client ID"""
    client_secret_key: str
    """Client Secret Key"""
    username: str
    """사용자명"""
    password: str
    """사용자 비밀번호"""

    def __getitem__(self, item):
        return getattr(self, item)


class EmbeddingConfig(BaseModel):
    """임베딩 관련 설정 클래스"""

    model_url: str
    """임베딩 API URL"""
    name: str
    """임베딩 모델 이름"""
    max_text_length: int
    """임베딩 처리 시 최대 텍스트 길이"""
    api_key: str | None
    """API 인증 키"""
    auth: AuthConfig
    """인증 관련 설정"""

    def __getitem__(self, item):
        return getattr(self, item)

class AttachmentsConfig(BaseModel):
    """첨부파일 관련 서버 설정 클래스"""
    host: str
    """서버 주소"""
    port: int
    """포트 번호"""
    username: str
    """사용자명"""
    password: str
    """사용자 비밀번호"""
    folder_path: str
    """첨부파일 경로"""

    def __getitem__(self, item):
        return getattr(self, item)

class Config(BaseModel):
    """전체 설정 클래스"""

    scheduler: SchedulerConfig
    """스케줄러 관련 설정"""
    source_db: DatabaseConfig
    """원본 데이터베이스 관련 설정"""
    pgvector_db: DatabaseConfig
    """pgvector 데이터베이스 관련 설정"""
    embedding: EmbeddingConfig
    """임베딩 관련 설정"""
    attachments: AttachmentsConfig
    """첨부파일 관련 서버 설정"""

    def __getitem__(self, item):
        return getattr(self, item)
