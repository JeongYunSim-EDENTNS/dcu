import os
import yaml
from models.config import Config, SchedulerConfig, DatabaseConfig, EmbeddingConfig
from utils.logger import refresh_log_level


CONFIG_FILE = "app_config.yml"  # 설정 파일 경로


def get_config() -> Config:
    """YAML 파일을 읽어 Config 객체로 변환"""

    # 설정 파일 존재 여부 확인
    if not os.path.isfile(CONFIG_FILE):
        raise FileNotFoundError(f"'{CONFIG_FILE}' 설정 파일을 찾을 수 없습니다.")

    # YAML 설정 파일 읽기
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # 환경 변수 설정
    environment_config = config.get("environment", {})
    os.environ["LOG_LEVEL"] = environment_config.get("log_level", "INFO")
    refresh_log_level()

    # Config 객체 생성
    return Config(
        scheduler=SchedulerConfig(**config["scheduler"]),
        embedding=EmbeddingConfig(**config["embedding"]),
        source_db=DatabaseConfig(**config["databases"]["source_db"]),
        pgvector_db=DatabaseConfig(**config["databases"]["pgvector_db"]),
        attachments=AttachmentsConfig(**config["attachments"])
    )


config = get_config()
    