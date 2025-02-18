import asyncio
import service.scheduler

from config import config
from models.tables import (
    SourceTable, 
    DCUEmbeddingTable, 
    SchedulerConfigTable
)
from utils.logger import get_logger
from utils.database import source_db, pgvector_db


logger = get_logger(__name__, log_dir="logs")


async def main():
    logger.debug(f"> Config: {config}")
    
    # Database
    logger.debug(f"source_db config: {config.source_db}")
    logger.debug(f"pgvector_db config: {config.pgvector_db}")
    
    try:
        # 소스 데이터베이스 테이블 생성
        if config.source_db.db_type == 'oracle':
            source_db.create_tables([SourceTable])
        else:
            source_db.create_tables([SourceTable], safe=True)
            
        # PGVector 데이터베이스 테이블 생성
        pgvector_db.create_tables([
            DCUEmbeddingTable,
            SchedulerConfigTable
        ], safe=True)
            
    except Exception as e:
        logger.error(f"데이터베이스 테이블 확인 및 생성 중 오류 발생: {e}")
    finally:
        if hasattr(source_db, 'close_async'):
            await source_db.close_async()
        if hasattr(pgvector_db, 'close_async'):
            await pgvector_db.close_async()

    try:
        await service.scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        service.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
