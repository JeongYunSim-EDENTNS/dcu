import asyncio
from config import config
from models.tables import SourceTable, DCUEmbeddingTable, SchedulerConfigTable
from utils.database import source_db, pgvector_db, manager_source, manager_pgvector
from utils.logger import get_logger
from utils.common import extract_text_from_html
from utils.embeder import create_embedding
from utils.console import bold, invert
from service.retry import retry_task

# Logger 설정
logger = get_logger(name="task", log_dir="logs")

async def get_last_checked_id():
    """
    DCUEmbedding 테이블에서 초기 last_checked_id를 가져옵니다.
    """
    job_id = "get_initial_last_checked_id"
    try:
        # 가장 큰 ID 가져오기
        query = DCUEmbeddingTable.select().order_by(DCUEmbeddingTable.original_idx.desc()).limit(1)
        result = await manager_pgvector.execute(query)
        if result:
            logger.info(f"{invert(f' {job_id} ')} {bold('last_checked_id : ' + str(result[0].original_idx))}")
            return result[0].id
        else:
            logger.info(f"{invert(f' {job_id} ')} {bold('DCUEmbedding 테이블이 비어 있습니다. 초기 ID는 0으로 설정됩니다.')}")
            return 0
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('초기 last_checked_id 가져오는 중 오류 발생')} / {e}")
        raise e


async def fetch_new_notices(last_checked_id):
    """
    공지사항 테이블에서 새로운 데이터를 가져옵니다.
    :param last_checked_id: 마지막으로 확인한 공지사항 ID
    :return: 새로운 공지사항 리스트
    """
    job_id = "fetch_new_notices"
    try:
        if config['source_db']['db_type'] == 'oracle':
            schema = config.source_db.schema_name.upper() if config.source_db.schema_name else None
            table_name = config.source_db.table_name.upper()
            
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            
            sql = f"""
                SELECT 
                    IDX, CODE, SUBJECT, USERID, NAME, CONTENT, SDATE 
                FROM {full_table_name} 
                WHERE IDX > :1 
                ORDER BY IDX ASC
            """
            bind_vars = [last_checked_id]            
            conn = await source_db.connect_async()
            cursor = conn.cursor()
            try:
                cursor.execute(sql, bind_vars)
                rows = cursor.fetchall()
                
                # 결과를 SourceTable 모델 인스턴스로 변환
                result = []
                if rows:
                    for row in rows:
                        notice = SourceTable()
                        for i, field in enumerate(SourceTable._meta.sorted_fields):
                            setattr(notice, field.name, row[i])
                        result.append(notice)
                return result
            finally:
                cursor.close()
        else:
            # PostgreSQL용 쿼리 실행
            query = (SourceTable
                    .select()
                    .where(SourceTable.idx > last_checked_id)
                    .order_by(SourceTable.idx.asc()))
            new_notices = await manager_source.execute(query)
            return list(new_notices)
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('공지사항 데이터 조회 중 오류 발생')} / {e}")
        raise e


async def embed_and_save_notice(notice, custom_embeder):
    """
    공지사항 데이터를 임베딩 후 DCUEmbedding 테이블에 저장합니다.
    :param notice: 공지사항 객체
    :param custom_embeder: CustomEmbeddings 객체
    """
    job_id = f"embed_and_save_notice:{notice.idx}" 
    try:
        logger.info(f"{invert(f' {job_id} ')} {bold(f'ID={notice.idx}, 제목={notice.subject}')}")

        # 텍스트 데이터를 결합 (예: 제목 + 내용)
        content = f"{notice.subject} {extract_text_from_html(notice.content)}"

        # 텍스트를 max_text_length 기준으로 분할
        chunks = [
            content[i:i + custom_embeder.embedding_ctx_length]
            for i in range(0, len(content), custom_embeder.embedding_ctx_length)
        ]

        for chunk_index, chunk in enumerate(chunks):
            # 텍스트 임베딩 생성
            embedding = await custom_embeder.aembed_query(chunk)

            # DCUEmbedding 테이블에 데이터 삽입
            query = DCUEmbeddingTable.insert(
                embedding=embedding,
                content=chunk,
                metadata={
                    "source": "notice_board",
                    "idx": notice.idx,
                    "chunk": chunk_index + 1,
                    "total_chunks": len(chunks),
                },
                original_idx=notice.idx,  # 공지사항 원본 ID
            )
            await manager_pgvector.execute(query)
            
            logger.info(f"{invert(f' {job_id} ')} {bold(f'original_idx={notice.idx}, chunk={chunk_index + 1} 저장 완료')}")
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('공지사항 처리 중 오류 발생')} / original_idx={notice.idx}, 오류={e}")
        raise e


async def process_notice(notice, custom_embeder):
    """
    공지사항 데이터를 처리하며 실패 시 재시도합니다.
    :param notice: 공지사항 객체
    :param custom_embeder: CustomEmbeddings 객체
    """
    job_id = f"process_notice:{notice.idx}"
    logger.info(f"{invert(f' {job_id} ')}")

    try:
        await embed_and_save_notice(notice, custom_embeder)
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('공지사항 처리 중 오류 발생')} / {e}")
        await retry_task(embed_and_save_notice, notice, custom_embeder)


async def process_new_notices(new_notices):
    """
    새로운 공지사항 데이터를 임베딩 후 저장합니다.
    :param new_notices: 공지사항 리스트
    """
    job_id = "process_new_notices"
    
    for notice in new_notices:
        try:
            logger.info(f"{invert(f' {job_id} ')} {notice}")
            custom_embeder = create_embedding(config["embedding"])
            await process_notice(notice, custom_embeder)
            logger.info(f"{invert(f' {job_id} ')} {bold('신규 공지사항 처리 완료')}")
            
        except Exception as e:
            logger.error(f"{invert(f' {job_id} ')} {bold('공지사항 처리 중 오류 발생')} / ID={notice.idx}, 오류={e}")


async def run_task():
    """
    신규 공지사항 데이터를 처리하는 메인 작업 함수.
    """ 
    job_id = "run_task"
    try:        
        # 마지막으로 확인한 데이터 ID
        last_checked_id = await get_last_checked_id()
        
        logger.info(f"{invert(f' {job_id} ')} {bold(f'신규 공지사항 데이터를 확인합니다...(last_checked_id={last_checked_id})')}")
        new_notices = await fetch_new_notices(last_checked_id)

        if new_notices:
            logger.info(f"{invert(f' {job_id} ')} {bold(f'{len(new_notices)}개의 신규 데이터를 발견했습니다.')}")
            
            # watcher_active 상태 False로 변경
            update_values = {"watcher_active": False}
            query = SchedulerConfigTable.update(**update_values)
            await manager_pgvector.execute(query)

            await process_new_notices(new_notices)
            
        else:
            logger.info(f"{invert(f' {job_id} ')} {bold('신규 데이터가 없습니다.')}")
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('작업 실행 중 오류 발생')} / {e}")
        raise e
    finally:
        # watcher_active 상태 True로 변경
        update_values = {"watcher_active": True}
        query = SchedulerConfigTable.update(**update_values)
        await manager_pgvector.execute(query)
