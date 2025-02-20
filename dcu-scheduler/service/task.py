import os
import datetime

from pathlib import Path
import asyncio
import paramiko
import tempfile
import io

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
            
            # 필드 목록 생성
            fields = []
            field_names = []
            for field in SourceTable._meta.sorted_fields:
                if hasattr(field, 'column_name'):
                    fields.append(field)
                    field_names.append(field.column_name.upper())
            
            # Oracle용 직접 쿼리 작성
            sql = f"""
                SELECT 
                    {', '.join(field_names)}
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
                        for i, field in enumerate(fields):
                            if i < len(row):  # 인덱스 범위 체크
                                value = row[i]
                                if value is not None:  # None이 아닌 경우에만 값 설정
                                    setattr(notice, field.name, value)
                        result.append(notice)
                        logger.debug(f"Converted row: {vars(notice)}")  # 변환된 객체 로깅
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
    공지사항 데이터를 임베딩하고 저장한 후, 첨부파일 목록을 반환합니다.
    :param notice: 공지사항 객체
    :param custom_embeder: CustomEmbeddings 객체
    :return: tuple - 공지사항 ID와 첨부파일 정보가 담긴 리스트
    """
    job_id = f"embed_and_save_notice:{notice.idx}" 
    try:
        logger.info(f"{invert(f' {job_id} ')} {bold(f'ID={notice.idx}, 제목={notice.subject}')}")

        # 텍스트 데이터를 결합 (예: 제목 + 내용)
        content = f"{notice.subject} {extract_text_from_html(notice.content)} {notice.memo1}"

        # 텍스트를 max_text_length 기준으로 분할
        chunks = [
            content[i:i + custom_embeder.embedding_ctx_length]
            for i in range(0, len(content), custom_embeder.embedding_ctx_length)
        ]

        for chunk_index, chunk in enumerate(chunks):
            # 텍스트 임베딩 생성
            embedding = await custom_embeder.aembed_query(chunk)

            # 각 데이터에서 none이 아닌 경우에  "/" 기준 뒤의 데이터(실제 파일명)을 추출
            # 첨부 파일 목록 생성
            attachments = []
            file_fields = [notice.bbs_file0, notice.bbs_file1, notice.bbs_file2, notice.bbs_file3, notice.bbs_file4]
            
            for file_field in file_fields:
                if file_field is not None:
                    file_name = file_field.split('/')[-1]
                    file_type = file_name.split('.')[-1]
                    file_name_converted = f"{file_field.split('/')[0]}.{file_type}"
                    attachments.append({
                        "name_org": file_name,
                        "name_converted": file_name_converted
                        })
                    
            # DCUEmbedding 테이블에 데이터 삽입
            query = DCUEmbeddingTable.insert(
                embedding=embedding,
                content=chunk,
                metadata={
                    "board_name": notice.code,
                    "subject": notice.subject,
                    "writer": notice.name,
                    "attachment" : attachments,
                    "chunk": chunk_index + 1,
                    "total_chunks": len(chunks)
                },
                original_idx=notice.idx,  # 공지사항 원본 ID
            )
            await manager_pgvector.execute(query)
            
            logger.info(f"{invert(f' {job_id} ')} {bold(f'original_idx={notice.idx}, chunk={chunk_index + 1} 저장 완료')}")
            
            return notice.idx, attachments

    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('공지사항 처리 중 오류 발생')} / original_idx={notice.idx}, 오류={e}")
        raise e

async def embed_and_save_attachment(notice_idx, attachment_info, custom_embeder):
    """
    첨부파일 목록을 임베딩하고 저장합니다.
    :param notice_idx: 공지사항 ID
    :param attachment_info: dict - 첨부파일 정보
    :param custom_embeder: CustomEmbeddings 객체
    """
    job_id = f"embed_and_save_attachment:{notice_idx}_{attachment_info['name_org']}"
    
    logger.info(f"{invert(f' {job_id} ')} {bold(f'첨부파일 임베딩 및 저장 시작')}")
    
    # SSH 클라이언트 설정
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # SSH 연결
        ssh.connect(
            hostname=config['attachments']['host'],
            port=config['attachments']['port'],
            username=config['attachments']['username'],
            password=config['attachments']['password']
        )
        
        # SFTP 세션 생성
        sftp = ssh.open_sftp()
        
        try:
            # 원격 파일 경로
            remote_path = Path(config['attachments']['folder_path']) / attachment_info['name_converted']
            
            # 파일 존재 여부 및 크기 확인
            try:
                file_attr = sftp.stat(str(remote_path))
                file_size = file_attr.st_size
                logger.info(f"{invert(f' {job_id} ')} {bold(f'파일 존재 확인: {remote_path} (크기: {file_size} bytes)')}")
                
                # 파일 전체 읽기
                with io.BytesIO() as bio:
                    with sftp.open(str(remote_path), 'rb') as remote_file:
                        # 파일을 청크 단위로 읽어서 처리
                        chunk_size = 8 * 1024 * 1024  # 8MB 청크
                        while True:
                            chunk = remote_file.read(chunk_size)
                            if not chunk:
                                break
                            bio.write(chunk)
                    
                    bio.seek(0)
                    content = bio.read()
                
                # 파일 정보 및 메타데이터 생성
                file_info = {
                    "name": attachment_info['name_org'],
                    "converted_name": attachment_info['name_converted'],
                    "size": file_size,
                    "path": str(remote_path),
                    "mime_type": attachment_info['name_org'].split('.')[-1].lower()  # 파일 확장자
                }
                
                # 텍스트 컨텐츠 생성 (파일 메타데이터)
                metadata = (
                    f"file: {file_info['name']}\n"
                    f"변환된 파일명: {file_info['converted_name']}\n"
                    f"파일 크기: {file_info['size']} bytes\n"
                    f"파일 경로: {file_info['path']}\n"
                    f"파일 형식: {file_info['mime_type']}"
                )
                
                # 텍스트 임베딩 생성
                embedding = await custom_embeder.aembed_query(text_content)
                
                # DCUEmbedding 테이블에 데이터 삽입
                query = DCUEmbeddingTable.insert(
                    embedding=embedding,
                    content=text_content,
                    metadata={
                        "type": "attachment",
                        "original_idx": notice_idx,
                        "file_info": file_info,
                        "content_size": len(content),
                        "processed_at": datetime.datetime.now().isoformat()
                    }
                )
                await manager_pgvector.execute(query)
                
                logger.info(f"{invert(f' {job_id} ')} {bold(f'첨부파일 임베딩 저장 완료')}")
                
            except FileNotFoundError:
                logger.warning(f"{invert(f' {job_id} ')} {bold(f'파일이 존재하지 않습니다: {remote_path}')}")
                return
                
        finally:
            sftp.close()
            
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('첨부파일 처리 중 오류 발생')} / {e}")
        raise e
    
    finally:
        ssh.close()

async def process_notice(notice, custom_embeder):
    """
    공지사항 데이터를 처리하며 실패 시 재시도합니다.
    :param notice: 공지사항 객체
    :param custom_embeder: CustomEmbeddings 객체
    """
    job_id = f"process_notice:{notice.idx}"
    logger.info(f"{invert(f' {job_id} ')}")

    try:
        # 공지사항 데이터 임베딩 및 저장
        notice_idx, attachments = await embed_and_save_notice(notice, custom_embeder)
    except Exception as e:
        logger.error(f"{invert(f' {job_id} ')} {bold('공지사항 처리 중 오류 발생')} / {e}")
        await retry_task(embed_and_save_notice, notice, custom_embeder)
        
    # 첨부파일 목록 임베딩 및 저장
    for attachment_info in attachments:
        try:
            await embed_and_save_attachment(notice_idx, attachment_info, custom_embeder)
        except Exception as e:
            logger.error(f"{invert(f' {job_id} ')} {bold('첨부파일 처리 중 오류 발생')} / {e}")
            await retry_task(embed_and_save_attachment, notice_idx, attachment_info, custom_embeder)
        



async def process_new_notices(new_notices):
    """
    새로운 공지사항 데이터를 임베딩 후 저장합니다.
    :param new_notices: 공지사항 리스트
    """
    job_id = "process_new_notices"
    
    for notice in new_notices:
        try:
            custom_embeder = create_embedding(config["embedding"])
            await process_notice(notice, custom_embeder)
            logger.info(f"{invert(f' {job_id} ')} {bold('신규 공지사항 처리 완료 original_idx={notice.idx}')}")
            
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
