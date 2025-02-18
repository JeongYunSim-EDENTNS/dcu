import asyncio

from config import config
from utils.logger import get_logger
from utils.console import bold, invert


# Logger 설정
logger = get_logger(name="retry", log_dir="logs")

async def retry_task(task, notice, embeder):
    """
    지정된 작업을 재시도합니다.
    :param task_func: 재시도할 작업 함수
    :param args: 작업 함수의 위치 인자
    :param kwargs: 작업 함수의 키워드 인자
    """
    job_id = f"retry_task:{notice.idx}"
    retry_count = config.scheduler.retry_count

    for attempt in range(1, retry_count + 1):
        try:
            logger.info(f"{invert(f' {job_id} ')} {bold(f'작업 재시도 {attempt}/{retry_count} 시작')}")
            await task(notice, embeder)
            logger.info(f"{invert(f' {job_id} ')} {bold('작업 재시도 성공')}")
            return
        except Exception as e:
            logger.error(f"{invert(f' {job_id} ')} {bold(f'작업 재시도 {attempt}/{retry_count} 실패')} / {e}")
            if attempt == retry_count:
                logger.error(f"{invert(f' {job_id} ')} {bold('최대 재시도 횟수 초과. 작업이 실패했습니다.')}")
                raise e

